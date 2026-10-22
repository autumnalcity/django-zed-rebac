"""S1 caches facts/templates, never decisions or live backing."""

from unittest.mock import patch

import pytest
from django.db import connection, transaction
from django.db.models import Value
from django.test import override_settings
from django.test.utils import CaptureQueriesContext

from rebac import LocalBackend, ObjectRef, RelationshipTuple, SubjectRef, evaluator_scope
from rebac.backends.local import _schema_operation
from rebac.backends.local_query import LocalQueryScope, UnsupportedScope
from rebac.schema import parse_zed
from tests.testapp.models import Post

pytestmark = pytest.mark.django_db(transaction=True)
ACTOR = SubjectRef.of("auth/user", "alice")
OTHER = SubjectRef.of("auth/user", "bob")
RESOURCE = ObjectRef("blog/post", "1")
SCHEMA = """
definition auth/user {}
definition auth/group { relation member: auth/user }
definition blog/post {
 relation owner: auth/user | auth/group#member
 relation viewer: auth/user | auth/group#member
 permission edit = owner
 permission read = (edit + viewer) & (edit + viewer)
}
"""


@pytest.fixture(params=["denormalized", "registry"])
def local(request):
    with override_settings(REBAC_LOCAL_BACKEND_STORAGE=request.param):
        engine = LocalBackend()
        engine.set_schema(parse_zed(SCHEMA))
        engine.write_relationships([RelationshipTuple(RESOURCE, "owner", ACTOR)])
        yield engine


def test_one_fact_selection_per_relation_per_operation(local):
    @_schema_operation
    def two_checks(engine):
        first = engine.check_access(subject=ACTOR, resource=RESOURCE, action="read")
        second = engine.check_access(subject=OTHER, resource=RESOURCE, action="read")
        return first.allowed, second.allowed

    with CaptureQueriesContext(connection) as capture:
        assert two_checks(local) == (True, False)
    tables = ("rebac_relationshipregistry", "rebac_relationship")
    statements = [q["sql"] for q in capture if any('FROM "' + t + '"' in q["sql"] for t in tables)]
    assert len(statements) == 1  # registry once, not one per branch/actor


def test_facts_do_not_cross_calls_or_rollback(local):
    grant = RelationshipTuple(RESOURCE, "owner", ACTOR)

    def allowed():
        return local.check_access(subject=ACTOR, resource=RESOURCE, action="read").allowed

    with evaluator_scope() as evaluator:
        with transaction.atomic():
            assert allowed()
            assert local._cache_generation("blog/post") is None
            local.delete_relationship(grant)
            assert not allowed()
            transaction.set_rollback(True)
        assert allowed()
        with transaction.atomic():
            sid = transaction.savepoint()
            local.delete_relationship(grant)
            assert not allowed()
            transaction.savepoint_rollback(sid)
            assert allowed()
        evaluator.invalidate()
        assert allowed()


def test_predicate_template_is_independent_and_lazy(local):
    scope = LocalQueryScope(local, ACTOR, "default")
    with patch.object(scope, "relation", wraps=scope.relation) as compiler:
        first = scope.predicate(Post, "read", "blog/post")
        calls = compiler.call_count
        second = scope.predicate(Post, "read", "blog/post")
        assert compiler.call_count == calls
        assert first is not second
        original = Post._base_manager.filter(second).query.sql_with_params()
        first.children.clear()
        assert Post._base_manager.filter(second).query.sql_with_params() == original
    # Executing a previously compiled predicate sees later grants/revocations.
    # Post id 1 need not exist; compare SQL evaluation to native fresh checks.
    local.delete_relationship(RelationshipTuple(RESOURCE, "owner", ACTOR))
    assert not local.check_access(subject=ACTOR, resource=RESOURCE, action="read").allowed
    assert not Post._base_manager.filter(second).exists()


def test_template_keys_include_recursion_actor_model_and_identity(local):
    one = LocalQueryScope(local, ACTOR, "default")
    two = LocalQueryScope(local, OTHER, "default")
    args = ("blog/post", "edit", Post, "pk")
    one.permission(*args, frozenset())
    old = len(one.memo)
    one.permission(*args, frozenset({("other/type", "read")}))
    assert len(one.memo) > old
    one.permission("blog/post", "edit", None, Value("1"), frozenset())
    assert len(one.memo) > old + 1
    assert one.memo is not two.memo and two.subject != one.subject
    with pytest.raises(UnsupportedScope):
        one.permission(*args, frozenset({("blog/post", "edit")}))


def test_expiring_predicates_do_not_reuse_clock(local):
    local.set_schema(
        parse_zed("""definition auth/user {}
    definition blog/post { relation owner: auth/user with expiration
      permission read = owner }""")
    )
    scope = LocalQueryScope(local, ACTOR, "default")
    scope.predicate(Post, "read", "blog/post")
    assert not scope.memo


def test_nested_raw_write_clears_facts(local):
    from rebac.models import active_relationship_model

    @_schema_operation
    def change(engine):
        assert engine.check_access(subject=ACTOR, resource=RESOURCE, action="edit").allowed
        table = connection.ops.quote_name(active_relationship_model()._meta.db_table)
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM " + table)
        assert not engine.check_access(subject=ACTOR, resource=RESOURCE, action="edit").allowed

    change(local)


def test_templates_reuse_without_decisions_or_stale_backing(local):
    with evaluator_scope() as evaluator:
        first = LocalQueryScope(local, ACTOR, "default")
        first.predicate(Post, "read", "blog/post")
        second = LocalQueryScope(local, ACTOR, "default")
        assert second.memo is first.memo
        with patch.object(second, "relation", side_effect=AssertionError("recompiled")):
            second.predicate(Post, "read", "blog/post")
        assert LocalQueryScope(local, OTHER, "default").memo is not first.memo
        previous = first.memo
        local.set_schema(parse_zed(SCHEMA))
        assert LocalQueryScope(local, ACTOR, "default").memo is not previous
        evaluator.invalidate()
        assert not evaluator._predicate_templates
