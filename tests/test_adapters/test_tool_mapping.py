"""Tests for adapters/tool_mapping.py.

These run against the live retail domain, which is the point: this module is
where the harness's assumptions meet tau2, so the assertions are about what tau2
actually provides rather than about a fixture that could drift from it.
"""

import json

import pytest
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.registry import registry

from policygate.adapters.tool_mapping import (
    RETAIL_ESCALATION,
    RETAIL_IDENTITY_LOOKUPS,
    retail_constraints,
    retail_facts,
    to_harness_call,
    tool_specs,
)
from policygate.harness.types import ToolKind, ToolRole


@pytest.fixture(scope="module")
def toolkit():
    return registry.get_env_constructor("retail")().tools


@pytest.fixture(scope="module")
def retail_db():
    from tau2.utils.utils import DATA_DIR

    return json.loads((DATA_DIR / "tau2" / "domains" / "retail" / "db.json").read_text())


@pytest.fixture(scope="module")
def catalogue(toolkit):
    return tool_specs(
        toolkit, identity_lookups=RETAIL_IDENTITY_LOOKUPS, escalation=RETAIL_ESCALATION
    )


class TestCatalogue:
    def test_every_tool_in_the_domain_is_present(self, toolkit, catalogue):
        from tau2.environment.toolkit import get_tool_types

        assert {s.name for s in catalogue} == set(get_tool_types(toolkit))

    def test_kinds_come_from_tau2_not_from_a_local_list(self, toolkit, catalogue):
        from tau2.environment.toolkit import get_tool_types

        types = get_tool_types(toolkit)
        assert all(s.kind.value == types[s.name].value for s in catalogue)

    def test_the_writes_are_the_ones_tau2_calls_writes(self, catalogue):
        from tau2.metrics.break_down_metrics import get_write_tools

        agent_writes, _ = get_write_tools("retail")
        assert {s.name for s in catalogue if s.kind is ToolKind.WRITE} == agent_writes

    def test_identity_lookups_are_marked_although_tau2_types_them_as_reads(self, catalogue):
        marked = {s.name for s in catalogue if s.role is ToolRole.IDENTITY_LOOKUP}
        assert marked == set(RETAIL_IDENTITY_LOOKUPS)
        assert all(s.kind is ToolKind.READ for s in catalogue if s.name in marked)

    def test_escalation_is_marked_although_tau2_types_it_as_generic(self, catalogue):
        escalation = next(s for s in catalogue if s.role is ToolRole.ESCALATION)
        assert escalation.name == "transfer_to_human_agents"
        assert escalation.kind is ToolKind.GENERIC

    def test_a_tool_cannot_hold_two_roles(self, toolkit):
        with pytest.raises(ValueError, match="both identity lookup and escalation"):
            tool_specs(
                toolkit,
                identity_lookups=["transfer_to_human_agents"],
                escalation=["transfer_to_human_agents"],
            )

    def test_declaring_a_role_for_an_absent_tool_is_refused(self, toolkit):
        # Catches a rename in tau2 at construction rather than as a silently
        # ungated tool at run time.
        with pytest.raises(ValueError, match="not in the toolkit"):
            tool_specs(toolkit, identity_lookups=["find_user_id_by_telepathy"])


class TestCallConversion:
    def test_the_fields_the_harness_judges_on_survive(self):
        call = to_harness_call(
            Tau2ToolCall(id="c1", name="cancel_pending_order", arguments={"order_id": "#W1"})
        )
        assert (call.id, call.name) == ("c1", "cancel_pending_order")
        assert call.arguments == {"order_id": "#W1"}

    def test_missing_arguments_become_an_empty_mapping(self):
        assert to_harness_call(Tau2ToolCall(id="c", name="x", arguments={})).arguments == {}


class TestRetailFacts:
    def test_an_identity_lookup_yields_the_user_id(self):
        assert retail_facts("find_user_id_by_email", "mia_garcia_4516").user_id == "mia_garcia_4516"

    def test_a_failed_call_establishes_nothing(self):
        # "Order not found" must not put the order into the world view, or the
        # referent check would accept the very ids it exists to catch.
        facts = retail_facts("get_order_details", "Error: Order not found", error=True)
        assert facts == retail_facts("anything", None)

    def test_the_user_profile_lists_orders_without_choosing_one(self):
        facts = retail_facts(
            "get_user_details", json.dumps({"user_id": "u1", "orders": ["#W1", "#W2"]})
        )
        assert {o.id for o in facts.objects} == {"#W1", "#W2"}
        assert facts.scope_targets is None

    def test_fetching_one_order_scopes_to_it(self):
        facts = retail_facts(
            "get_order_details", json.dumps({"order_id": "#W1", "status": "delivered"})
        )
        assert facts.scope_targets == ("#W1",)
        assert facts.objects[0].attributes["status"] == "delivered"

    def test_a_write_result_refreshes_what_is_known_about_the_order(self):
        facts = retail_facts(
            "cancel_pending_order", json.dumps({"order_id": "#W1", "status": "cancelled"})
        )
        assert facts.objects[0].attributes["status"] == "cancelled"

    def test_an_unparseable_result_is_tolerated(self):
        assert retail_facts("calculate", "4") == retail_facts("calculate", None)

    def test_a_real_order_round_trips(self, toolkit):
        order = toolkit.get_order_details(order_id="#W5490111")
        facts = retail_facts("get_order_details", order.model_dump_json())
        assert facts.scope_targets == ("#W5490111",)
        assert facts.objects[0].attributes["status"] == order.status


class TestRetailConstraints:
    def test_canonical_values_are_read_from_the_database(self, retail_db):
        constraints = retail_constraints(retail_db)
        by_field = {c.field: c for c in constraints.value_constraints}
        assert by_field["country"].allowed == frozenset({"USA"})
        assert all(len(s) == 2 for s in by_field["state"].allowed)

    def test_the_arguments_task_22_wrote_are_not_canonical(self, retail_db):
        by_field = {c.field: c for c in retail_constraints(retail_db).value_constraints}
        assert by_field["country"].canonical_for("United States") == "USA"
        assert by_field["state"].canonical_for("New York") == "NY"

    def test_preconditions_match_the_policy_text(self, retail_db):
        by_tool = {p.tool: p for p in retail_constraints(retail_db).preconditions}
        assert by_tool["cancel_pending_order"].allowed_values == frozenset({"pending"})
        assert by_tool["return_delivered_order_items"].allowed_values == frozenset({"delivered"})
        assert by_tool["exchange_delivered_order_items"].allowed_values == frozenset({"delivered"})

    def test_every_precondition_cites_the_policy(self, retail_db):
        preconditions = retail_constraints(retail_db).preconditions
        assert all(p.clause.doc_id == "retail/policy.md" for p in preconditions)

    def test_every_order_write_declares_its_referent_field(self, retail_db):
        from tau2.metrics.break_down_metrics import get_write_tools

        agent_writes, _ = get_write_tools("retail")
        declared = set(retail_constraints(retail_db).referent_fields)
        # modify_user_address acts on the user, not on an order.
        assert agent_writes - declared == {"modify_user_address"}

    def test_an_empty_database_yields_no_value_constraints(self):
        assert retail_constraints({}).value_constraints == ()
