"""Tests for harness/audit.py.

The chain is not in the ablation matrix -- it observes without changing
behaviour -- so it is verified against the three properties that make it
evidence rather than a log: completeness, linkage, and verifiable order.

No LLM, no benchmark, no network.
"""

from datetime import datetime, timedelta

from policygate.harness.audit import (
    AuditChain,
    BreakKind,
    verify_records,
    world_diff,
)
from policygate.harness.config import HarnessConfig
from policygate.harness.types import (
    GENESIS_HASH,
    ApprovalDecision,
    ApprovalStatus,
    AuditEvent,
    ClauseRef,
    Decision,
    DecisionOutcome,
    ToolCall,
)

START = datetime(2026, 1, 1, 12, 0, 0)

CANCEL = ToolCall(id="c1", name="cancel_pending_order", arguments={"order_id": "#W1"})
RETURN = ToolCall(id="c2", name="return_delivered_order_items", arguments={"order_id": "#W2"})
ALLOW = Decision(
    outcome=DecisionOutcome.ALLOW,
    rule="gating.allowed",
    reason="scoped",
    clauses=(ClauseRef(doc_id="retail/policy.md", clause="cancel"),),
)


def ticking_clock(step_seconds: int = 1):
    """A clock that advances predictably, so hashes stay reproducible."""
    state = {"n": 0}

    def clock() -> datetime:
        state["n"] += 1
        return START + timedelta(seconds=step_seconds * state["n"])

    return clock


def chain(**overrides) -> AuditChain:
    return AuditChain("session-1", ticking_clock(), HarnessConfig.full(**overrides))


def populated() -> AuditChain:
    c = chain()
    c.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)
    c.append(
        AuditEvent.APPROVAL_RESOLVED,
        tool_call=CANCEL,
        approval=ApprovalDecision(
            request_id="r1", status=ApprovalStatus.APPROVED, policy="oracle"
        ),
    )
    c.append(
        AuditEvent.TOOL_EXECUTED,
        tool_call=CANCEL,
        decision=ALLOW,
        diff=world_diff({"status": "pending"}, {"status": "cancelled"}),
    )
    return c


class TestVerifiableOrder:
    def test_an_untouched_chain_verifies(self):
        assert populated().verify().intact is True

    def test_an_empty_chain_verifies(self):
        assert chain().verify().intact is True

    def test_the_first_record_starts_from_genesis(self):
        assert populated().records[0].prev_hash == GENESIS_HASH

    def test_each_record_names_the_previous_one(self):
        records = populated().records
        assert [r.prev_hash for r in records[1:]] == [r.record_hash for r in records[:-1]]

    def test_altering_content_is_detected(self):
        records = list(populated().records)
        records[1] = records[1].model_copy(update={"session_id": "someone-else"})
        result = verify_records(records)
        assert result.intact is False
        assert result.first_break.kind is BreakKind.CONTENT_TAMPERED

    def test_altering_content_and_resealing_breaks_the_next_link(self):
        # The property that matters: you cannot rewrite a record and hide it by
        # recomputing its own hash, because the record after it names the old one.
        records = list(populated().records)
        forged = records[1].model_copy(update={"session_id": "someone-else"})
        records[1] = forged.model_copy(update={"record_hash": forged.compute_hash()})
        result = verify_records(records)
        assert result.intact is False
        assert BreakKind.LINK_BROKEN in {b.kind for b in result.breaks}

    def test_removing_a_record_is_detected(self):
        records = list(populated().records)
        del records[1]
        kinds = {b.kind for b in verify_records(records).breaks}
        assert BreakKind.LINK_BROKEN in kinds
        assert BreakKind.SEQUENCE_BROKEN in kinds

    def test_reordering_records_is_detected(self):
        records = list(populated().records)
        records[0], records[1] = records[1], records[0]
        assert verify_records(records).intact is False

    def test_dropping_the_head_is_detected(self):
        records = list(populated().records)[1:]
        assert verify_records(records).first_break.kind is BreakKind.GENESIS_WRONG

    def test_every_break_is_reported_not_just_the_first(self):
        # Naming one break when there are three invites the assumption that
        # fixing it restores the chain.
        records = list(populated().records)
        del records[1]
        assert len(verify_records(records).breaks) > 1

    def test_the_chain_is_append_only(self):
        c = populated()
        snapshot = c.records
        list(snapshot)  # a tuple: no in-place mutation is available
        assert c.records == snapshot
        c.append(AuditEvent.TOOL_PROPOSED, tool_call=RETURN, decision=ALLOW)
        assert len(c.records) == len(snapshot) + 1
        assert c.records[: len(snapshot)] == snapshot


class TestCompleteness:
    def test_a_write_that_was_never_recorded_is_reported(self):
        # A chain with no gaps in its own numbering is still incomplete if
        # something happened that was never written down.
        assert populated().missing_executions([CANCEL, RETURN]) == (RETURN,)

    def test_nothing_is_missing_when_everything_was_recorded(self):
        assert populated().missing_executions([CANCEL]) == ()

    def test_matching_is_by_action_not_by_call_id(self):
        same_action = CANCEL.model_copy(update={"id": "a-different-message-id"})
        assert populated().missing_executions([same_action]) == ()

    def test_only_executions_count_towards_completeness(self):
        # A proposal that was refused did not change the world and must not be
        # mistaken for one that did.
        c = chain()
        c.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)
        assert c.missing_executions([CANCEL]) == (CANCEL,)


class TestLinkage:
    def test_an_execution_without_its_decision_is_reported(self):
        c = chain()
        c.append(AuditEvent.TOOL_EXECUTED, tool_call=CANCEL)
        assert c.unjustified_executions() == c.records

    def test_a_justified_execution_is_not_reported(self):
        assert populated().unjustified_executions() == ()

    def test_the_rule_that_decided_is_carried(self):
        assert populated().records[0].decision.rule == "gating.allowed"

    def test_the_cited_clauses_are_carried(self):
        assert populated().records[0].decision.clauses[0].doc_id == "retail/policy.md"

    def test_the_approval_is_carried(self):
        assert populated().records[1].approval.status is ApprovalStatus.APPROVED

    def test_the_world_change_is_carried(self):
        diff = populated().records[2].world_diff
        assert diff["values_changed"]["root['status']"]["new_value"] == "cancelled"


class TestWorldDiff:
    def test_the_diff_is_plain_json_types(self):
        import json

        json.dumps(world_diff({"a": 1}, {"a": 2}))  # must not raise

    def test_no_change_produces_an_empty_diff(self):
        assert world_diff({"a": 1}, {"a": 1}) == {}

    def test_the_diff_is_deterministic(self):
        before, after = {"a": 1, "b": {"c": 2}}, {"a": 9, "b": {"c": 3}}
        assert world_diff(before, after) == world_diff(before, after)


class TestReproducibility:
    def test_two_identical_sessions_hash_identically(self):
        # Why the clock is injected: a chain that read datetime.now() could not
        # be compared across two runs of the same experiment.
        assert [r.record_hash for r in populated().records] == [
            r.record_hash for r in populated().records
        ]

    def test_a_different_timestamp_changes_the_hash(self):
        a = AuditChain("s", lambda: START, HarnessConfig.full())
        b = AuditChain("s", lambda: START + timedelta(seconds=1), HarnessConfig.full())
        a.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)
        b.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)
        assert a.records[0].record_hash != b.records[0].record_hash

    def test_a_different_session_id_changes_the_hash(self):
        a = AuditChain("s1", lambda: START, HarnessConfig.full())
        b = AuditChain("s2", lambda: START, HarnessConfig.full())
        a.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)
        b.append(AuditEvent.TOOL_PROPOSED, tool_call=CANCEL, decision=ALLOW)
        assert a.records[0].record_hash != b.records[0].record_hash


class TestTheSwitch:
    def test_disabled_records_nothing_at_all(self):
        c = chain(audit_enabled=False)
        assert c.append(AuditEvent.TOOL_EXECUTED, tool_call=CANCEL, decision=ALLOW) is None
        assert c.records == ()

    def test_a_disabled_chain_still_verifies_vacuously(self):
        c = chain(audit_enabled=False)
        c.append(AuditEvent.TOOL_EXECUTED, tool_call=CANCEL)
        assert c.verify().intact is True

    def test_the_baseline_arm_carries_no_audit(self):
        c = AuditChain("s", lambda: START, HarnessConfig.baseline())
        assert c.enabled is False
