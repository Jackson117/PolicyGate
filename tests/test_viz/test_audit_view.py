"""Tests for the audit-chain view: the chain as rows, its integrity, and the
anchors that say where to look.

tau2 marks no record as the failure, so every anchor has to rest on something
observable. These tests pin what each one claims.
"""

from datetime import datetime

import pytest
from tau2.data_model.tasks import Action

from policygate.eval.runner import SessionRecord
from policygate.harness.audit import AuditChain
from policygate.harness.config import HarnessConfig
from policygate.harness.types import (
    AuditEvent,
    Decision,
    DecisionOutcome,
    ToolCall,
)
from policygate.viz.data import anchors, chain_integrity, chain_rows

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


def call(name, **args):
    return ToolCall(id=f"c-{name}", name=name, arguments=args)


def allow(rule="gating.allowed"):
    return Decision(outcome=DecisionOutcome.ALLOW, rule=rule, reason="fine")


def deny(rule, reason="no", remedy=None):
    return Decision(outcome=DecisionOutcome.DENY, rule=rule, reason=reason, remedy=remedy)


def session(chain) -> SessionRecord:
    return SessionRecord(task_id="1", seed=7, session_id="s", written_at="t", proposals=1,
                         llm_cost=0.0, chain_intact=True, records=tuple(chain.records))


def built(*events) -> SessionRecord:
    chain = AuditChain("s", CLOCK, HarnessConfig.full())
    for event, tool_call, decision in events:
        chain.append(event, tool_call=tool_call, decision=decision)
    return session(chain)


def action(name, **args):
    return Action(action_id=f"a-{name}", requestor="assistant", name=name, arguments=args)


class TestChainRows:
    def test_each_row_links_to_the_one_before_it(self):
        record = built(
            (AuditEvent.TOOL_PROPOSED, call("get_order_details", order_id="#W1"), allow()),
            (AuditEvent.TOOL_EXECUTED, call("cancel_pending_order", order_id="#W1"), allow()),
        )
        rows = chain_rows(record)
        assert [r.seq for r in rows] == [0, 1]  # the chain numbers from zero
        assert all(r.linked for r in rows)
        assert rows[1].prev_hash == rows[0].record_hash

    def test_a_refusal_row_carries_the_rule_the_reason_and_the_remedy(self):
        record = built((AuditEvent.TOOL_DENIED, call("cancel_pending_order", order_id="#W1"),
                        deny("validation.refused", "not retrieved", "look it up first")))
        row = chain_rows(record)[0]
        assert (row.rule, row.reason, row.remedy) == (
            "validation.refused", "not retrieved", "look it up first")

    def test_no_session_is_not_an_empty_chain_but_no_chain(self):
        assert chain_rows(None) == []
        intact, problems = chain_integrity(None)
        assert intact is False and problems


class TestIntegrity:
    def test_an_untouched_chain_verifies(self):
        record = built((AuditEvent.TOOL_PROPOSED, call("get_user_details"), allow()))
        assert chain_integrity(record) == (True, [])

    def test_an_edited_record_is_caught_and_named(self):
        record = built(
            (AuditEvent.TOOL_PROPOSED, call("get_order_details", order_id="#W1"), allow()),
            (AuditEvent.TOOL_EXECUTED, call("cancel_pending_order", order_id="#W1"), allow()),
        )
        # Rewrite what the second record says happened, keeping its stored hash.
        tampered = record.records[1].model_copy(update={"event": AuditEvent.TOOL_DENIED})
        edited = record.model_copy(update={"records": (record.records[0], tampered)})
        intact, problems = chain_integrity(edited)
        assert intact is False
        # Located by position: the record's own seq is under suspicion too.
        assert any("record #1" in p for p in problems)
        assert not chain_rows(edited)[1].linked or problems  # the view shows it either way


class TestAnchors:
    REFERENCE = [action("cancel_pending_order", order_id="#W1"),
                 action("return_delivered_order_items", order_id="#W2")]

    def test_a_reference_action_that_never_ran_is_an_anchor(self):
        record = built((AuditEvent.TOOL_EXECUTED, call("cancel_pending_order", order_id="#W1"),
                        allow()))
        missing = [a for a in anchors(record, self.REFERENCE)
                   if a.kind == "missing_reference_action"]
        assert [a.label for a in missing] == ["never ran · return_delivered_order_items on #W2"]
        assert missing[0].seq is None and missing[0].detail == "never proposed"

    def test_a_reference_action_that_was_refused_points_at_the_refusal(self):
        record = built(
            (AuditEvent.TOOL_DENIED, call("cancel_pending_order", order_id="#W1"),
             deny("validation.refused")),
            (AuditEvent.TOOL_EXECUTED, call("return_delivered_order_items", order_id="#W2"),
             allow()),
        )
        missing = next(a for a in anchors(record, self.REFERENCE)
                       if a.kind == "missing_reference_action")
        assert missing.seq == 0 and missing.detail == "proposed and refused here"

    def test_a_write_the_reference_does_not_contain_is_an_anchor(self):
        record = built((AuditEvent.TOOL_EXECUTED, call("cancel_pending_order", order_id="#W9"),
                        allow()))
        extra = [a for a in anchors(record, self.REFERENCE) if a.kind == "unreferenced_write"]
        assert [a.label for a in extra] == [
            "ran, not in the reference · cancel_pending_order on #W9"]

    def test_the_first_refusal_says_whether_anything_ran_after_it(self):
        stuck = built((AuditEvent.TOOL_DENIED, call("cancel_pending_order", order_id="#W1"),
                       deny("validation.refused", "nope")))
        recovered = built(
            (AuditEvent.TOOL_DENIED, call("cancel_pending_order", order_id="#W1"),
             deny("validation.refused", "nope")),
            (AuditEvent.TOOL_EXECUTED, call("cancel_pending_order", order_id="#W1"), allow()),
        )
        assert "Nothing ran after it." in next(
            a for a in anchors(stuck) if a.kind == "first_refusal").detail
        assert "Something ran after it." in next(
            a for a in anchors(recovered) if a.kind == "first_refusal").detail

    def test_anchors_are_ordered_by_position_with_absences_last(self):
        record = built(
            (AuditEvent.TOOL_DENIED, call("cancel_pending_order", order_id="#W1"),
             deny("validation.refused")),
            (AuditEvent.TOOL_EXECUTED, call("transfer_to_human_agents"), allow()),
        )
        found = anchors(record, self.REFERENCE)
        seqs = [a.seq for a in found]
        assert seqs == sorted(seqs, key=lambda s: (s is None, s or 0))
        assert found[-1].seq is None

    def test_a_session_matching_its_reference_exactly_has_nothing_to_flag(self):
        record = built(
            (AuditEvent.TOOL_EXECUTED, call("cancel_pending_order", order_id="#W1"), allow()),
            (AuditEvent.TOOL_EXECUTED, call("return_delivered_order_items", order_id="#W2"),
             allow()),
        )
        assert anchors(record, self.REFERENCE) == []


@pytest.mark.parametrize("session_record", [None])
def test_nothing_to_anchor_without_a_session(session_record):
    assert anchors(session_record, []) == []


class TestJumpingToARecord:
    """A record's seq is not a message index: refused proposals have records and
    no message at all, so the jump joins the two on the tool call's id."""

    class FakeMessage:
        def __init__(self, tool_calls=()):
            self.tool_calls = tool_calls
            self.role = "assistant"

    class FakeCall:
        def __init__(self, call_id):
            self.id = call_id

    class FakeSim:
        def __init__(self, messages):
            self.messages = messages

    def build(self):
        chain = AuditChain("s", CLOCK, HarnessConfig.full())
        chain.append(AuditEvent.TOOL_EXECUTED, tool_call=call("get_order_details", order_id="#W1"),
                     decision=allow())
        chain.append(AuditEvent.TOOL_DENIED, tool_call=call("cancel_pending_order", order_id="#W1"),
                     decision=deny("validation.refused"))
        chain.append(AuditEvent.TOOL_EXECUTED,
                     tool_call=call("return_delivered_order_items", order_id="#W1"),
                     decision=allow())
        record = session(chain)
        messages = [
            self.FakeMessage(),                                        # 0 greeting
            self.FakeMessage([self.FakeCall("c-get_order_details")]),  # 1 the read
            self.FakeMessage(),                                        # 2 talk
            self.FakeMessage([self.FakeCall("c-return_delivered_order_items")]),  # 3 the write
        ]
        return self.FakeSim(messages), record

    def test_an_executed_call_lands_on_its_own_message(self):
        from policygate.viz.data import message_index_for

        sim, record = self.build()
        assert message_index_for(sim, record, 0) == 1
        assert message_index_for(sim, record, 2) == 3

    def test_a_refused_call_lands_after_the_last_message_that_ran(self):
        from policygate.viz.data import message_index_for

        sim, record = self.build()
        # seq 1 was refused and never reached tau2; the model was at message 1
        # when it proposed, so the trace opens at the message after it.
        assert message_index_for(sim, record, 1) == 2

    def test_without_a_session_there_is_nowhere_to_jump(self):
        from policygate.viz.data import message_index_for

        sim, _ = self.build()
        assert message_index_for(sim, None, 3) == 0


def test_every_page_file_is_registered_in_the_navigation():
    """A page that exists but is not listed is invisible: st.navigation takes an
    explicit list, and the audit page was added to pages/ without it once."""
    from pathlib import Path

    import policygate.viz.app as app_module

    pages = Path(app_module.__file__).parent / "pages"
    on_disk = {p.name for p in pages.glob("*.py")}
    listed = set()
    for line in Path(app_module.__file__).read_text().splitlines():
        if 'PAGES / "' in line:
            listed.add(line.split('PAGES / "')[1].split('"')[0])
    assert on_disk == listed
