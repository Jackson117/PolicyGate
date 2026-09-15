"""Contract tests for harness/types.py.

These pin the two properties the rest of the harness depends on: an action's
fingerprint identifies the action rather than the message that carried it, and
the audit chain's hashes are reproducible and tamper-evident.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from policygate.harness.types import (
    GENESIS_HASH,
    AuditEvent,
    AuditRecord,
    CheckFailure,
    CheckResult,
    Decision,
    DecisionOutcome,
    FailureCode,
    ToolCall,
    canonical_json,
)

TS = datetime(2026, 1, 1, 12, 0, 0)


def call(**kw) -> ToolCall:
    base = {"id": "c1", "name": "return_delivered_order_items", "arguments": {"order_id": "#W1"}}
    return ToolCall(**{**base, **kw})


class TestCanonicalJson:
    def test_key_order_does_not_matter(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_nested_keys_are_sorted_too(self):
        assert canonical_json({"x": {"b": 1, "a": 2}}) == canonical_json({"x": {"a": 2, "b": 1}})

    def test_list_order_is_significant(self):
        # Argument lists such as item_ids are ordered data, not sets.
        assert canonical_json([1, 2]) != canonical_json([2, 1])


class TestToolCallFingerprint:
    def test_ignores_the_provider_call_id(self):
        assert call(id="a").fingerprint() == call(id="b").fingerprint()

    def test_ignores_argument_key_order(self):
        a = call(arguments={"order_id": "#W1", "item_ids": ["x"]})
        b = call(arguments={"item_ids": ["x"], "order_id": "#W1"})
        assert a.fingerprint() == b.fingerprint()

    def test_distinguishes_different_referents(self):
        # task 14: same tool, different order -- must not collapse together.
        a = call(arguments={"order_id": "#W7387996"})
        b = call(arguments={"order_id": "#W5490111"})
        assert a.fingerprint() != b.fingerprint()

    def test_distinguishes_different_tools(self):
        assert call().fingerprint() != call(name="exchange_delivered_order_items").fingerprint()

    def test_is_immutable(self):
        with pytest.raises(ValidationError):
            call().name = "other"


class TestCheckResult:
    def test_failed_requires_a_failure(self):
        with pytest.raises(ValueError):
            CheckResult.failed()

    def test_merge_keeps_every_failure(self):
        f1 = CheckFailure(code=FailureCode.VALUE_NOT_CANONICAL, message="state")
        f2 = CheckFailure(code=FailureCode.REFERENT_NOT_OWNED, message="order")
        merged = CheckResult.failed(f1).merged_with(CheckResult.failed(f2))
        assert merged.passed is False
        assert len(merged.failures) == 2

    def test_merging_two_passes_still_passes(self):
        assert CheckResult.ok().merged_with(CheckResult.ok()).passed is True

    def test_ok_merged_with_failure_fails(self):
        f = CheckFailure(code=FailureCode.SCHEMA_INVALID, message="bad")
        assert CheckResult.ok().merged_with(CheckResult.failed(f)).passed is False


def record(seq: int, prev: str, **kw) -> AuditRecord:
    fields = {
        "seq": seq,
        "timestamp": TS,
        "event": AuditEvent.TOOL_PROPOSED,
        "session_id": "s1",
        "tool_call": call(),
        "prev_hash": prev,
        "record_hash": "",
        **kw,
    }
    draft = AuditRecord(**fields)
    return draft.model_copy(update={"record_hash": draft.compute_hash()})


class TestAuditRecordHashing:
    def test_hash_is_reproducible(self):
        assert record(1, GENESIS_HASH).record_hash == record(1, GENESIS_HASH).record_hash

    def test_self_consistency_holds_for_an_untouched_record(self):
        assert record(1, GENESIS_HASH).is_self_consistent()

    def test_tampering_with_content_breaks_self_consistency(self):
        r = record(1, GENESIS_HASH)
        forged = r.model_copy(update={"tool_call": call(arguments={"order_id": "#W999"})})
        assert forged.is_self_consistent() is False

    def test_hash_covers_prev_hash_so_reordering_breaks_the_chain(self):
        assert record(1, GENESIS_HASH).record_hash != record(1, "f" * 64).record_hash

    def test_decision_is_part_of_the_hashed_payload(self):
        allow = Decision(outcome=DecisionOutcome.ALLOW, rule="r", reason="ok")
        deny = Decision(outcome=DecisionOutcome.DENY, rule="r", reason="ok")
        a = record(1, GENESIS_HASH, decision=allow)
        b = record(1, GENESIS_HASH, decision=deny)
        assert a.record_hash != b.record_hash

    def test_timestamp_is_caller_supplied_not_read_from_the_clock(self):
        earlier = record(1, GENESIS_HASH, timestamp=datetime(2020, 1, 1))
        assert earlier.record_hash != record(1, GENESIS_HASH).record_hash
        assert earlier.is_self_consistent()
