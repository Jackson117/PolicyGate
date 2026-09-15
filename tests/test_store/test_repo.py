"""Tests for store/repo.py, against SQLite: the properties are the database's, not the driver's.

Postgres is what the service runs on; the behaviour tested here -- idempotent
appends that refuse to overwrite evidence, verification from rows alone, an
approval that resolves once -- does not depend on which SQL engine holds it.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from policygate.harness.audit import AuditChain
from policygate.harness.config import HarnessConfig
from policygate.harness.types import (
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    Decision,
    DecisionOutcome,
    ToolCall,
)
from policygate.store.models import AuditRow
from policygate.store.repo import AuditStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)
CALL = ToolCall(id="c1", name="cancel_pending_order",
                arguments={"order_id": "#W1", "reason": "no longer needed"})
ALLOW = Decision(outcome=DecisionOutcome.ALLOW, rule="gating.allowed", reason="scoped")


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    return AuditStore(engine)


def chain(session_id="s1", n=3):
    ticks = iter(T0 + timedelta(seconds=i) for i in range(100))
    c = AuditChain(session_id, lambda: next(ticks), HarnessConfig.full())
    for _ in range(n):
        c.append(AuditEvent.TOOL_PROPOSED, tool_call=CALL, decision=ALLOW)
    return c


def test_records_round_trip_and_verify_from_the_database_alone(store):
    c = chain()
    assert store.append(c.records) == 3
    assert store.records("s1") == c.records
    assert store.verify("s1").intact


def test_appending_the_same_records_again_is_a_no_op(store):
    c = chain()
    store.append(c.records)
    assert store.append(c.records) == 0
    assert len(store.records("s1")) == 3


def test_a_different_record_at_a_stored_position_is_refused(store):
    store.append(chain().records)
    forged = chain().records[1].model_copy(update={"record_hash": "f" * 64})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        store.append([forged])


def test_tampering_with_a_stored_row_is_detected(store):
    store.append(chain().records)
    with store.session() as db, db.begin():
        row = db.query(AuditRow).filter_by(session_id="s1", seq=1).one()
        record = dict(row.record)
        record["event"] = "tool_executed"
        row.record = record
    assert not store.verify("s1").intact


def test_sessions_are_kept_apart(store):
    store.append(chain("s1").records)
    store.append(chain("s2", n=1).records)
    assert len(store.records("s1")) == 3
    assert len(store.records("s2")) == 1


def request(request_id="r1"):
    return ApprovalRequest(request_id=request_id, tool_call=CALL, reason="writes to the database")


def test_an_approval_is_pending_until_resolved_once(store):
    store.open_approval("s1", request(), T0)
    assert [r.request_id for r in store.pending()] == ["r1"]
    store.resolve_approval("r1", ApprovalStatus.APPROVED, "reviewer@example.com", "ok", T0)
    assert store.pending() == []
    assert store.approval("r1").status == "approved"
    with pytest.raises(ValueError, match="already approved"):
        store.resolve_approval("r1", ApprovalStatus.REJECTED, "someone", None, T0)


def test_reopening_the_same_request_is_idempotent(store):
    """LangGraph re-runs the interrupting node on resume; the request must not duplicate."""
    store.open_approval("s1", request(), T0)
    store.open_approval("s1", request(), T0)
    assert len(store.pending()) == 1


def test_pending_can_be_scoped_to_a_session(store):
    store.open_approval("s1", request("r1"), T0)
    store.open_approval("s2", request("r2"), T0 + timedelta(seconds=1))
    assert [r.request_id for r in store.pending("s2")] == ["r2"]


def test_resolving_an_unknown_request_raises(store):
    with pytest.raises(KeyError):
        store.resolve_approval("nope", ApprovalStatus.APPROVED, "x", None, T0)
