"""Tests for the dashboard's client of the running service.

Driven through httpx's mock transport: no server, no network, no key. What is
pinned here is the part a person sees -- that a turn, an approval and the chain
come back as the service's own types, and that every failure arrives as a
sentence rather than a traceback.
"""

import httpx
import pytest

from policygate.viz.live import ServiceClient, ServiceUnavailable, records_as_audit


@pytest.fixture(autouse=True)
def mock_http(monkeypatch):
    """Route every httpx.Client through the handler the test installs."""
    state = {"handler": None}

    original = httpx.Client

    def patched(*args, **kwargs):
        if state["handler"] is not None:
            kwargs["transport"] = httpx.MockTransport(state["handler"])
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched)
    return state


def answer(mock_http, handler):
    mock_http["handler"] = handler
    return ServiceClient("http://service:8000")


TURN = {
    "session_id": "s1",
    "status": "awaiting_approval",
    "reply": None,
    "session_state": "pending_approval",
    "approval": {
        "request_id": "r1",
        "session_id": "s1",
        "tool": "cancel_pending_order",
        "arguments": {"order_id": "#W1", "reason": "no longer needed"},
        "reason": "cancel_pending_order changes stored state",
    },
}


class TestTurns:
    def test_a_turn_comes_back_as_the_services_own_type(self, mock_http):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = request.read()
            return httpx.Response(200, json=TURN)

        result = answer(mock_http, handler).send("s1", "cancel my order")
        assert seen["url"].endswith("/sessions/s1/messages")
        assert b"cancel my order" in seen["body"]
        assert result.status == "awaiting_approval"
        assert result.approval.tool == "cancel_pending_order"

    def test_opening_a_session_returns_its_id(self, mock_http):
        c = answer(mock_http, lambda r: httpx.Response(200, json={"session_id": "abc"}))
        assert c.open_session() == "abc"


class TestApprovals:
    def test_answering_sends_the_decision_and_the_reviewer(self, mock_http):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = request.read().decode()
            return httpx.Response(200, json={**TURN, "status": "replied", "reply": "Done.",
                                             "approval": None})

        result = answer(mock_http, handler).answer(
            "r1", approved=True, reviewer="olivia", note="confirmed by phone"
        )
        assert seen["url"].endswith("/approvals/r1")
        assert '"approved":true' in seen["body"] and "olivia" in seen["body"]
        assert result.reply == "Done."

    def test_pending_approvals_are_typed(self, mock_http):
        c = answer(mock_http, lambda r: httpx.Response(200, json=[TURN["approval"]]))
        assert [p.request_id for p in c.pending()] == ["r1"]


class TestTheChain:
    def test_the_audit_endpoint_returns_records_the_chain_views_can_read(self, mock_http):
        from datetime import datetime

        from policygate.harness.audit import AuditChain
        from policygate.harness.config import HarnessConfig
        from policygate.harness.types import (
            AuditEvent,
            Decision,
            DecisionOutcome,
            ToolCall,
        )
        from policygate.viz.data import chain_integrity, chain_rows

        chain = AuditChain("s1", lambda: datetime(2026, 1, 1), HarnessConfig.full())
        chain.append(
            AuditEvent.TOOL_DENIED,
            tool_call=ToolCall(id="c1", name="cancel_pending_order", arguments={}),
            decision=Decision(outcome=DecisionOutcome.DENY, rule="validation.refused",
                              reason="not retrieved", remedy="look it up first"),
        )
        body = {
            "intact": True,
            "breaks": [],
            "records": [r.model_dump(mode="json") for r in chain.records],
        }
        c = answer(mock_http, lambda r: httpx.Response(200, json=body))
        intact, breaks, records = c.audit("s1")
        assert (intact, breaks) == (True, [])

        parsed = records_as_audit(records)
        assert chain_integrity(parsed) == (True, [])
        assert chain_rows(parsed)[0].remedy == "look it up first"


class TestFailuresArriveAsSentences:
    def test_a_service_that_is_not_running_says_how_to_start_it(self, mock_http):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(ServiceUnavailable, match="make service"):
            answer(mock_http, handler).open_session()

    def test_a_slow_turn_says_so_rather_than_hanging_forever(self, mock_http):
        def handler(request):
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(ServiceUnavailable, match="did not answer within"):
            answer(mock_http, handler).send("s1", "hello")

    def test_an_http_error_carries_the_services_own_detail(self, mock_http):
        c = answer(mock_http, lambda r: httpx.Response(409, json={"detail": "not the one awaited"}))
        with pytest.raises(ServiceUnavailable, match="409.*not the one awaited"):
            c.answer("r1", approved=True, reviewer="x")

    def test_a_missing_key_is_named_rather_than_left_as_a_500(self, mock_http):
        c = answer(mock_http, lambda r: httpx.Response(
            500, json={"detail": "AuthenticationError: no api key provided"}))
        with pytest.raises(ServiceUnavailable, match="provider key"):
            c.send("s1", "hello")

    def test_reachable_is_false_rather_than_raising(self, mock_http):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        assert answer(mock_http, handler).reachable() is False
