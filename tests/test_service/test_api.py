"""The deployment end to end: HTTP in, LangGraph interrupt, a human's answer, tau2's world.

The LLM is scripted; everything else is real -- retail's tools and database,
the harness modules W4 measured, a LangGraph checkpointer, the audit store. The
tests are about the one thing the service adds: a write that pauses for a human
and resumes correctly, including on a process that did not start it.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from tau2.data_model.message import AssistantMessage, ToolCall

from policygate.service.api import Service, create_app
from policygate.service.graph import ESCALATION_TEXT, EXHAUSTED_TEXT
from policygate.service.runtime import RetailRuntime
from policygate.store.repo import AuditStore

EMAIL = "sofia.rossi2645@example.com"
ORDER = "#W5918442"  # pending in retail's initial database


class Script:
    """Scripted model replies; records the tool names it was offered each time."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.offered = []

    def __call__(self, *, model, tools, messages, **kwargs):
        self.offered.append({t.name for t in tools})
        if not self.replies:
            raise AssertionError("the model was asked for more replies than scripted")
        return self.replies.pop(0)


def calls(name, call_id, **arguments):
    return AssistantMessage(role="assistant", content=None,
                            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def says(text):
    return AssistantMessage(role="assistant", content=text)


def to_cancellation():
    return [
        calls("find_user_id_by_email", "c1", email=EMAIL),
        calls("get_order_details", "c2", order_id=ORDER),
        calls("cancel_pending_order", "c3", order_id=ORDER, reason="no longer needed"),
    ]


@pytest.fixture
def store():
    return AuditStore(create_engine("sqlite://", connect_args={"check_same_thread": False},
                                    poolclass=StaticPool))


@pytest.fixture
def saver():
    return InMemorySaver()


def client_for(script, store, saver, **runtime_args):
    service = Service(RetailRuntime(generate_fn=script, **runtime_args), store, saver)
    return TestClient(create_app(service)), service


def open_session(client):
    return client.post("/sessions").json()["session_id"]


def test_a_write_pauses_for_a_human_and_runs_once_approved(store, saver):
    script = Script(*to_cancellation(), says("Your order has been cancelled."))
    client, _ = client_for(script, store, saver)
    sid = open_session(client)

    paused = client.post(f"/sessions/{sid}/messages", json={"content": "Cancel my order"}).json()
    assert paused["status"] == "awaiting_approval"
    assert paused["approval"]["tool"] == "cancel_pending_order"
    assert paused["session_state"] == "pending_approval"
    assert [a["request_id"] for a in client.get("/approvals").json()] == [
        paused["approval"]["request_id"]
    ]

    done = client.post(f"/approvals/{paused['approval']['request_id']}",
                       json={"approved": True, "reviewer": "ops@example.com"}).json()
    assert done["status"] == "replied"
    assert done["reply"] == "Your order has been cancelled."
    assert client.get("/approvals").json() == []

    view = client.get(f"/sessions/{sid}").json()
    cancel_result = next(m for m in view["transcript"]
                         if m["role"] == "tool" and m["content"] and "cancelled" in m["content"])
    assert cancel_result["error"] is False
    assert view["audit_intact"] is True


def test_the_audit_chain_links_the_request_the_answer_and_the_execution(store, saver):
    script = Script(*to_cancellation(), says("Done."))
    client, _ = client_for(script, store, saver)
    sid = open_session(client)
    paused = client.post(f"/sessions/{sid}/messages", json={"content": "Cancel"}).json()
    client.post(f"/approvals/{paused['approval']['request_id']}",
                json={"approved": True, "reviewer": "ops@example.com", "note": "verified"})

    audit = client.get(f"/sessions/{sid}/audit").json()
    assert audit["intact"] is True
    trail = [(r["event"], (r["tool_call"] or {}).get("name")) for r in audit["records"]]
    cancel = [e for e, name in trail if name == "cancel_pending_order"]
    assert cancel == ["tool_proposed", "approval_requested", "approval_resolved", "tool_executed"]
    resolved = next(r for r in audit["records"] if r["event"] == "approval_resolved")
    assert resolved["approval"]["policy"] == "human:ops@example.com"
    assert resolved["approval"]["reason"] == "verified"


def test_a_rejected_write_never_runs_and_the_model_is_told_why(store, saver):
    script = Script(*to_cancellation(), says("I can't cancel that without approval."))
    client, _ = client_for(script, store, saver)
    sid = open_session(client)
    paused = client.post(f"/sessions/{sid}/messages", json={"content": "Cancel"}).json()

    done = client.post(f"/approvals/{paused['approval']['request_id']}",
                       json={"approved": False, "reviewer": "ops@example.com",
                             "note": "customer not verified"}).json()
    assert done["reply"] == "I can't cancel that without approval."
    audit = client.get(f"/sessions/{sid}/audit").json()["records"]
    assert not any(
        r["event"] == "tool_executed" and r["tool_call"]["name"] == "cancel_pending_order"
        for r in audit
    )
    view = client.get(f"/sessions/{sid}").json()
    refusal = [m for m in view["transcript"] if m["role"] == "tool" and m["error"]]
    assert "customer not verified" in refusal[-1]["content"]


def test_nothing_proceeds_while_an_approval_is_pending(store, saver):
    client, _ = client_for(Script(*to_cancellation()), store, saver)
    sid = open_session(client)
    client.post(f"/sessions/{sid}/messages", json={"content": "Cancel"})
    blocked = client.post(f"/sessions/{sid}/messages", json={"content": "hello?"})
    assert blocked.status_code == 409


def test_an_approval_can_be_answered_by_a_process_that_did_not_start_it(store, saver):
    """The world is rebuilt by replay: the second runtime has never seen this session."""
    first, _ = client_for(Script(*to_cancellation()), store, saver)
    sid = open_session(first)
    paused = first.post(f"/sessions/{sid}/messages", json={"content": "Cancel"}).json()

    after_restart = Script(
        says("Cancelled."),
        calls("get_order_details", "c4", order_id=ORDER),
        says("It shows as cancelled."),
    )
    second, _ = client_for(after_restart, store, saver)
    done = second.post(f"/approvals/{paused['approval']['request_id']}",
                       json={"approved": True, "reviewer": "ops@example.com"}).json()
    assert done["status"] == "replied"

    # A third process, a later question: the cancellation must still be in the world.
    third, _ = client_for(Script(calls("get_order_details", "c5", order_id=ORDER),
                                 says("It is cancelled.")), store, saver)
    third.post(f"/sessions/{sid}/messages", json={"content": "Is it cancelled?"})
    tool_results = [m for m in third.get(f"/sessions/{sid}").json()["transcript"]
                    if m["role"] == "tool" and m["content"] and ORDER in m["content"]]
    assert json.loads(tool_results[-1]["content"])["status"] == "cancelled"


def test_answering_a_stale_or_unknown_approval_is_refused(store, saver):
    client, _ = client_for(Script(*to_cancellation(), says("Done.")), store, saver)
    sid = open_session(client)
    paused = client.post(f"/sessions/{sid}/messages", json={"content": "Cancel"}).json()
    rid = paused["approval"]["request_id"]
    answer = {"approved": True, "reviewer": "x"}
    assert client.post("/approvals/nope", json=answer).status_code == 404
    client.post(f"/approvals/{rid}", json=answer)
    assert client.post(f"/approvals/{rid}", json=answer).status_code == 409


def test_before_verification_only_identity_lookups_are_offered(store, saver):
    script = Script(says("Could I have your email?"))
    client, _ = client_for(script, store, saver)
    sid = open_session(client)
    client.post(f"/sessions/{sid}/messages", json={"content": "hi"})
    assert script.offered[0] == {"find_user_id_by_email", "find_user_id_by_name_zip",
                                 "transfer_to_human_agents"}


def test_a_model_that_keeps_proposing_refused_actions_is_handed_to_a_human(store, saver):
    refused = [calls("get_user_details", f"x{i}", user_id="sofia_rossi_8776") for i in range(3)]
    client, _ = client_for(Script(*refused), store, saver)
    sid = open_session(client)
    result = client.post(f"/sessions/{sid}/messages", json={"content": "show my profile"}).json()
    assert result["status"] == "transferred"
    assert result["reply"] == ESCALATION_TEXT
    assert client.post(f"/sessions/{sid}/messages", json={"content": "hi"}).status_code == 409
    rules = [r["decision"]["rule"] for r in client.get(f"/sessions/{sid}/audit").json()["records"]
             if r["decision"]]
    assert rules.count("gating.identity_first") == 3
    assert rules[-1] == "gating.budget_exhausted"


# --- the spend cap: what makes a public URL survivable -----------------------


def costing(text, usd):
    """A scripted reply that reports what it cost, the way litellm does."""
    return AssistantMessage(role="assistant", content=text, cost=usd)


def test_the_session_reports_what_it_has_spent(store, saver):
    script = Script(costing("Hello.", 0.004), costing("Still here.", 0.006))
    client, _ = client_for(script, store, saver, session_usd_cap=1.0)
    sid = open_session(client)

    client.post(f"/sessions/{sid}/messages", json={"content": "hi"})
    client.post(f"/sessions/{sid}/messages", json={"content": "again"})

    view = client.get(f"/sessions/{sid}").json()
    assert view["spent_usd"] == pytest.approx(0.010)
    assert view["session_usd_cap"] == 1.0


def test_a_spent_session_stops_before_calling_the_model(store, saver):
    """The cap is checked before the call. One reply is scripted; a second
    request must not reach for it."""
    script = Script(costing("That cost a lot.", 0.50))
    client, _ = client_for(script, store, saver, session_usd_cap=0.25)
    sid = open_session(client)

    first = client.post(f"/sessions/{sid}/messages", json={"content": "hi"}).json()
    assert first["reply"] == "That cost a lot."

    second = client.post(f"/sessions/{sid}/messages", json={"content": "again"}).json()
    assert second["reply"] == EXHAUSTED_TEXT
    assert second["status"] == "replied"
    # Script raises if asked for a reply it does not have, so an empty one that
    # was never consulted is the proof that no model call was made.
    assert script.replies == []
    assert len(script.offered) == 1

    view = client.get(f"/sessions/{sid}").json()
    assert view["spent_usd"] == pytest.approx(0.50)
    # The refusal is in the conversation, not only in a log.
    assert view["transcript"][-1]["content"] == EXHAUSTED_TEXT


def test_no_cap_means_no_cap(store, saver):
    script = Script(costing("One.", 5.0), costing("Two.", 5.0))
    client, _ = client_for(script, store, saver)
    sid = open_session(client)

    client.post(f"/sessions/{sid}/messages", json={"content": "hi"})
    second = client.post(f"/sessions/{sid}/messages", json={"content": "again"}).json()

    assert second["reply"] == "Two."
    assert client.get(f"/sessions/{sid}").json()["session_usd_cap"] is None
