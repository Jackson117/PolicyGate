"""HTTP surface: a customer's session, and a reviewer's approval queue.

The graph never touches the audit store; this layer does, after every step.
That keeps one writer per concern -- the checkpointer holds what the graph needs
to resume, the audit store holds what an auditor needs to verify -- and makes the
store's idempotent append the only thing standing between a retried request and
a duplicated record.

Run with:
    uvicorn policygate.service.api:build --factory
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import ExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.types import Command

from policygate.harness.state import SessionFacts, SessionTracker
from policygate.harness.types import ApprovalRequest, ApprovalStatus, AuditRecord
from policygate.service.graph import build_graph
from policygate.service.runtime import RetailRuntime
from policygate.service.schemas import (
    ApprovalAnswer,
    PendingApproval,
    SessionView,
    TurnResult,
    UserTurn,
)
from policygate.store.repo import AuditStore

# uvicorn configures its own loggers and leaves the root one alone, so a module
# logger here prints nothing under the server this app is run by. Its logger is
# the one a reader of the service log will actually see.
logger = logging.getLogger("uvicorn.error")


class Service:
    def __init__(self, runtime: RetailRuntime, store: AuditStore, checkpointer: Any) -> None:
        self.runtime = runtime
        self.store = store
        self.graph = build_graph(runtime, checkpointer)

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _values(self, session_id: str) -> dict[str, Any]:
        return self.graph.get_state(self._config(session_id)).values or {}

    def _state_label(self, values: dict[str, Any]) -> str:
        return SessionTracker(SessionFacts.model_validate(values.get("facts") or {})).state.value

    def _pending(self, session_id: str, values: dict[str, Any]) -> PendingApproval | None:
        raw = values.get("pending_approval")
        if not raw:
            return None
        request = ApprovalRequest.model_validate(raw)
        return PendingApproval(
            request_id=request.request_id, session_id=session_id, tool=request.tool_call.name,
            arguments=dict(request.tool_call.arguments), reason=request.reason,
        )

    def _sync(self, session_id: str) -> tuple[dict[str, Any], PendingApproval | None]:
        values = self._values(session_id)
        self.store.append([AuditRecord.model_validate(r) for r in values.get("audit", [])])
        pending = self._pending(session_id, values)
        if pending is not None:
            self.store.open_approval(
                session_id, ApprovalRequest.model_validate(values["pending_approval"]),
                self.runtime.clock(),
            )
        return values, pending

    def _result(self, session_id: str) -> TurnResult:
        values, pending = self._sync(session_id)
        label = self._state_label(values)
        if pending is not None:
            status = "awaiting_approval"
        elif label == "transferred":
            status = "transferred"
        else:
            status = "replied"
        return TurnResult(session_id=session_id, status=status, reply=values.get("reply"),
                          approval=pending, session_state=label)

    # --- operations ------------------------------------------------------

    def send(self, session_id: str, content: str) -> TurnResult:
        values = self._values(session_id)
        if values.get("pending_approval"):
            raise HTTPException(409, "a write is awaiting approval; nothing proceeds until then")
        if values and self._state_label(values) == "transferred":
            raise HTTPException(409, "this session was handed to a human agent")
        user = {"role": "user", "content": content}
        self.graph.invoke({"session_id": session_id, "messages": [user]}, self._config(session_id))
        return self._result(session_id)

    def answer(self, request_id: str, answer: ApprovalAnswer) -> TurnResult:
        row = self.store.approval(request_id)
        if row is None:
            raise HTTPException(404, f"no approval {request_id}")
        values = self._values(row.session_id)
        pending = self._pending(row.session_id, values)
        if pending is None or pending.request_id != request_id:
            raise HTTPException(409, f"approval {request_id} is not the one this session awaits")
        status = ApprovalStatus.APPROVED if answer.approved else ApprovalStatus.REJECTED
        self.store.resolve_approval(request_id, status, answer.reviewer, answer.note,
                                    self.runtime.clock())
        self.graph.invoke(Command(resume=answer.model_dump()), self._config(row.session_id))
        return self._result(row.session_id)

    def view(self, session_id: str) -> SessionView:
        values = self._values(session_id)
        if not values:
            raise HTTPException(404, f"no session {session_id}")
        transcript = [
            {"role": m["role"], "content": m.get("content"),
             "tool_calls": [c["name"] for c in (m.get("tool_calls") or [])] or None,
             "error": m.get("error")}
            for m in values.get("messages", [])
        ]
        verification = self.store.verify(session_id)
        return SessionView(
            session_id=session_id, session_state=self._state_label(values), transcript=transcript,
            pending_approval=self._pending(session_id, values),
            audit_records=len(self.store.records(session_id)),
            audit_intact=verification.intact,
        )


def create_app(service: Service, lifespan: Any = None) -> FastAPI:
    api = FastAPI(title="policygate service", lifespan=lifespan)

    @api.post("/sessions")
    def open_session() -> dict[str, str]:
        return {"session_id": str(uuid.uuid4())}

    @api.post("/sessions/{session_id}/messages", response_model=TurnResult)
    def send(session_id: str, turn: UserTurn) -> TurnResult:
        return service.send(session_id, turn.content)

    @api.get("/sessions/{session_id}", response_model=SessionView)
    def view(session_id: str) -> SessionView:
        return service.view(session_id)

    @api.get("/sessions/{session_id}/audit")
    def audit(session_id: str) -> dict[str, Any]:
        records = service.store.records(session_id)
        verification = service.store.verify(session_id)
        return {"intact": verification.intact,
                "breaks": [b.model_dump(mode="json") for b in verification.breaks],
                "records": [r.model_dump(mode="json") for r in records]}

    @api.get("/approvals", response_model=list[PendingApproval])
    def pending() -> list[PendingApproval]:
        return [
            PendingApproval(request_id=r.request_id, session_id=r.session_id, tool=r.tool_name,
                            arguments=r.arguments, reason=r.reason)
            for r in service.store.pending()
        ]

    @api.post("/approvals/{request_id}", response_model=TurnResult)
    def answer(request_id: str, body: ApprovalAnswer) -> TurnResult:
        return service.answer(request_id, body)

    return api


def build() -> FastAPI:
    """Application factory: ``uvicorn policygate.service.api:build --factory``.

    Opens the checkpointer and the audit store for the life of the app and
    closes them on shutdown. ``POLICYGATE_CHECKPOINT_URL`` and
    ``POLICYGATE_AUDIT_URL`` point at Postgres; they must be different databases
    from anything the agent acts on (docs/constraints.md #3).
    """
    from policygate.environment import apply_project_env, describe
    from policygate.service.checkpointer import checkpointer

    # Before anything reads a key. Launching this with `uvicorn ...` directly,
    # from a shell that exports another project's OPENAI_API_KEY, otherwise
    # sends that key to this project's base URL and fails on the first turn.
    logger.info("provider configuration: %s", describe(apply_project_env()))

    resources = ExitStack()
    saver = resources.enter_context(checkpointer())
    store = AuditStore.from_url(os.environ.get("POLICYGATE_AUDIT_URL", "sqlite:///audit.db"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            resources.close()

    return create_app(Service(RetailRuntime(), store, saver), lifespan=lifespan)
