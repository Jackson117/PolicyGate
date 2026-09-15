"""Request and response bodies for the service API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class UserTurn(BaseModel):
    content: str = Field(min_length=1)


class ApprovalAnswer(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1)
    note: str | None = None


class PendingApproval(BaseModel):
    request_id: str
    session_id: str
    tool: str
    arguments: dict[str, Any]
    reason: str


class TurnResult(BaseModel):
    session_id: str
    status: Literal["replied", "awaiting_approval", "transferred"]
    reply: str | None = None
    approval: PendingApproval | None = None
    session_state: str


class SessionView(BaseModel):
    session_id: str
    session_state: str
    transcript: list[dict[str, Any]]
    pending_approval: PendingApproval | None
    audit_records: int
    audit_intact: bool
