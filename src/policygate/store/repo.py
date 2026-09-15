"""Writes and reads against the audit store."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from policygate.harness.audit import VerificationResult, verify_records
from policygate.harness.types import ApprovalRequest, ApprovalStatus, AuditRecord
from policygate.store.models import ApprovalRow, AuditRow, Base


class AuditStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)
        Base.metadata.create_all(engine)

    @classmethod
    def from_url(cls, url: str) -> AuditStore:
        return cls(create_engine(url, future=True))

    # --- audit records ---------------------------------------------------

    def append(self, records: Sequence[AuditRecord]) -> int:
        """Store records not already stored. Returns how many were new.

        Idempotent by (session, seq): a graph step replayed after a crash writes
        the same records again, and the second write must not duplicate them --
        or, if the content differs, must not silently keep either.
        """
        if not records:
            return 0
        with self._sessions.begin() as db:
            session_ids = {r.session_id for r in records}
            existing = {
                (row.session_id, row.seq): row.record_hash
                for row in db.scalars(
                    select(AuditRow).where(AuditRow.session_id.in_(session_ids))
                )
            }
            added = 0
            for record in records:
                key = (record.session_id, record.seq)
                if key in existing:
                    if existing[key] != record.record_hash:
                        raise ValueError(
                            f"record {key} already stored with different content; "
                            "refusing to overwrite audit evidence"
                        )
                    continue
                db.add(
                    AuditRow(
                        session_id=record.session_id,
                        seq=record.seq,
                        event=record.event.value,
                        record_hash=record.record_hash,
                        prev_hash=record.prev_hash,
                        record=record.model_dump(mode="json"),
                    )
                )
                added += 1
            return added

    def records(self, session_id: str) -> tuple[AuditRecord, ...]:
        with self._sessions() as db:
            rows = db.scalars(
                select(AuditRow).where(AuditRow.session_id == session_id).order_by(AuditRow.seq)
            )
            return tuple(AuditRecord.model_validate(row.record) for row in rows)

    def verify(self, session_id: str) -> VerificationResult:
        return verify_records(self.records(session_id))

    # --- approvals -------------------------------------------------------

    def open_approval(self, session_id: str, request: ApprovalRequest, at: datetime) -> None:
        with self._sessions.begin() as db:
            if db.get(ApprovalRow, request.request_id) is not None:
                return  # the interrupting node re-ran on resume; already recorded
            db.add(
                ApprovalRow(
                    request_id=request.request_id,
                    session_id=session_id,
                    status=ApprovalStatus.PENDING.value,
                    tool_name=request.tool_call.name,
                    arguments=dict(request.tool_call.arguments),
                    reason=request.reason,
                    requested_at=at,
                )
            )

    def resolve_approval(
        self, request_id: str, status: ApprovalStatus, reviewer: str, note: str | None, at: datetime
    ) -> None:
        with self._sessions.begin() as db:
            row = db.get(ApprovalRow, request_id)
            if row is None:
                raise KeyError(request_id)
            if row.status != ApprovalStatus.PENDING.value:
                raise ValueError(f"approval {request_id} is already {row.status}")
            row.status = status.value
            row.reviewer = reviewer
            row.note = note
            row.resolved_at = at

    def pending(self, session_id: str | None = None) -> list[ApprovalRow]:
        with self._sessions() as db:
            query = select(ApprovalRow).where(ApprovalRow.status == ApprovalStatus.PENDING.value)
            if session_id is not None:
                query = query.where(ApprovalRow.session_id == session_id)
            return list(db.scalars(query.order_by(ApprovalRow.requested_at)))

    def approval(self, request_id: str) -> ApprovalRow | None:
        with self._sessions() as db:
            return db.get(ApprovalRow, request_id)

    def session(self) -> Session:
        return self._sessions()
