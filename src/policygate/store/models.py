"""Tables for the deployment's audit store.

CLAUDE.md constraint 3: this is its own database, never the world-state store the
agent acts on. In the demo the world is tau2's in-memory retail database; the
audit store is Postgres. Nothing here can reach the former.

Only two tables. An audit row is one hash-chained record, stored whole so the
chain can be re-verified from the database alone; an approval row is one paused
write, so a reviewer can find what is waiting without walking every session's
checkpoint.

Schema is created with ``create_all``. There is one version of it and no data to
migrate; Alembic would be ceremony until there is.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AuditRow(Base):
    __tablename__ = "audit_records"
    __table_args__ = (UniqueConstraint("session_id", "seq", name="uq_audit_session_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    event: Mapped[str] = mapped_column(String(32))
    record_hash: Mapped[str] = mapped_column(String(64))
    prev_hash: Mapped[str] = mapped_column(String(64))
    record: Mapped[dict] = mapped_column(JSON)
    """The full AuditRecord, so verification needs nothing but this table."""


class ApprovalRow(Base):
    __tablename__ = "approvals"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    arguments: Mapped[dict] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(Text)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
