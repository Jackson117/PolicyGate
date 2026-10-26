"""The audit chain: evidence that an action happened, and was not altered since.

Core contribution #3. A log records what happened; this is meant to stand up
when someone doubts it. Three properties, and each is checkable rather than
asserted:

* **Completeness.** Every action that changed the world has a record.
  :meth:`AuditChain.missing_executions` answers it against the writes that
  actually ran, so a gap is a finding rather than an absence nobody notices.
* **Linkage.** Each record carries what justified it -- the decision and the
  rule that produced it, the policy clauses cited, the approval, and the
  resulting change to the world. A record that cannot be traced back to a rule
  proves only that something occurred.
* **Verifiable order.** Append-only, contiguous, and each record hashes the one
  before it, so altering or reordering history breaks the chain at a nameable
  point. :meth:`AuditChain.verify` reports where.

This module is deliberately outside the ablation matrix. It observes behaviour
without changing it, so ablating it moves no reward and the row would be flat by
construction; it is tested on the three properties above instead.

The clock is injected rather than read. Verification works either way, since the
hash is recomputed from the stored timestamp -- but a chain that calls
``datetime.now()`` cannot be reproduced by re-running the same experiment, and
cannot be asserted on in a test.

Pydantic, deepdiff and the standard library only. See docs/constraints.md #2.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from deepdiff import DeepDiff
from pydantic import BaseModel, ConfigDict

from policygate.harness.config import HarnessConfig
from policygate.harness.types import (
    GENESIS_HASH,
    ApprovalDecision,
    AuditEvent,
    AuditRecord,
    Decision,
    ToolCall,
)


def world_diff(before: Any, after: Any) -> dict[str, Any]:
    """What changed between two snapshots of the world, as plain JSON types.

    Routed through deepdiff's own JSON serialisation because the chain hashes
    this: the diff has to survive a round trip through storage and hash the same
    on the way back, which its native objects do not guarantee.
    """
    return json.loads(DeepDiff(before, after).to_json())


class BreakKind(StrEnum):
    """How a chain failed verification. Named so a report can say what is wrong."""

    CONTENT_TAMPERED = "content_tampered"
    """A record's stored hash no longer matches the content it covers."""

    LINK_BROKEN = "link_broken"
    """A record's prev_hash is not the hash of the record before it."""

    SEQUENCE_BROKEN = "sequence_broken"
    """Sequence numbers are not contiguous from zero -- a record was removed or reordered."""

    GENESIS_WRONG = "genesis_wrong"
    """The first record does not start from the genesis hash, so the head is missing."""


class ChainBreak(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    """Position in the sequence handed to verification, not the record's own seq,
    which is itself one of the things under suspicion."""

    kind: BreakKind
    detail: str


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    intact: bool
    breaks: tuple[ChainBreak, ...] = ()

    @property
    def first_break(self) -> ChainBreak | None:
        return self.breaks[0] if self.breaks else None


def verify_records(records: Sequence[AuditRecord]) -> VerificationResult:
    """Check a chain, however it was obtained -- built here or loaded from storage.

    Every record is examined rather than stopping at the first problem: a report
    that names one break when there are three invites the assumption that fixing
    it restores the chain.
    """
    breaks: list[ChainBreak] = []
    expected_prev = GENESIS_HASH

    for index, record in enumerate(records):
        if not record.is_self_consistent():
            breaks.append(
                ChainBreak(
                    index=index,
                    kind=BreakKind.CONTENT_TAMPERED,
                    detail=(
                        f"record {record.seq} hashes to {record.compute_hash()[:12]}... "
                        f"but stores {record.record_hash[:12]}..."
                    ),
                )
            )
        if record.prev_hash != expected_prev:
            kind = BreakKind.GENESIS_WRONG if index == 0 else BreakKind.LINK_BROKEN
            breaks.append(
                ChainBreak(
                    index=index,
                    kind=kind,
                    detail=(
                        f"record {record.seq} expects to follow {expected_prev[:12]}... "
                        f"but names {record.prev_hash[:12]}..."
                    ),
                )
            )
        if record.seq != index:
            breaks.append(
                ChainBreak(
                    index=index,
                    kind=BreakKind.SEQUENCE_BROKEN,
                    detail=f"record at position {index} claims seq {record.seq}",
                )
            )
        expected_prev = record.record_hash

    return VerificationResult(intact=not breaks, breaks=tuple(breaks))


class AuditChain:
    """An append-only, hash-linked record of one session.

    There is no update or delete, and :attr:`records` hands back a tuple: the
    only supported way to change what the chain says is to append to it, which
    is what makes a later contradiction detectable.
    """

    def __init__(
        self,
        session_id: str,
        clock: Callable[[], datetime],
        config: HarnessConfig,
    ) -> None:
        self._session_id = session_id
        self._clock = clock
        self._config = config
        self._records: list[AuditRecord] = []

    @classmethod
    def resume(
        cls,
        session_id: str,
        clock: Callable[[], datetime],
        config: HarnessConfig,
        records: Sequence[AuditRecord],
    ) -> AuditChain:
        """Continue a chain persisted elsewhere -- a checkpoint, a database row set.

        Refuses anything that does not verify, or that belongs to another session:
        appending to a broken chain would give later records a valid-looking link
        to evidence that has already been altered.
        """
        verification = verify_records(records)
        if not verification.intact:
            raise ValueError(f"refusing to resume a broken chain: {verification.first_break}")
        foreign = {r.session_id for r in records} - {session_id}
        if foreign:
            raise ValueError(f"records belong to other sessions: {sorted(foreign)}")
        chain = cls(session_id, clock, config)
        chain._records = list(records)
        return chain

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def enabled(self) -> bool:
        return self._config.audit_enabled

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    @property
    def head_hash(self) -> str:
        return self._records[-1].record_hash if self._records else GENESIS_HASH

    def append(
        self,
        event: AuditEvent,
        *,
        tool_call: ToolCall | None = None,
        decision: Decision | None = None,
        approval: ApprovalDecision | None = None,
        diff: Mapping[str, Any] | None = None,
    ) -> AuditRecord | None:
        """Add a record and return it, or return None when auditing is switched off.

        The switch genuinely records nothing rather than recording into a void,
        so an arm with auditing disabled carries none of its cost.
        """
        if not self._config.audit_enabled:
            return None

        draft = AuditRecord(
            seq=len(self._records),
            timestamp=self._clock(),
            event=event,
            session_id=self._session_id,
            tool_call=tool_call,
            decision=decision,
            approval=approval,
            world_diff=diff,
            prev_hash=self.head_hash,
            record_hash="",
        )
        record = draft.model_copy(update={"record_hash": draft.compute_hash()})
        self._records.append(record)
        return record

    def verify(self) -> VerificationResult:
        return verify_records(self._records)

    # --- completeness ------------------------------------------------------

    def executed_fingerprints(self) -> tuple[str, ...]:
        """Fingerprints of the actions this chain says were carried out."""
        return tuple(
            r.tool_call.fingerprint()
            for r in self._records
            if r.event is AuditEvent.TOOL_EXECUTED and r.tool_call is not None
        )

    def missing_executions(self, executed: Iterable[ToolCall]) -> tuple[ToolCall, ...]:
        """Writes that really ran but which the chain does not account for.

        Completeness cannot be established from the inside: a chain with no gaps
        in its own numbering is still incomplete if something happened that was
        never recorded. This compares it against what actually ran.
        """
        recorded = set(self.executed_fingerprints())
        return tuple(call for call in executed if call.fingerprint() not in recorded)

    # --- linkage -----------------------------------------------------------

    def unjustified_executions(self) -> tuple[AuditRecord, ...]:
        """Executions recorded without the decision that permitted them.

        A record proving only that something occurred is a log entry. Linkage is
        the property that separates the two.
        """
        return tuple(
            r
            for r in self._records
            if r.event is AuditEvent.TOOL_EXECUTED and r.decision is None
        )
