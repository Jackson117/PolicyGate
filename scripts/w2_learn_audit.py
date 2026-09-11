from __future__ import annotations

import json
from collections.abc import Iterable, Sequence, Mapping, Callable
from datetime import datetime
from enum import StrEnum
from typing import Any

from deepdiff import DeepDiff
from pydantic import BaseModel, ConfigDict

from policygate.harness.audit import VerificationResult, verify_records
from policygate.harness.config import HarnessConfig
from policygate.harness.types import (
    GENESIS_HASH,
    ApprovalDecision,
    AuditEvent,
    AuditRecord,
    Decision,
    ToolCall
)

class AuditChain:
    def __init__(
        self,
        session_id: str,
        clock: Callable[[], datetime],
        config: HarnessConfig,
    ) -> None:
        self._session_id = session_id
        self._clock = clock,
        self._config = config
        self._records: list[AuditRecord] = []
    
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
        call: ToolCall | None = None,
        decision: Decision | None = None,
        approval: ApprovalDecision | None = None,
        diff: Mapping[str, Any] | None = None,
    ) -> AuditRecord | None:
        if not self._config.audit_enabled:
            return None
        
        draft = AuditRecord(
            seq=len(self._records),
            timestamp=self._clock(),
            event=event,
            session_id=self._session_id,
            tool_call=call,
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
    
    