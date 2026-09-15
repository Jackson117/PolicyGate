"""The deployment's fixed parts: retail's tools, the harness, and the world they act on.

The graph is stateless code; everything a session has done lives in its
checkpointed state as plain JSON. That includes the world. tau2's retail
database is in memory, and a checkpoint that embedded it would carry megabytes
per step -- so instead the state records every tool call that was executed, in
order, and the environment is rebuilt by replaying them against a fresh copy of
the database. The executed calls are already the audit chain's subject; making
them the source of the world state too means the two cannot disagree.

A per-process cache keeps replay to the calls a session has not yet applied.
After a restart the first request for a session replays from the start, which is
what makes resuming on another process correct rather than merely possible.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.registry import registry
from tau2.utils.llm_utils import generate
from tau2.utils.utils import DATA_DIR

from policygate.adapters.tau2_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from policygate.adapters.tool_mapping import (
    RETAIL_ESCALATION,
    RETAIL_IDENTITY_LOOKUPS,
    retail_constraints,
    retail_facts,
    tool_specs,
)
from policygate.harness.config import HarnessConfig
from policygate.harness.gating import ToolGate
from policygate.harness.types import ToolKind
from policygate.harness.validation import Validator

Generate = Callable[..., AssistantMessage]

SERVICE_CONFIG = HarnessConfig.full(name="service", approval_gate_enabled=True)
"""The full harness with the approval gate on -- the one mechanism W4 left out,
because only a deployment has a human to answer it."""


def to_message(data: dict[str, Any]) -> Message:
    role = data.get("role")
    model = {
        "assistant": AssistantMessage,
        "user": UserMessage,
        "tool": ToolMessage,
        "system": SystemMessage,
    }.get(role)
    if model is None:
        raise ValueError(f"unknown message role: {role!r}")
    return model.model_validate(data)


def dump(message: Message) -> dict[str, Any]:
    return json.loads(message.model_dump_json())


class RetailRuntime:
    """Everything a session needs that is not its own state."""

    def __init__(
        self,
        *,
        llm: str = "openai/gpt-4.1-mini",
        llm_args: dict[str, Any] | None = None,
        config: HarnessConfig = SERVICE_CONFIG,
        generate_fn: Generate = generate,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_internal_attempts: int = 3,
    ) -> None:
        self.llm = llm
        self.llm_args = llm_args if llm_args is not None else {"temperature": 0.0}
        self.config = config
        self.generate = generate_fn
        self.clock = clock
        self.max_internal_attempts = max_internal_attempts

        environment = self._fresh_environment()
        self.tools = environment.get_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self.policy = environment.get_policy()
        self.system_prompt = SYSTEM_PROMPT.format(
            domain_policy=self.policy, agent_instruction=AGENT_INSTRUCTION
        )
        self.catalogue = tool_specs(
            environment.tools,
            identity_lookups=RETAIL_IDENTITY_LOOKUPS,
            escalation=RETAIL_ESCALATION,
        )
        db = json.loads((DATA_DIR / "tau2" / "domains" / "retail" / "db.json").read_text())
        self.constraints = retail_constraints(db, tools=self.tools)
        self._specs = {spec.name: spec for spec in self.catalogue}
        self.extract_facts = retail_facts
        self.gate = ToolGate(self.catalogue, config)
        self.validator = Validator(self.constraints, config)

        self._worlds: dict[str, tuple[Any, int]] = {}
        self._lock = threading.Lock()

    def spec(self, name: str):
        """The harness's view of a tool, or None for a name the domain does not have."""
        return self._specs.get(name)

    def is_write(self, name: str) -> bool:
        spec = self.spec(name)
        return spec is not None and spec.kind is ToolKind.WRITE

    @staticmethod
    def _fresh_environment():
        return registry.get_env_constructor("retail")()

    def execute(
        self, session_id: str, already_executed: Sequence[dict[str, Any]], call: Tau2ToolCall
    ) -> ToolMessage:
        """Run one call against the session's world, rebuilding the world if needed."""
        with self._lock:
            environment, applied = self._worlds.get(session_id, (None, -1))
            if environment is None or applied != len(already_executed):
                environment = self._fresh_environment()
                for previous in already_executed:
                    environment.get_response(Tau2ToolCall.model_validate(previous))
            result = environment.get_response(call)
            self._worlds[session_id] = (environment, len(already_executed) + 1)
            return result

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._worlds.pop(session_id, None)
