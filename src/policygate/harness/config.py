"""Harness configuration -- the pivot the whole ablation rests on.

Every mechanism gets its own switch, because the ablation matrix is generated
from this class rather than hand-listed somewhere else. A mechanism without a
switch cannot produce a row, and a row that has to be added by hand is a row
someone will forget.

Where W1 evidence splits one architectural module into several independent
mechanisms, the switches are split too. "Precondition checking" covers at least
three distinct things in the retail traces -- non-canonical enum values
(task 22), an ambiguous referent (task 14), and preconditions the environment
itself rejects (31x "Order not found") -- and folding them into one flag would
make it impossible to say which of them earned any improvement.

Pydantic and the standard library only. See CLAUDE.md constraint #2.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from policygate.harness.types import ToolKind


class LoopGuardConfig(BaseModel):
    """Budgets that stop a session running away.

    W1 saw zero non-normal terminations in 90 simulations, so these are
    insurance rather than a measured need. Defaults are set well clear of
    observed behaviour (longest run: 13 tool calls) so that turning guards on
    does not silently truncate normal work and get scored as a harness effect.
    """

    model_config = ConfigDict(frozen=True)

    max_tool_calls: int = Field(default=40, gt=0)
    max_wall_clock_seconds: float = Field(default=600.0, gt=0)
    max_repeated_actions: int = Field(default=3, gt=0)
    """How many times the same action fingerprint may be proposed before it is refused."""


class ApprovalPolicy(StrEnum):
    """Who decides, when the gate pauses a write.

    ORACLE hands the agent a free correctness checker and will inflate any
    end-to-end score, so results under it are only interpretable next to the
    gate trigger rate. See :class:`ApprovalConfig`.
    """

    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    ORACLE = "oracle"
    NOISY_ORACLE = "noisy_oracle"


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy: ApprovalPolicy = ApprovalPolicy.AUTO_APPROVE
    noise_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Probability the noisy oracle inverts its answer. Ignored by other policies."""

    require_for_kinds: tuple[ToolKind, ...] = (ToolKind.WRITE,)
    """Which tool kinds pause for approval. Writes only, by default."""

    @model_validator(mode="after")
    def _noise_only_for_noisy_oracle(self) -> ApprovalConfig:
        if self.noise_rate > 0 and self.policy is not ApprovalPolicy.NOISY_ORACLE:
            raise ValueError(
                f"noise_rate is only meaningful for {ApprovalPolicy.NOISY_ORACLE.value}, "
                f"got policy={self.policy.value}"
            )
        return self


class HarnessConfig(BaseModel):
    """Which harness mechanisms are active for a run.

    Construct experiment arms with :meth:`baseline`, :meth:`full` and
    :meth:`without`; do not hand-assemble arms at call sites, or the recorded
    config and the actual behaviour will drift apart.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "unnamed"

    # --- contribution 1: state gating -------------------------------------
    gating_enabled: bool = False
    """Narrow the visible toolset by session state, eliminating wrong turns up front."""

    context_pruning_enabled: bool = False
    """Trim policy text and tool schemas to the current state. Also cuts tokens."""

    # --- validation, split by mechanism (each has its own W1 evidence) -----
    schema_check_enabled: bool = False
    """Strict argument validation against the tool signature."""

    value_canonicalization_enabled: bool = False
    """Reject non-canonical enum values. Evidence: task 22, "United States" vs "USA"."""

    referent_uniqueness_enabled: bool = False
    """Refuse a write while several candidate objects match. Evidence: task 14."""

    precondition_check_enabled: bool = False
    """Check business preconditions before proposing. Evidence: 31x "Order not found"."""

    parallel_call_block_enabled: bool = False
    """Enforce retail policy.md: at most one tool call per turn. Violated in 78% of W1 runs."""

    # --- contribution 2: idempotent execution -----------------------------
    idempotency_enabled: bool = False
    """Idempotency keys, replay detection, first-result caching.

    Deferred past W3 by decision: W1 found zero duplicate write calls across 90
    simulations, so on this domain the mechanism has no events to act on and its
    ablation row would be flat by construction. The switch exists so that the
    row appears the moment a domain does exercise it.
    """

    # --- failure handling --------------------------------------------------
    classified_retry_enabled: bool = False
    """Distinguish retryable failures (timeout, transport) from business refusals."""

    compensation_enabled: bool = False
    """Roll back or compensate a partially applied write."""

    # --- approval gate -----------------------------------------------------
    approval_gate_enabled: bool = False
    """Pause writes for an approval decision. Not in the model's action space."""

    approval: ApprovalConfig = ApprovalConfig()

    # --- contribution 3: audit chain ---------------------------------------
    audit_enabled: bool = False
    """Hash-chained audit records.

    Excluded from :meth:`ablation_switches` on purpose: it observes behaviour
    without changing it, so ablating it moves no reward and the row would be
    flat by construction. It is tested on its own properties -- completeness,
    linkage, tamper-evidence -- not through the reward.
    """

    loop_guards: LoopGuardConfig = LoopGuardConfig()
    loop_guards_enabled: bool = False

    seed: int = 0

    # --- arm construction --------------------------------------------------

    NON_ABLATABLE: ClassVar[tuple[str, ...]] = ("audit_enabled",)
    """Switches excluded from the matrix, with the reason stated on each field."""

    @classmethod
    def baseline(cls) -> HarnessConfig:
        """Every mechanism off: an unmodified agent, the control arm."""
        return cls(name="baseline")

    @classmethod
    def full(cls, **overrides: object) -> HarnessConfig:
        """Everything the current milestone implements, switched on.

        Mechanisms deferred by schedule (idempotency) or not yet written stay
        off here, so that `full` always describes an arm that can actually run.
        """
        return cls(
            name="full",
            gating_enabled=True,
            context_pruning_enabled=True,
            schema_check_enabled=True,
            value_canonicalization_enabled=True,
            referent_uniqueness_enabled=True,
            precondition_check_enabled=True,
            parallel_call_block_enabled=True,
            classified_retry_enabled=True,
            compensation_enabled=True,
            approval_gate_enabled=True,
            audit_enabled=True,
            loop_guards_enabled=True,
            **overrides,  # type: ignore[arg-type]
        )

    @classmethod
    def ablation_switches(cls) -> tuple[str, ...]:
        """Boolean fields whose value can change agent behaviour.

        Derived from the model's own fields so that adding a mechanism adds its
        ablation row automatically.
        """
        return tuple(
            n
            for n, f in cls.model_fields.items()
            if f.annotation is bool and n not in cls.NON_ABLATABLE
        )

    def enabled_ablation_switches(self) -> tuple[str, ...]:
        """The switches this arm actually turns on -- the rows worth running."""
        return tuple(n for n in self.ablation_switches() if getattr(self, n) is True)

    def without(self, *switches: str) -> HarnessConfig:
        """This arm with the named mechanisms disabled (leave-one-out ablation)."""
        unknown = [s for s in switches if s not in self.ablation_switches()]
        if unknown:
            raise ValueError(f"not ablatable switches: {unknown}")
        off = dict.fromkeys(switches, False)
        return self.model_copy(update={"name": f"{self.name}-no_{'_'.join(switches)}", **off})