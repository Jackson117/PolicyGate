"""State gating: the legal action set, as a function of session state.

Core contribution #1. The claim it implements is narrow and testable: an
unmodified agent picks its next action from the whole conversation, so a
paraphrase can change which tool gets called -- W1 measured a 53% run-to-run
flip rate on the DB component at temperature 0, with the forks clustered around
the write operations. Gating moves the basis for that choice from the text to
facts that wording cannot touch.

Gating acts twice, and the two are worth keeping distinct when reporting:

* **Prevention.** :meth:`ToolGate.visible_tools` narrows the schemas the model
  is shown, so a wrong turn is never on offer. This is where the mechanism is
  supposed to earn its keep.
* **Enforcement.** :meth:`ToolGate.review_turn` refuses a proposal that breaks
  the rules anyway. A model can still name a tool it was not shown, and
  enforcement counts are the honest measure of how often prevention leaked.

Rules are grounded in the domains' policy text, not invented. Retail: identity
must be established by a lookup before anything else, at most one tool call per
turn, transfer only when the request is out of scope. Airline carries the same
skeleton without the authentication step -- see
:class:`~policygate.harness.state.DomainStateProfile`. banking_knowledge
verifies in two steps, a lookup and then a logged verification, and lets
general questions through unauthenticated; the roles
:attr:`~policygate.harness.types.ToolRole.IDENTITY_COMMIT` and
:attr:`~policygate.harness.types.ToolRole.SUBJECT_INDEPENDENT` carry that, and a
catalogue without them gates exactly as before.

Pydantic and the standard library only. See docs/constraints.md #2.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from policygate.harness.config import HarnessConfig
from policygate.harness.state import SessionState
from policygate.harness.types import (
    ClauseRef,
    Decision,
    DecisionOutcome,
    ToolCall,
    ToolKind,
    ToolRole,
    ToolSpec,
)


class GateRule(StrEnum):
    """Why the gate decided what it decided.

    Every decision names one of these. Ablation tables are built by counting
    decisions per rule, so a verdict that cannot say which rule produced it is
    not measurable and therefore not evidence.
    """

    ALLOWED = "gating.allowed"
    DISABLED = "gating.disabled"
    """Gating is switched off for this arm: everything passes, unexamined."""

    UNKNOWN_TOOL = "gating.unknown_tool"
    IDENTITY_FIRST = "gating.identity_first"
    WRITE_REQUIRES_SCOPE = "gating.write_requires_scope"
    FROZEN_PENDING_APPROVAL = "gating.frozen_pending_approval"
    SESSION_ENDED = "gating.session_ended"
    ONE_CALL_PER_TURN = "gating.one_call_per_turn"
    IDENTITY_ALREADY_ESTABLISHED = "gating.identity_already_established"
    """An identity commit proposed after the session is verified."""


class TurnReview(BaseModel):
    """The gate's verdict on everything a model proposed in one turn."""

    model_config = ConfigDict(frozen=True)

    decisions: tuple[Decision, ...]
    """One per proposed call, in the order proposed."""

    allowed_calls: tuple[ToolCall, ...]
    """The calls that may proceed. Empty when the turn is refused as a whole."""

    @property
    def refused(self) -> bool:
        return any(d.outcome is not DecisionOutcome.ALLOW for d in self.decisions)


_BEFORE_VERIFICATION = frozenset(
    {ToolRole.IDENTITY_LOOKUP, ToolRole.IDENTITY_COMMIT, ToolRole.ESCALATION}
)
"""Roles legal before verification whatever their kind."""


class ToolGate:
    """Decides which tools exist, and which proposals survive, in a given state.

    Domain-agnostic by construction: it is handed a catalogue of
    :class:`~policygate.harness.types.ToolSpec` and knows nothing about retail,
    orders or tau2. Building that catalogue -- reading tau2's ``__tool_type__``
    and declaring which tool authenticates -- is the adapter's job.
    """

    def __init__(
        self,
        catalogue: Iterable[ToolSpec],
        config: HarnessConfig,
        clauses: Mapping[GateRule, tuple[ClauseRef, ...]] | None = None,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in catalogue:
            if spec.name in self._tools:
                raise ValueError(f"duplicate tool in catalogue: {spec.name!r}")
            self._tools[spec.name] = spec
        self._config = config
        self._clauses = dict(clauses or {})

    @property
    def catalogue(self) -> Mapping[str, ToolSpec]:
        return dict(self._tools)

    # --- prevention --------------------------------------------------------

    def visible_tools(self, state: SessionState) -> frozenset[str]:
        """Tool names a call may use in ``state`` -- what enforcement admits.

        With gating off this is the whole catalogue, which is what makes the
        baseline arm a genuine control rather than a differently-configured
        treatment.
        """
        if not self._config.gating_enabled:
            return frozenset(self._tools)
        return self.state_tools(state)

    def state_tools(self, state: SessionState) -> frozenset[str]:
        """Tool names that fit ``state``, whatever the gating switch says.

        Split from :meth:`visible_tools` so that context pruning -- hiding
        schemas from the model -- can run without gating, which refuses calls.
        Before the split pruning read the switched method, so an arm with
        pruning on and gating off pruned nothing, and could not be measured.
        """
        if state in (SessionState.TRANSFERRED, SessionState.PENDING_APPROVAL):
            # Terminal, or frozen awaiting a decision. Nothing proceeds -- not
            # even escalation, because the adapter resolves the pending approval
            # before the model is given another turn.
            return frozenset()

        if state is SessionState.UNVERIFIED:
            # policy.md: identity must be established first, "even when the user
            # already provides the user id". Reads are withheld because they
            # return the profile and orders of a subject nobody has confirmed.
            # Subject-independent tools are not about anyone, so they stay, unless
            # they write.
            return frozenset(
                name
                for name, spec in self._tools.items()
                if spec.role in _BEFORE_VERIFICATION
                or (spec.role is ToolRole.SUBJECT_INDEPENDENT and spec.kind is not ToolKind.WRITE)
            )

        # From here on identity is settled, and committing it again would log a
        # second verification nobody performed.
        live = {
            name: spec
            for name, spec in self._tools.items()
            if spec.role is not ToolRole.IDENTITY_COMMIT
        }

        if state is SessionState.SCOPED:
            return frozenset(live)

        if state is SessionState.VERIFIED:
            # VERIFIED: reads are open, but no unique write target exists yet, so a
            # write would have to guess its referent -- which is what task 14 was.
            return frozenset(
                name for name, spec in live.items() if spec.kind is not ToolKind.WRITE
            )

        raise AssertionError(f"{state} is not a legal SessionState")

    # --- enforcement -------------------------------------------------------

    def review_call(self, call: ToolCall, state: SessionState) -> Decision:
        """Judge one proposed call, naming the rule that decided it."""
        if not self._config.gating_enabled:
            return self._allow(GateRule.DISABLED, "gating is disabled for this arm")

        spec = self._tools.get(call.name)
        if spec is None:
            return self._deny(
                GateRule.UNKNOWN_TOOL,
                f"{call.name!r} is not a tool in this domain",
                remedy="choose a tool from the ones offered in this turn",
            )

        if call.name in self.visible_tools(state):
            return self._allow(GateRule.ALLOWED, f"{call.name!r} is available in state {state}")

        rule, reason, remedy = self._why_not(spec, state)
        return self._deny(rule, reason, remedy=remedy)

    def review_turn(self, calls: Iterable[ToolCall], state: SessionState) -> TurnReview:
        """Judge everything proposed in one turn.

        The one-call-per-turn rule refuses the whole turn rather than keeping
        the first call and dropping the rest. Both retail and airline state it
        ("You should at most make one tool call at a time"), and W1 found it
        violated in 78% of runs. Executing part of what was proposed would leave
        the world half-changed against an intent the model never expressed --
        the partial-application problem, which is far worse than spending a turn.

        This rule has its own switch: it stays in force with gating disabled, so
        the two effects can be attributed separately.
        """
        calls = tuple(calls)

        if self._config.parallel_call_block_enabled and len(calls) > 1:
            decision = self._deny(
                GateRule.ONE_CALL_PER_TURN,
                f"{len(calls)} tool calls proposed in one turn; policy allows at most one",
                remedy=(
                    f"propose only {calls[0].name!r} now; its result comes back before the "
                    "next call, and the others can follow one per turn"
                ),
            )
            return TurnReview(decisions=(decision,) * len(calls), allowed_calls=())

        decisions = tuple(self.review_call(c, state) for c in calls)
        allowed = tuple(c for c, d in zip(calls, decisions, strict=True) if d.allowed)
        return TurnReview(decisions=decisions, allowed_calls=allowed)

    # --- internals ---------------------------------------------------------

    def _why_not(self, spec: ToolSpec, state: SessionState) -> tuple[GateRule, str, str]:
        """Attribute a refusal to a specific rule, not a generic 'not visible'.

        Returns the rule, why it fired, and what to do next -- the last of these
        separately, because W4 v3 found the model stops acting when told only why.
        """
        if state is SessionState.TRANSFERRED:
            return (
                GateRule.SESSION_ENDED,
                "the conversation has been handed to a human agent",
                "make no further tool calls; the human agent takes over from here",
            )
        if state is SessionState.PENDING_APPROVAL:
            return (
                GateRule.FROZEN_PENDING_APPROVAL,
                "a write is awaiting approval; the session is paused until it resolves",
                "wait for the approval decision before proposing anything else",
            )
        if state is SessionState.UNVERIFIED:
            lookups = sorted(
                name for name, s in self._tools.items() if s.role is ToolRole.IDENTITY_LOOKUP
            )
            commits = sorted(
                name for name, s in self._tools.items() if s.role is ToolRole.IDENTITY_COMMIT
            )
            remedy = (
                f"call one of {lookups} to establish who the user is, then propose "
                f"{spec.name!r} again"
                if lookups
                else f"establish the user's identity, then propose {spec.name!r} again"
            )
            if commits:
                # Only domains that log verification as a separate step say so;
                # the wording for every other domain is left exactly as it was.
                find = f"look the user up with one of {lookups}, " if lookups else ""
                remedy = (
                    f"{find}verify their identity as the policy requires, record it with "
                    f"one of {commits}, then propose {spec.name!r} again"
                )
            return (
                GateRule.IDENTITY_FIRST,
                f"{spec.name!r} needs an authenticated subject; "
                "resolve the user's identity with a lookup first",
                remedy,
            )
        if spec.role is ToolRole.IDENTITY_COMMIT:
            return (
                GateRule.IDENTITY_ALREADY_ESTABLISHED,
                f"{spec.name!r} records a verification, and this session's identity is "
                "already established",
                "do not verify again; continue with the user's request",
            )
        if spec.kind is ToolKind.WRITE:
            return (
                GateRule.WRITE_REQUIRES_SCOPE,
                f"{spec.name!r} writes to the database but exactly one target is not yet "
                "in scope; narrow to a single object before proposing a write",
                (
                    "retrieve the one object this should act on -- ask the user which it is "
                    f"if that is unclear -- and then propose {spec.name!r} again"
                ),
            )
        raise AssertionError(  # pragma: no cover - defensive
            f"{spec.name!r} is not visible in {state} and no rule explains why"
        )

    def _allow(self, rule: GateRule, reason: str) -> Decision:
        return Decision(
            outcome=DecisionOutcome.ALLOW,
            rule=rule.value,
            reason=reason,
            clauses=self._clauses.get(rule, ()),
        )

    def _deny(self, rule: GateRule, reason: str, *, remedy: str | None = None) -> Decision:
        return Decision(
            outcome=DecisionOutcome.DENY,
            rule=rule.value,
            reason=reason,
            remedy=remedy,
            clauses=self._clauses.get(rule, ()),
        )
