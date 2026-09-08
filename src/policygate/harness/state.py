"""Session state: what the harness knows to be true, and what may follow.

This module is the reason gating works. An unmodified agent chooses its next
action from the whole conversation, so a paraphrase in the user's wording feeds
straight through into which tool gets called -- which is what W1 measured as a
53% run-to-run flip rate on the DB component at temperature 0. Here the next
action is constrained by *state*, and state is a function of facts established
by tool results: has an identity been resolved, is there exactly one target in
scope, is a write awaiting approval.

Wording cannot change those facts, so wording cannot change the legal action
set. That is the whole mechanism.

Two consequences are load-bearing and are enforced below rather than left as
comments:

* Facts come from tool *results*, never from what anyone said. The retail
  policy is explicit that identity must be established by a lookup "even when
  the user already provides the user id", so :meth:`SessionTracker.record_identity`
  is meant to be fed a tool result and nothing else.
* State is derived, never assigned. :func:`derive_state` is a pure function of
  :class:`SessionFacts`; there is no setter. An illegal state is therefore not
  something a caller can reach by mistake -- it can only come from an illegal
  fact update, and those are refused at the point of update.

Pydantic and the standard library only. See CLAUDE.md constraint #2.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class SessionState(StrEnum):
    """Where a conversation stands, as far as the harness can prove.

    Deliberately coarse. Every distinction here has to be decidable from tool
    results alone; anything needing the text of the conversation belongs in a
    validator, not in the state.
    """

    UNVERIFIED = "unverified"
    """No identity resolved. Only identity lookups and transfer are legitimate."""

    VERIFIED = "verified"
    """An identity is bound. Reads are open; no unique write target yet."""

    SCOPED = "scoped"
    """Exactly one write target is in scope. A write may now be proposed."""

    PENDING_APPROVAL = "pending_approval"
    """A proposed write is paused at the gate. Nothing proceeds until resolved."""

    TRANSFERRED = "transferred"
    """Handed to a human. Terminal: the conversation is over for the agent."""


class StateError(Exception):
    """An illegal fact update.

    This is the one place the harness raises rather than returning a
    :class:`~policygate.harness.types.CheckResult`. A model proposing something
    invalid is normal operation and gets a structured refusal; a caller trying
    to rebind the session to a second user is a bug in the adapter, and hiding
    it behind a result object would let it pass silently.
    """


class SessionFacts(BaseModel):
    """Everything the harness has established, and the only input to state.

    Immutable: updates produce a new instance, so a fact set can be recorded in
    an audit record and still mean the same thing later.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str | None = None
    """Resolved by an identity lookup tool. Never taken from what the user said."""

    candidate_targets: frozenset[str] = frozenset()
    """Referents that could be the object of a write -- order ids, typically.

    Zero means nothing is in scope yet; more than one means the referent is
    ambiguous, which is what task 14 looked like when the agent returned items
    from a different order than the one under discussion. Both cases keep the
    session out of SCOPED, so neither can reach a write.
    """

    pending_approval_id: str | None = None
    transferred: bool = False

    @model_validator(mode="after")
    def _targets_require_an_identity(self) -> SessionFacts:
        # Scoping a target before knowing whose it is would let a write be
        # prepared against another user's order, which the policy forbids
        # outright ("must deny any requests for tasks related to any other user").
        if self.candidate_targets and self.user_id is None:
            raise StateError("cannot scope a target before an identity is resolved")
        return self


def derive_state(facts: SessionFacts) -> SessionState:
    """The session's state. A pure function of facts -- nothing else.

    Order matters: transfer is terminal and outranks everything, and a pending
    approval freezes the session wherever it stood.
    """
    if facts.transferred:
        return SessionState.TRANSFERRED
    if facts.pending_approval_id is not None:
        return SessionState.PENDING_APPROVAL
    if facts.user_id is None:
        return SessionState.UNVERIFIED
    if len(facts.candidate_targets) == 1:
        return SessionState.SCOPED
    return SessionState.VERIFIED


ALLOWED_TRANSITIONS: Mapping[SessionState, frozenset[SessionState]] = {
    SessionState.UNVERIFIED: frozenset({SessionState.VERIFIED, SessionState.TRANSFERRED}),
    SessionState.VERIFIED: frozenset(
        {SessionState.SCOPED, SessionState.TRANSFERRED}
    ),
    SessionState.SCOPED: frozenset(
        {SessionState.VERIFIED, SessionState.PENDING_APPROVAL, SessionState.TRANSFERRED}
    ),
    SessionState.PENDING_APPROVAL: frozenset(
        {SessionState.SCOPED, SessionState.VERIFIED, SessionState.TRANSFERRED}
    ),
    SessionState.TRANSFERRED: frozenset(),
}
"""Which state may follow which.

Notably absent: UNVERIFIED -> SCOPED (a target cannot be scoped before an
identity exists) and anything at all out of TRANSFERRED (the agent's turn in the
conversation has ended).

SCOPED -> VERIFIED is present because scope legitimately widens again when a
second candidate order comes into play; that drops the session out of write
range, which is the intended behaviour rather than an error.
"""


def is_legal_transition(before: SessionState, after: SessionState) -> bool:
    """Whether ``before`` may be followed by ``after``. Staying put is always legal."""
    return after is before or after in ALLOWED_TRANSITIONS[before]


class StateTransition(BaseModel):
    """One state change, with what caused it. Feeds the audit chain."""

    model_config = ConfigDict(frozen=True)

    before: SessionState
    after: SessionState
    trigger: str
    facts: SessionFacts


class SessionTracker:
    """Accumulates facts and reports the state they imply.

    The adapter owns the mapping from a domain's tool results to these calls;
    this class knows nothing about retail, orders or tau2, which is what lets it
    be unit-tested without an LLM and reused across domains.
    """

    def __init__(self, facts: SessionFacts | None = None) -> None:
        self._facts = facts or SessionFacts()
        self._history: list[StateTransition] = []

    @property
    def facts(self) -> SessionFacts:
        return self._facts

    @property
    def state(self) -> SessionState:
        return derive_state(self._facts)

    @property
    def history(self) -> tuple[StateTransition, ...]:
        """Transitions so far, oldest first. Steps that changed nothing are not recorded."""
        return tuple(self._history)

    def _update(self, trigger: str, **changes: object) -> SessionState:
        before = self.state
        if before is SessionState.TRANSFERRED:
            # Terminal means terminal. Without this the update would be waved
            # through as a no-op, because `transferred` dominates derive_state
            # and the state would not appear to change -- so facts would keep
            # accumulating after handoff and the audit trail would show
            # activity in a session that had already ended.
            raise StateError(f"session was transferred to a human; refusing {trigger!r}")
        candidate = self._facts.model_copy(update=changes)
        # Re-run validators: model_copy skips them, and the identity invariant
        # is exactly the kind of thing that must not be bypassable.
        candidate = SessionFacts.model_validate(candidate.model_dump())
        after = derive_state(candidate)
        if not is_legal_transition(before, after):
            raise StateError(f"illegal transition {before} -> {after} (trigger: {trigger})")
        self._facts = candidate
        if after is not before:
            self._history.append(
                StateTransition(before=before, after=after, trigger=trigger, facts=candidate)
            )
        return after

    def record_identity(self, user_id: str) -> SessionState:
        """Bind the session to a user, from an identity lookup's result.

        Re-binding to a different user is refused: the policy allows one user
        per conversation and requires denying anything concerning another.
        """
        if self._facts.user_id is not None and self._facts.user_id != user_id:
            raise StateError(
                f"session is bound to {self._facts.user_id!r}; "
                f"one user per conversation, refusing to rebind to {user_id!r}"
            )
        return self._update("identity_resolved", user_id=user_id)

    def record_targets(self, targets: Iterable[str]) -> SessionState:
        """Replace the set of candidate write targets.

        Replacement rather than accumulation: the candidate set describes what
        is in scope *now*, and a narrowing question that rules orders out has to
        be able to shrink it, otherwise nothing could ever reach SCOPED.
        """
        return self._update("targets_updated", candidate_targets=frozenset(targets))

    def open_approval(self, request_id: str) -> SessionState:
        if self._facts.pending_approval_id is not None:
            raise StateError(
                f"approval {self._facts.pending_approval_id!r} is already pending; "
                "the gate holds one write at a time"
            )
        return self._update("approval_opened", pending_approval_id=request_id)

    def close_approval(self) -> SessionState:
        if self._facts.pending_approval_id is None:
            raise StateError("no approval is pending")
        return self._update("approval_closed", pending_approval_id=None)

    def record_transfer(self) -> SessionState:
        """Hand off to a human. Terminal."""
        return self._update("transferred", transferred=True)
