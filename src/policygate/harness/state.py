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

Four properties are load-bearing and are enforced below rather than left as
comments:

* Facts come from tool *results*, never from what anyone said -- in domains
  that require it. Retail is explicit that identity must be established by a
  lookup "even when the user already provides the user id"; airline says the
  opposite, that the agent "must first obtain the user id from the user". That
  difference is what :class:`DomainStateProfile` exists to carry.
* State is derived, never assigned. :func:`derive_state` is a pure function of
  facts and a profile; there is no setter.
* An approval is bound to one specific action and is spent once. It authorises
  a fingerprint, not a session, so approving one write cannot licence another.
* Anything that moves what is in scope voids an outstanding approval. Reviewing
  a write against one order and then executing it against another is the
  machine version of the task 14 failure, and an approval gate that allows it
  is not a gate.

Pydantic and the standard library only. See docs/constraints.md #2.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from policygate.harness.types import ToolCall


class SessionState(StrEnum):
    """Where a conversation stands, as far as the harness can prove.

    Deliberately coarse. Every distinction here has to be decidable from tool
    results alone; anything needing the text of the conversation belongs in a
    validator, not in the state.
    """

    UNVERIFIED = "unverified"
    """No identity resolved. Only identity lookups and transfer are legitimate.

    Unreachable in domains whose profile does not require an identity lookup.
    """

    VERIFIED = "verified"
    """Identity settled as far as the domain requires. Reads are open; no unique
    write target yet."""

    SCOPED = "scoped"
    """Exactly one write target is in scope. A write may now be proposed."""

    PENDING_APPROVAL = "pending_approval"
    """A proposed write is paused at the gate. Nothing proceeds until resolved."""

    TRANSFERRED = "transferred"
    """Handed to a human. Terminal: the conversation is over for the agent."""


class StateError(Exception):
    """An illegal fact update.

    The one place the harness raises rather than returning a
    :class:`~policygate.harness.types.CheckResult`. A model proposing something
    invalid is normal operation and gets a structured refusal; a caller trying
    to execute an unapproved write is a bug in the adapter, and hiding it behind
    a result object would let it pass silently.
    """


class DomainStateProfile(BaseModel):
    """Domain rules the state machine cannot infer for itself.

    The skeleton -- confirm before writing, transfer is terminal -- is shared:
    retail, airline and telecom carry those rules in near-identical wording.
    What differs is the front of the conversation, so only that is configurable.
    """

    model_config = ConfigDict(frozen=True)

    name: str

    requires_identity_lookup: bool = True
    """Whether an identity must be established by a tool before anything else.

    Retail: yes, and explicitly "even when the user already provides the user
    id". Airline: no -- its policy has no authentication step at all and says
    the agent obtains the user id from the user. With this off, UNVERIFIED is
    unreachable and a session starts VERIFIED.

    Which tool's result establishes it is the adapter's call, made when it
    extracts facts: retail's lookup result does; banking_knowledge's lookup only
    finds a candidate, and identity is settled by the result of its verification
    log (a tool with role ``IDENTITY_COMMIT``). The name predates that case.
    """

    single_subject_per_session: bool = True
    """Whether the session is locked to the first subject it resolves.

    Retail: yes -- "you can only help one user per conversation ... must deny
    any requests for tasks related to any other user". Airline states no such
    rule.
    """


RETAIL_PROFILE = DomainStateProfile(
    name="retail", requires_identity_lookup=True, single_subject_per_session=True
)
AIRLINE_PROFILE = DomainStateProfile(
    name="airline", requires_identity_lookup=False, single_subject_per_session=False
)


class SessionFacts(BaseModel):
    """Everything the harness has established, and the only input to state.

    Immutable: updates produce a new instance, so a fact set recorded in an
    audit entry still means the same thing when someone checks it later.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str | None = None
    """Resolved by an identity lookup, in domains that require one."""

    candidate_targets: frozenset[str] = frozenset()
    """Referents that could be the object of a write -- order ids, typically.

    Zero means nothing is in scope yet; more than one means the referent is
    ambiguous, which is what task 14 looked like when the agent returned items
    from a different order than the one under discussion. Both keep the session
    out of SCOPED, so neither can reach a write.
    """

    pending_approval_id: str | None = None
    pending_action_fingerprint: str | None = None
    """The action under review. Set and cleared together with the id above."""

    authorized_action: str | None = None
    """A fingerprint cleared for exactly one execution, then spent.

    Without this an approval would authorise a *state* rather than an action,
    and every later write against the same target would ride on it.
    """

    transferred: bool = False

    @model_validator(mode="after")
    def _approval_fields_move_together(self) -> SessionFacts:
        if (self.pending_approval_id is None) != (self.pending_action_fingerprint is None):
            raise StateError(
                "pending_approval_id and pending_action_fingerprint must be set or cleared together"
            )
        return self


def derive_state(facts: SessionFacts, profile: DomainStateProfile) -> SessionState:
    """The session's state. A pure function of facts and the domain profile.

    Order matters: transfer is terminal and outranks everything, and a pending
    approval freezes the session wherever it stood.
    """
    if facts.transferred:
        return SessionState.TRANSFERRED
    if facts.pending_approval_id is not None:
        return SessionState.PENDING_APPROVAL
    if profile.requires_identity_lookup and facts.user_id is None:
        return SessionState.UNVERIFIED
    if len(facts.candidate_targets) == 1:
        return SessionState.SCOPED
    return SessionState.VERIFIED


ALLOWED_TRANSITIONS: Mapping[SessionState, frozenset[SessionState]] = {
    SessionState.UNVERIFIED: frozenset(
        {SessionState.VERIFIED, SessionState.PENDING_APPROVAL, SessionState.TRANSFERRED}
    ),
    SessionState.VERIFIED: frozenset(
        {SessionState.SCOPED, SessionState.PENDING_APPROVAL, SessionState.TRANSFERRED}
    ),
    SessionState.SCOPED: frozenset(
        {SessionState.VERIFIED, SessionState.PENDING_APPROVAL, SessionState.TRANSFERRED}
    ),
    SessionState.PENDING_APPROVAL: frozenset(
        {
            SessionState.UNVERIFIED,
            SessionState.VERIFIED,
            SessionState.SCOPED,
            SessionState.TRANSFERRED,
        }
    ),
    SessionState.TRANSFERRED: frozenset(),
}
"""Which state may follow which. Staying put is always legal and is not listed.

Notably absent: UNVERIFIED -> SCOPED (nothing can be scoped before an identity
exists) and anything out of TRANSFERRED (the agent's turn has ended).

SCOPED -> VERIFIED is present because scope legitimately widens again when a
second candidate order comes into play; dropping out of write range is the
intended behaviour, not an error.

An approval may open from any live state, and returns to wherever it opened.
"Writes only once exactly one target is in scope" is a rule, and rules belong
to gating, which enforces it before a write can reach the gate. Encoding it here
as well made the approval gate crash whenever gating was switched off -- an
exception tau2 would retry three times and then file as an infrastructure error,
silently dropping the task from that arm.
"""


def is_legal_transition(before: SessionState, after: SessionState) -> bool:
    """Whether ``before`` may be followed by ``after``. Staying put is always legal."""
    return after is before or after in ALLOWED_TRANSITIONS[before]


class StateTransition(BaseModel):
    """One recorded change, with what caused it. Feeds the audit chain.

    Recorded whenever the *facts* change, not only when the state label does.
    Swapping the write target from one order to another leaves the label on
    SCOPED, and an audit trail that skipped it would lose the single most
    important thing that happened in that turn.
    """

    model_config = ConfigDict(frozen=True)

    before: SessionState
    after: SessionState
    trigger: str
    facts: SessionFacts

    @property
    def changed_state(self) -> bool:
        return self.before is not self.after


class SessionTracker:
    """Accumulates facts and reports the state they imply.

    The adapter owns the mapping from a domain's tool results to these calls;
    this class knows nothing about retail, orders or tau2, which is what lets it
    be unit-tested without an LLM and reused across domains.
    """

    def __init__(
        self,
        facts: SessionFacts | None = None,
        profile: DomainStateProfile = RETAIL_PROFILE,
    ) -> None:
        self._profile = profile
        self._facts = facts or SessionFacts()
        self._check_scope_invariant(self._facts)
        self._history: list[StateTransition] = []

    @property
    def profile(self) -> DomainStateProfile:
        return self._profile

    @property
    def facts(self) -> SessionFacts:
        return self._facts

    @property
    def state(self) -> SessionState:
        return derive_state(self._facts, self._profile)

    @property
    def history(self) -> tuple[StateTransition, ...]:
        """Fact changes so far, oldest first. Updates that changed nothing are not recorded."""
        return tuple(self._history)

    def _check_scope_invariant(self, facts: SessionFacts) -> None:
        """Nothing may be scoped before the subject is known.

        Lives here rather than on the model because whether an identity is
        required at all is a domain question, and the model has no profile.
        """
        if (
            self._profile.requires_identity_lookup
            and facts.candidate_targets
            and facts.user_id is None
        ):
            raise StateError("cannot scope a target before an identity is resolved")

    def _update(self, trigger: str, **changes: object) -> SessionState:
        before = self.state
        if before is SessionState.TRANSFERRED:
            # Terminal means terminal. Without this the update would be waved
            # through as a no-op, because `transferred` dominates derive_state
            # and the state would not appear to change -- so facts would keep
            # accumulating after handoff and the audit trail would show activity
            # in a session that had already ended.
            raise StateError(f"session was transferred to a human; refusing {trigger!r}")

        candidate = self._facts.model_copy(update=changes)
        # Re-run validators: model_copy skips them, and the approval invariant is
        # exactly the kind of thing that must not be bypassable.
        candidate = SessionFacts.model_validate(candidate.model_dump())
        self._check_scope_invariant(candidate)

        after = derive_state(candidate, self._profile)
        if not is_legal_transition(before, after):
            raise StateError(f"illegal transition {before} -> {after} (trigger: {trigger})")

        if candidate == self._facts:
            return after
        self._facts = candidate
        self._history.append(
            StateTransition(before=before, after=after, trigger=trigger, facts=candidate)
        )
        return after

    def record_identity(self, user_id: str) -> SessionState:
        """Bind the session to a subject, from an identity lookup's result.

        Under a single-subject profile, rebinding to a different subject is
        refused: retail allows one user per conversation and requires denying
        anything concerning another.
        """
        if (
            self._profile.single_subject_per_session
            and self._facts.user_id is not None
            and self._facts.user_id != user_id
        ):
            raise StateError(
                f"session is bound to {self._facts.user_id!r}; "
                f"one user per conversation, refusing to rebind to {user_id!r}"
            )
        return self._update("identity_resolved", user_id=user_id)

    def record_targets(self, targets: Iterable[str]) -> SessionState:
        """Replace the set of candidate write targets.

        Replacement rather than accumulation: the candidate set describes what
        is in scope *now*, and a narrowing question that rules orders out has to
        be able to shrink it, or nothing could ever reach SCOPED.

        Moving scope voids any approval that was granted or is being reviewed
        against the old scope. The alternative -- refusing the update -- would
        make a user changing their mind an error, which it is not; but letting
        the approval survive would mean reviewing a write against one order and
        performing it against another.
        """
        new_targets = frozenset(targets)
        if new_targets == self._facts.candidate_targets:
            return self.state

        changes: dict[str, object] = {"candidate_targets": new_targets}
        trigger = "targets_updated"
        if self._facts.pending_approval_id is not None:
            changes |= {"pending_approval_id": None, "pending_action_fingerprint": None}
            trigger = "targets_updated_voiding_pending_approval"
        elif self._facts.authorized_action is not None:
            changes["authorized_action"] = None
            trigger = "targets_updated_voiding_authorization"
        return self._update(trigger, **changes)

    def open_approval(self, request_id: str, tool_call: ToolCall) -> SessionState:
        """Pause a specific proposed write at the gate.

        Takes the call itself, not just an id: the approval has to be about an
        action, otherwise resolving it would licence whatever the session
        happened to be pointing at by then.
        """
        if self._facts.pending_approval_id is not None:
            raise StateError(
                f"approval {self._facts.pending_approval_id!r} is already pending; "
                "the gate holds one write at a time"
            )
        return self._update(
            "approval_opened",
            pending_approval_id=request_id,
            pending_action_fingerprint=tool_call.fingerprint(),
        )

    def resolve_approval(self, approved: bool) -> SessionState:
        """Settle the pending approval.

        On approval the reviewed fingerprint becomes authorised for exactly one
        execution. On rejection nothing is authorised and the session simply
        resumes.
        """
        if self._facts.pending_approval_id is None:
            raise StateError("no approval is pending")
        return self._update(
            "approval_granted" if approved else "approval_rejected",
            pending_approval_id=None,
            pending_action_fingerprint=None,
            authorized_action=self._facts.pending_action_fingerprint if approved else None,
        )

    def record_execution(self, tool_call: ToolCall) -> SessionState:
        """Note that an authorised write has been carried out, spending the authorisation.

        The fingerprint must match what was approved. A mismatch is the
        approve-A-execute-B failure and is refused outright rather than
        reported, because by the time this is called the adapter is about to
        hand the call to the environment.
        """
        if self._facts.authorized_action is None:
            raise StateError(
                f"{tool_call.name!r} has no approval; writes must be approved before execution"
            )
        if self._facts.authorized_action != tool_call.fingerprint():
            raise StateError(
                f"{tool_call.name!r} does not match the approved action; "
                "approving one write does not authorise another"
            )
        return self._update("execution_recorded", authorized_action=None)

    def record_transfer(self) -> SessionState:
        """Hand off to a human. Terminal."""
        return self._update("transferred", transferred=True)
