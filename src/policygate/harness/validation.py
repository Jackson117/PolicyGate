"""Argument validation: refuse a write that cannot be right, before it runs.

The module with the strongest W1 evidence. Three mechanisms, each with its own
switch, because each answers a failure the retail traces actually produced and
folding them together would make an improvement unattributable.

* **Canonical values.** Task 22 wrote ``state="New York"``, ``country="United
  States"``; the passing trial wrote ``"NY"``, ``"USA"``. Same address, and the
  world-state hash compares the surface form. The canonical sets are read out of
  the domain database -- all 500 retail users have ``country="USA"`` and one of
  13 two-letter states -- so this constraint is data, not an opinion.
* **Referent identity.** Task 14 returned items from a different order than the
  one under discussion. Separately, 17 of the 31 "Order not found" errors used
  an id that had never appeared in any earlier tool result -- ``#12345``,
  ``#67890`` and similar inventions.
* **Preconditions.** retail policy.md states them exactly: an order "can only be
  cancelled if its status is 'pending'" (and modified likewise), "can only be
  returned if its status is 'delivered'", the same for exchange.

Two decisions worth stating, because both could reasonably have gone the other
way:

*Refuse, do not repair.* A canonicalising harness that rewrote ``"New York"``
to ``"NY"`` would fix task 22 outright, and it is tempting. It is not done here.
Silently altering the arguments of a write means the recorded action is no
longer the one the model chose, which destroys the attribution the whole project
rests on -- an improvement could no longer be told apart from the harness doing
the work. The refusal carries the canonical value in ``expected``, so correcting
it costs the model one turn and stays its own decision.

*Facts come from tool results.* The world view is built from what reads
returned, never from what anyone said. An order id the user mentioned but that
was never looked up is not a known referent, which is the same rule the state
machine applies to identity and the reason invented ids are caught at all.

Pydantic and the standard library only. See CLAUDE.md constraint #2.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from policygate.harness.config import HarnessConfig
from policygate.harness.types import (
    CheckFailure,
    CheckResult,
    ClauseRef,
    FailureCode,
    ToolCall,
)


class ValueConstraint(BaseModel):
    """The set of surface forms a field may take.

    ``aliases`` maps known-wrong spellings to the canonical one so a refusal can
    say what to write instead. It is a courtesy for the message, never an
    instruction to rewrite the call.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    allowed: frozenset[str]
    aliases: Mapping[str, str] = {}
    clause: ClauseRef | None = None

    def canonical_for(self, value: str) -> str | None:
        """The canonical form of ``value``, if one can be named. Case-insensitive."""
        if value in self.aliases:
            return self.aliases[value]
        lowered = value.casefold()
        for candidate in self.allowed:
            if candidate.casefold() == lowered:
                return candidate
        return None


class ArgumentSchema(BaseModel):
    """Which arguments a tool takes. Presence and spelling only.

    Deliberately not a full type checker: the provider already validates against
    the JSON schema it was given, and what W1 actually produced were missing and
    misspelled fields, not type errors.
    """

    model_config = ConfigDict(frozen=True)

    tool: str
    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()

    @property
    def known(self) -> frozenset[str]:
        return self.required | self.optional


class Precondition(BaseModel):
    """A condition the target object must already satisfy.

    Stated as an attribute of an observed object rather than in domain terms, so
    the harness stays free of retail vocabulary: retail supplies
    ``attribute="status"``, ``allowed_values={"delivered"}``.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    tool: str
    attribute: str
    allowed_values: frozenset[str]
    clause: ClauseRef | None = None


class KnownObject(BaseModel):
    """Something a tool result actually returned.

    Only objects in here can be the referent of a write. That is what catches an
    id the model invented: it was never in a result, so it is not known.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    attributes: Mapping[str, str] = {}
    owner_id: str | None = None


class DomainConstraints(BaseModel):
    """Everything the validator needs to know about one domain.

    Assembled by the adapter from the domain database and policy text. The
    harness never hardcodes any of it.
    """

    model_config = ConfigDict(frozen=True)

    schemas: Mapping[str, ArgumentSchema] = {}
    value_constraints: tuple[ValueConstraint, ...] = ()
    referent_fields: Mapping[str, str] = {}
    """tool name -> the argument naming the object it acts on."""

    preconditions: tuple[Precondition, ...] = ()

    def preconditions_for(self, tool: str) -> tuple[Precondition, ...]:
        return tuple(p for p in self.preconditions if p.tool == tool)


class Validator:
    """Runs the enabled checks over a proposed call and reports every problem.

    All failures are returned together rather than one per turn: fixing errors
    one at a time burns turns and, worse, makes the harness look like the cause
    of the extra steps.
    """

    def __init__(self, constraints: DomainConstraints, config: HarnessConfig) -> None:
        self._c = constraints
        self._config = config

    def review(
        self,
        call: ToolCall,
        *,
        scoped_target: str | None = None,
        world: Mapping[str, KnownObject] | None = None,
    ) -> CheckResult:
        """Check ``call`` against the constraints that are switched on.

        ``scoped_target`` is the single object the session has narrowed to, and
        ``world`` is everything reads have returned so far.
        """
        world = world or {}
        result = CheckResult.ok()

        if self._config.schema_check_enabled:
            result = result.merged_with(self._check_schema(call))
        if self._config.value_canonicalization_enabled:
            result = result.merged_with(self._check_values(call))

        referent = self._referent_of(call)
        referent_known = referent is not None and referent in world

        if self._config.referent_uniqueness_enabled:
            result = result.merged_with(self._check_referent(call, referent, scoped_target, world))

        # Preconditions describe the target's state, so they are only meaningful
        # once the target is known. Reporting "status must be delivered" about an
        # order that does not exist would be noise on top of the real problem.
        if self._config.precondition_check_enabled and referent_known:
            result = result.merged_with(self._check_preconditions(call, world[referent]))

        return result

    # --- individual checks -------------------------------------------------

    def _check_schema(self, call: ToolCall) -> CheckResult:
        schema = self._c.schemas.get(call.name)
        if schema is None:
            return CheckResult.ok()

        failures: list[CheckFailure] = []
        supplied = set(call.arguments)
        for missing in sorted(schema.required - supplied):
            failures.append(
                CheckFailure(
                    code=FailureCode.SCHEMA_INVALID,
                    field=missing,
                    message=f"{call.name!r} requires {missing!r}",
                    expected=missing,
                )
            )
        for unknown in sorted(supplied - schema.known):
            failures.append(
                CheckFailure(
                    code=FailureCode.SCHEMA_INVALID,
                    field=unknown,
                    message=f"{call.name!r} has no argument {unknown!r}",
                    observed=unknown,
                    expected=sorted(schema.known),
                )
            )
        return CheckResult.failed(*failures) if failures else CheckResult.ok()

    def _check_values(self, call: ToolCall) -> CheckResult:
        failures: list[CheckFailure] = []
        for constraint in self._c.value_constraints:
            if constraint.field not in call.arguments:
                continue
            value = call.arguments[constraint.field]
            if not isinstance(value, str) or value in constraint.allowed:
                continue
            canonical = constraint.canonical_for(value)
            hint = (
                f"write {canonical!r}"
                if canonical
                else f"expected one of {sorted(constraint.allowed)}"
            )
            failures.append(
                CheckFailure(
                    code=FailureCode.VALUE_NOT_CANONICAL,
                    field=constraint.field,
                    message=f"{constraint.field}={value!r} is not the stored form; {hint}",
                    observed=value,
                    expected=canonical or sorted(constraint.allowed),
                )
            )
        return CheckResult.failed(*failures) if failures else CheckResult.ok()

    def _check_referent(
        self,
        call: ToolCall,
        referent: str | None,
        scoped_target: str | None,
        world: Mapping[str, KnownObject],
    ) -> CheckResult:
        field = self._c.referent_fields.get(call.name)
        if field is None:
            return CheckResult.ok()

        if referent is None:
            return CheckResult.failed(
                CheckFailure(
                    code=FailureCode.SCHEMA_INVALID,
                    field=field,
                    message=f"{call.name!r} needs {field!r} to say which object it acts on",
                    expected=field,
                )
            )

        if referent not in world:
            return CheckResult.failed(
                CheckFailure(
                    code=FailureCode.REFERENT_UNKNOWN,
                    field=field,
                    message=(
                        f"{referent!r} has not been returned by any lookup in this session; "
                        "retrieve it before acting on it"
                    ),
                    observed=referent,
                    expected=sorted(world),
                )
            )

        if scoped_target is not None and referent != scoped_target:
            return CheckResult.failed(
                CheckFailure(
                    code=FailureCode.REFERENT_AMBIGUOUS,
                    field=field,
                    message=(
                        f"the session is scoped to {scoped_target!r} but this acts on "
                        f"{referent!r}; confirm which object is meant before writing"
                    ),
                    observed=referent,
                    expected=scoped_target,
                )
            )
        return CheckResult.ok()

    def _check_preconditions(self, call: ToolCall, target: KnownObject) -> CheckResult:
        failures: list[CheckFailure] = []
        for pre in self._c.preconditions_for(call.name):
            observed = target.attributes.get(pre.attribute)
            if observed in pre.allowed_values:
                continue
            failures.append(
                CheckFailure(
                    code=FailureCode.PRECONDITION_UNMET,
                    field=pre.attribute,
                    message=(
                        f"{pre.name}: {target.id!r} has {pre.attribute}={observed!r}, "
                        f"but {call.name!r} requires {sorted(pre.allowed_values)}"
                    ),
                    observed=observed,
                    expected=sorted(pre.allowed_values),
                )
            )
        return CheckResult.failed(*failures) if failures else CheckResult.ok()

    def _referent_of(self, call: ToolCall) -> str | None:
        field = self._c.referent_fields.get(call.name)
        if field is None:
            return None
        value = call.arguments.get(field)
        return value if isinstance(value, str) else None


def world_from(objects: Iterable[KnownObject]) -> dict[str, KnownObject]:
    """Index observed objects by id, for passing to :meth:`Validator.review`."""
    return {o.id: o for o in objects}
