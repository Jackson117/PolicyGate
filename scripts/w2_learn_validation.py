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
    the JSON schema it was given. W1 recorded no type errors and no missing
    arguments either, so this check is insurance rather than a response to
    observed behaviour -- see the module docstring.
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
    def __init__(
        self,
        constraints: DomainConstraints,
        config: HarnessConfig,
    ) -> None:
        self._constraints = constraints
        self._config = config
    
    def review(
        self,
        call: ToolCall,
        *,
        scoped_target: str | None = None,
        world: Mapping[str, KnownObject] | None = None
    ) -> CheckResult:
        world = world or {}
        result = CheckResult.ok()

        if self._config.schema_check_enabled:
            result = result.merged_with(self._check_schema(call))
        
        if self._config.value_canonicalization_enabled:
            result = result.merged_with(self._check_values(call))
        
        referent = self._referent_of(call)
        referent_known = referent is not None and referent in world
        if self._config.referent_uniqueness_enabled and referent:
            result = result.merged_with(self._check_referent(call, referent, scoped_target, world))
        
        if self._config.precondition_check_enabled and referent_known:
            result = result.merged_with(self._check_preconditions(call, world[referent]))
        
        return result


    def _check_schema(self, call: ToolCall) -> CheckResult:
        pass

    def _check_values(self, call: ToolCall) -> CheckResult:
        pass

    def _check_referent(
        self, 
        call: ToolCall, 
        referent: str | None,
        scoped_target: str | None,
        world: Mapping[str, KnownObject], 
    ) -> CheckResult:
        pass

    def _check_preconditions(self, call: ToolCall, target: KnownObject) -> CheckResult:
        pass
    
    def _referent_of(self, call: ToolCall) -> str | None:
        field = self._constraints.referent_fields.get(call.name)
        if field is None:
            return None
        
        referent = call.arguments.get(field)
        return referent if isinstance(referent, str) else None