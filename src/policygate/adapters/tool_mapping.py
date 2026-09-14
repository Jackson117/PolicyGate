"""Translation between tau2's world and the harness's.

The harness is forbidden to import tau2 (CLAUDE.md #2), so everything that
knows about both lives here. Three jobs:

* turn a tau2 toolkit into a :class:`~policygate.harness.types.ToolSpec`
  catalogue, taking each tool's kind from ``__tool_type__`` rather than from a
  hand-kept list that would drift;
* convert a proposed tau2 tool call into the harness's own type;
* read tool *results* into facts -- who the session is about, which object it is
  scoped to, and what objects have actually been seen.

That last job is where the domain knowledge sits. The harness deliberately has
no idea what an order is; this module does, and states it once.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.data_model.tasks import Task
from tau2.environment.toolkit import ToolKitBase, ToolType, get_tool_types

from policygate.harness.approval import ExpectedAction, GroundTruth
from policygate.harness.types import ClauseRef, ToolCall, ToolKind, ToolRole, ToolSpec
from policygate.harness.validation import (
    ArgumentSchema,
    DomainConstraints,
    KnownObject,
    Precondition,
    ValueConstraint,
)

_KIND_BY_TOOL_TYPE = {
    ToolType.READ: ToolKind.READ,
    ToolType.WRITE: ToolKind.WRITE,
    ToolType.THINK: ToolKind.THINK,
    ToolType.GENERIC: ToolKind.GENERIC,
}


def to_harness_call(call: Tau2ToolCall) -> ToolCall:
    """Convert a proposed tau2 call. Only the fields the harness judges on."""
    return ToolCall(id=call.id, name=call.name, arguments=dict(call.arguments or {}))


def tool_specs(
    toolkit: ToolKitBase,
    *,
    identity_lookups: Iterable[str] = (),
    escalation: Iterable[str] = (),
) -> tuple[ToolSpec, ...]:
    """Build the gate's catalogue from a live toolkit.

    Kinds come from tau2's own ``__tool_type__``, which is the authority; roles
    have to be declared, because neither distinction the gate needs is visible
    in the kind. In retail the identity lookups are typed READ exactly like
    ``get_user_details``, and ``transfer_to_human_agents`` is typed GENERIC
    alongside ``calculate``.
    """
    identity = set(identity_lookups)
    escalate = set(escalation)
    overlap = identity & escalate
    if overlap:
        raise ValueError(f"tools declared both identity lookup and escalation: {sorted(overlap)}")

    types = get_tool_types(toolkit)
    unknown = (identity | escalate) - set(types)
    if unknown:
        raise ValueError(f"declared roles for tools not in the toolkit: {sorted(unknown)}")

    specs = []
    for name, tool_type in sorted(types.items()):
        role = ToolRole.ORDINARY
        if name in identity:
            role = ToolRole.IDENTITY_LOOKUP
        elif name in escalate:
            role = ToolRole.ESCALATION
        specs.append(ToolSpec(name=name, kind=_KIND_BY_TOOL_TYPE[tool_type], role=role))
    return tuple(specs)


class ExtractedFacts(BaseModel):
    """What one tool result established.

    ``scope_targets`` is ``None`` when the result says nothing about what the
    session is acting on, which is different from saying it acts on nothing.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str | None = None
    scope_targets: tuple[str, ...] | None = None
    objects: tuple[KnownObject, ...] = ()


def _parse(content: str | None) -> Any:
    if content is None:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content  # a bare scalar, e.g. the user id a lookup returns


# --- retail ----------------------------------------------------------------

RETAIL_IDENTITY_LOOKUPS = ("find_user_id_by_email", "find_user_id_by_name_zip")
RETAIL_ESCALATION = ("transfer_to_human_agents",)

_POLICY = "retail/policy.md"


def retail_facts(tool_name: str, content: str | None, *, error: bool = False) -> ExtractedFacts:
    """Read a retail tool result into facts.

    A failed call establishes nothing: "Order not found" must not put the order
    into the world view, or the referent check would accept the very ids it
    exists to catch.

    Which result sets scope is a judgement, and this is it. ``get_user_details``
    lists the user's orders -- a menu, not a choice -- so it registers them as
    objects that exist without narrowing anything. ``get_order_details`` names
    one order, and that is the one under discussion.
    """
    if error:
        return ExtractedFacts()

    data = _parse(content)

    if tool_name in RETAIL_IDENTITY_LOOKUPS:
        return ExtractedFacts(user_id=data) if isinstance(data, str) and data else ExtractedFacts()

    if tool_name == "get_user_details" and isinstance(data, dict):
        orders = tuple(o for o in (data.get("orders") or []) if isinstance(o, str))
        return ExtractedFacts(
            objects=tuple(KnownObject(id=o, owner_id=data.get("user_id")) for o in orders)
        )

    if isinstance(data, dict) and isinstance(data.get("order_id"), str):
        # Every tool that returns an order -- the reads, and the writes whose
        # result is the updated order -- refreshes what is known about it.
        order = KnownObject(
            id=data["order_id"],
            attributes={"status": str(data.get("status"))} if data.get("status") else {},
            owner_id=data.get("user_id"),
        )
        return ExtractedFacts(scope_targets=(order.id,), objects=(order,))

    return ExtractedFacts()


def argument_schemas(tools: Iterable[Any]) -> dict[str, ArgumentSchema]:
    """Each tool's arguments, read from the schema tau2 hands the provider.

    Derived, never written by hand. The hand-written version declared
    ``cancel_pending_order`` as taking only ``order_id``, while its signature is
    ``(order_id, reason)`` with both required -- so the check refused every
    correct cancellation and passed every incomplete one. Reading the same JSON
    schema the model is given means the check and the tool cannot disagree.
    """
    schemas: dict[str, ArgumentSchema] = {}
    for tool in tools:
        parameters = tool.openai_schema["function"].get("parameters") or {}
        known = frozenset(parameters.get("properties") or {})
        required = frozenset(parameters.get("required") or ())
        schemas[tool.name] = ArgumentSchema(
            tool=tool.name, required=required, optional=known - required
        )
    return schemas


def retail_constraints(db: Mapping[str, Any], *, tools: Iterable[Any]) -> DomainConstraints:
    """Assemble retail's constraints from its database, its tools and its policy.

    The canonical value sets are read out of the data rather than asserted: all
    500 users carry ``country="USA"`` and one of thirteen two-letter states, and
    task 22 failed by writing "United States" and "New York" instead.

    ``tools`` is required rather than defaulted: argument schemas come from the
    tool signatures, and a call site that forgot them would silently switch the
    schema check off.
    """
    countries: set[str] = set()
    states: set[str] = set()
    for user in (db.get("users") or {}).values():
        address = user.get("address") or {}
        if isinstance(address.get("country"), str):
            countries.add(address["country"])
        if isinstance(address.get("state"), str):
            states.add(address["state"])

    # Scoped by field name rather than by tool: "state" means the same thing
    # wherever it appears, and both tools that take an address take it under
    # these names.
    value_constraints = tuple(
        ValueConstraint(
            field=field,
            allowed=frozenset(values),
            aliases=aliases,
            clause=ClauseRef(doc_id=_POLICY, clause="user address"),
        )
        for field, values, aliases in (
            ("country", countries, {"United States": "USA", "United States of America": "USA"}),
            ("state", states, _US_STATE_ALIASES),
        )
        if values
    )

    order_tools = (
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "get_order_details",
    )
    # Quoted from policy.md: an order "can only be cancelled if its status is
    # 'pending'", the same for modification, and "can only be returned" or
    # exchanged "if its status is 'delivered'".
    preconditions = tuple(
        Precondition(
            name=f"{tool} requires {sorted(statuses)[0]}",
            tool=tool,
            attribute="status",
            allowed_values=frozenset(statuses),
            clause=ClauseRef(doc_id=_POLICY, clause=tool),
        )
        for tool, statuses in (
            ("cancel_pending_order", {"pending"}),
            ("modify_pending_order_address", {"pending"}),
            ("modify_pending_order_items", {"pending"}),
            ("modify_pending_order_payment", {"pending"}),
            ("return_delivered_order_items", {"delivered"}),
            ("exchange_delivered_order_items", {"delivered"}),
        )
    )

    return DomainConstraints(
        value_constraints=value_constraints,
        referent_fields={t: "order_id" for t in order_tools if t != "get_order_details"},
        preconditions=preconditions,
        schemas=argument_schemas(tools),
    )


def ground_truth(task: Task) -> GroundTruth:
    """Lift a task's reference trajectory into the form the oracle reads.

    Only the assistant's own actions: a ``user`` requestor marks a call made in
    the user's environment, which the agent never proposes and the approval gate
    therefore never sees.

    Note what this is not. ``evaluation_criteria.actions`` is one path to the
    target end state, and retail is scored on the database, so an agent taking a
    different route to the same state passes while this list does not sanction
    it. The oracle built on it is strict, not omniscient -- see
    :class:`~policygate.harness.approval.GroundTruth`.
    """
    criteria = task.evaluation_criteria
    actions = () if criteria is None else tuple(criteria.actions or ())
    return GroundTruth(
        task_id=task.id,
        actions=tuple(
            ExpectedAction(
                action_id=a.action_id,
                name=a.name,
                arguments=dict(a.arguments or {}),
                compare_args=None if a.compare_args is None else tuple(a.compare_args),
            )
            for a in actions
            if a.requestor == "assistant"
        ),
    )


_US_STATE_ALIASES = {
    "Arizona": "AZ",
    "California": "CA",
    "Colorado": "CO",
    "Florida": "FL",
    "Indiana": "IN",
    "New York": "NY",
    "Pennsylvania": "PA",
    "Texas": "TX",
}
