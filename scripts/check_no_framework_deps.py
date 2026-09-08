#!/usr/bin/env python
"""Enforce CLAUDE.md constraint #2: harness/ imports no agent framework.

The harness is the original contribution of this project. Handing any of it to
LangGraph, LiteLLM or tau2 would outsource the part worth showing, and it would
also cost the property that makes it testable: every module here runs without an
LLM, a benchmark or a network, so its unit tests finish in milliseconds.

Allowed: the standard library, pydantic, deepdiff, and policygate.harness itself.
Anything else is a failure. Run from the repo root:

    python scripts/check_no_framework_deps.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "src" / "policygate" / "harness"

ALLOWED_THIRD_PARTY = {"pydantic", "deepdiff"}

# Named rather than inferred, so the failure message can say what the rule is
# about instead of just "not on the allowlist".
BANNED_WITH_REASON = {
    "tau2": "the benchmark; harness/ must run without it installed",
    "litellm": "an LLM client; harness/ logic must be testable without a model",
    "openai": "an LLM client; harness/ logic must be testable without a model",
    "anthropic": "an LLM client; harness/ logic must be testable without a model",
    "langgraph": "an agent framework; the harness owns this logic, see CLAUDE.md #2",
    "langchain": "an agent framework; the harness owns this logic, see CLAUDE.md #2",
    "llama_index": "an agent framework; the harness owns this logic, see CLAUDE.md #2",
    "autogen": "an agent framework; the harness owns this logic, see CLAUDE.md #2",
    "crewai": "an agent framework; the harness owns this logic, see CLAUDE.md #2",
}


def top_level_imports(tree: ast.AST) -> set[tuple[str, int]]:
    """(module, lineno) for every import, including ones inside functions."""
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has no module name and cannot leave the package.
            if node.level == 0 and node.module:
                found.add((node.module.split(".")[0], node.lineno))
    return found


def main() -> int:
    if not HARNESS.is_dir():
        print(f"harness package not found at {HARNESS}", file=sys.stderr)
        return 2

    stdlib = sys.stdlib_module_names
    violations: list[str] = []
    files = sorted(HARNESS.rglob("*.py"))

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, lineno in sorted(top_level_imports(tree)):
            if module in stdlib or module in ALLOWED_THIRD_PARTY or module == "policygate":
                continue
            reason = BANNED_WITH_REASON.get(module, "not on the harness allowlist")
            rel = path.relative_to(HARNESS.parents[2])
            violations.append(f"{rel}:{lineno}: imports `{module}` -- {reason}")

    if violations:
        print("harness/ may import the stdlib, "
              f"{', '.join(sorted(ALLOWED_THIRD_PARTY))} and itself. Found:\n", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    print(f"ok: {len(files)} file(s) in harness/ import only the stdlib, "
          f"{', '.join(sorted(ALLOWED_THIRD_PARTY))} and policygate itself")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
