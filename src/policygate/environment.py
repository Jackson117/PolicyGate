"""The project's own ``.env``, applied so that how a process was launched stops mattering.

``python-dotenv`` -- which tau2 calls on import -- never overrides a variable
that is already set. That is the right default for a library and the wrong one
here: a shell that exports ``OPENAI_API_KEY`` for some other project silently
wins, and this project's key, sitting in its own ``.env`` next to the base URL
it belongs with, is ignored. The result is a 401 from a provider that was never
meant to see that key, surfacing as a 500 somewhere far away.

So an entry point calls :func:`apply_project_env`, and the file wins. The rules
that keep that honest:

* **Only when the file exists.** In a container the environment *is* the
  configuration, there is no ``.env`` (it is in .dockerignore), and this is a
  no-op.
* **It says what it did.** The names it took from the file are returned, and
  entry points log them, so a surprising key is traceable to this.
* **It can be refused.** ``POLICYGATE_IGNORE_DOTENV=1`` turns it off for anyone
  deliberately running against another provider.

Standard library only: this is imported by the service, the dashboard and the
runner, and it must not drag a dependency into any of them.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATH = Path(".env")


def parse_env_file(text: str) -> dict[str, str]:
    """``KEY=value`` lines, the subset a ``.env`` actually uses.

    Blank lines and ``#`` comments are skipped, an ``export`` prefix is allowed,
    and a value may be quoted. A line without ``=`` is not a definition and is
    ignored rather than guessed at.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def apply_project_env(path: Path | str = DEFAULT_PATH) -> list[str]:
    """Put ``path``'s variables into the environment, overriding what is there.

    Returns the names whose value this changed -- not every name in the file --
    so a caller can report the ones that would otherwise have been shadowed.
    """
    if os.environ.get("POLICYGATE_IGNORE_DOTENV"):
        return []
    file = Path(path)
    if not file.is_file():
        return []
    changed = []
    for name, value in parse_env_file(file.read_text()).items():
        if os.environ.get(name) != value:
            changed.append(name)
        os.environ[name] = value
    return changed


def describe(changed: list[str]) -> str:
    """One line for a log, naming what the file overrode."""
    if not changed:
        return "no variables taken from .env"
    return "taken from .env (overriding the environment): " + ", ".join(sorted(changed))
