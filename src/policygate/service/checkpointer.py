"""Where interrupted sessions wait: Postgres in deployment, memory in tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


@contextmanager
def checkpointer(url: str | None = None) -> Iterator[Any]:
    """A Postgres checkpointer for ``url``, or an in-memory one when there is none.

    In-memory is only for tests and local poking: an approval that waits on it
    dies with the process, which is exactly the property the service exists to
    avoid.
    """
    url = url or os.environ.get("POLICYGATE_CHECKPOINT_URL")
    if not url:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(url) as saver:
        saver.setup()
        yield saver
