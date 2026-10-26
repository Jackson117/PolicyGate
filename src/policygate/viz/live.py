"""A client for the running service, for the dashboard's live pages.

The dashboard talks to the service over HTTP rather than building the graph
itself. Two reasons, and they are the same reason: there is one implementation
of the harness path, and it is the one the deployment runs. A second in-process
copy inside Streamlit would be a second thing to keep correct, and the demo
would stop proving that the service works.

It also fixes the deployment shape. The dashboard is the only container that
needs a public address; the service answers on the private network, so the
only way to spend the model budget is through whatever guards the dashboard.

Every failure here is something the page has to explain to a person -- the
service is not running, the model took too long, the key is missing -- so they
arrive as ServiceUnavailable with a sentence, not as a traceback.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from policygate.service.schemas import PendingApproval, SessionView, TurnResult

DEFAULT_URL = os.environ.get("POLICYGATE_SERVICE_URL", "http://localhost:8000")

TURN_TIMEOUT = 180.0
"""A turn is several model calls, and a refused one is several more."""


class ServiceUnavailable(RuntimeError):
    """The service could not be reached, or answered with an error."""


class SessionNotFound(ServiceUnavailable):
    """The service holds no state for that session id.

    Not an error on its own: ``POST /sessions`` only mints an id, and the
    session exists from its first message, so a session just opened is in this
    state too. The caller knows which case it is; this only reports the fact.
    """


class ServiceClient:
    """The six endpoints of policygate.service.api, typed by its own schemas."""

    def __init__(self, base_url: str = DEFAULT_URL, *, timeout: float = TURN_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
                response = client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ServiceUnavailable(
                f"No service at {self.base_url}. Start it with `make service`, "
                "or set POLICYGATE_SERVICE_URL."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ServiceUnavailable(
                f"The service did not answer within {self._timeout:.0f}s. A turn calls the "
                "model several times; check the service log."
            ) from exc
        if response.status_code == 404:
            raise SessionNotFound(_explain(response))
        if response.status_code >= 400:
            raise ServiceUnavailable(_explain(response))
        return response.json()

    # --- sessions ---------------------------------------------------------

    def open_session(self) -> str:
        return self._request("POST", "/sessions")["session_id"]

    def send(self, session_id: str, content: str) -> TurnResult:
        return TurnResult.model_validate(
            self._request("POST", f"/sessions/{session_id}/messages", json={"content": content})
        )

    def view(self, session_id: str) -> SessionView:
        return SessionView.model_validate(self._request("GET", f"/sessions/{session_id}"))

    # --- the audit chain --------------------------------------------------

    def audit(self, session_id: str) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
        """``(intact, breaks, records)`` -- the records as the store holds them."""
        body = self._request("GET", f"/sessions/{session_id}/audit")
        return body["intact"], body["breaks"], body["records"]

    # --- approvals --------------------------------------------------------

    def pending(self) -> list[PendingApproval]:
        return [PendingApproval.model_validate(row) for row in self._request("GET", "/approvals")]

    def answer(
        self, request_id: str, *, approved: bool, reviewer: str, note: str | None = None
    ) -> TurnResult:
        return TurnResult.model_validate(
            self._request(
                "POST",
                f"/approvals/{request_id}",
                json={"approved": approved, "reviewer": reviewer, "note": note},
            )
        )

    def inbox(self) -> list[PendingApproval] | None:
        """Pending approvals, or None when the service cannot be reached.

        One call answers both questions the page asks -- is it up, and what is
        waiting -- so a rerun does not hit /approvals twice to learn the same
        thing.
        """
        try:
            return self.pending()
        except ServiceUnavailable:
            return None


def _explain(response: httpx.Response) -> str:
    """A sentence a person can act on, from whatever the service returned.

    A 500 from FastAPI carries no detail -- the traceback stays in the service
    log -- so the common causes are named here instead. The first turn of a
    live session failing on the provider's key is the one that has happened.
    """
    detail = ""
    try:
        body = response.json()
        detail = body.get("detail") if isinstance(body, dict) else str(body)
    except ValueError:
        detail = response.text[:200]

    if response.status_code == 500:
        return (
            "The service failed on this turn (500). Its log has the traceback. "
            "The usual cause is the model call: a provider key that does not match the "
            "base URL it is sent to, or no key at all. A shell that exports "
            "OPENAI_API_KEY shadows the project's .env unless the service was started "
            "after that was fixed — restart it with `make service`."
        )
    return f"The service answered {response.status_code}: {detail or 'no detail'}"


def records_as_audit(records: list[dict[str, Any]]) -> list[Any]:
    """Parse the store's JSON back into AuditRecord, for the chain views."""
    from policygate.harness.types import AuditRecord

    return [AuditRecord.model_validate(r) for r in records]
