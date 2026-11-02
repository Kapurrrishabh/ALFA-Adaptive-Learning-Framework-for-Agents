"""Client for the self-learning agent from the sibling project (scripts/serve.py there).

That agent is a from-scratch NumPy Transformer with its own guardrails: it answers
only when its evidence carries the figures and its confidence clears a fitted cut,
and otherwise says which stage stopped it. StockIntel shows it as a second opinion.
It runs as its own process (it needs ~5 GB of RAM), reached over HTTP + WebSocket.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import BUSY_WAIT_S

DEFAULT_URL = "http://127.0.0.1:8010"
CREDENTIALS = Path.home() / ".stockintel" / "selfagent.json"
TIMEOUT_S = 60
# Provisional, not measured: far more chat tabs than one deployment has open. A session dropped past it
# only starts a new agent conversation, so its next "why?" loses the instrument.
CONVERSATIONS_KEPT = 1000


class AgentUnavailable(Exception):
    pass


@contextmanager
def _unless_busy(lock: threading.Lock):
    """The agent answers one question at a time, and a queued question would hold a server thread."""
    if not lock.acquire(timeout=BUSY_WAIT_S):
        raise AgentUnavailable("the agent is answering someone else; try again in a moment")
    try:
        yield
    finally:
        lock.release()


class SelfAgentClient:
    def __init__(self, url: Optional[str] = None):
        self.url = (url or os.environ.get("SELFAGENT_URL", DEFAULT_URL)).rstrip("/")
        self._token: Optional[str] = None
        # one agent conversation per caller session, so a follow-up like "why?" keeps the last instrument
        self._conversations: Dict[str, int] = {}
        self._lock = threading.Lock()

    def status(self) -> Dict[str, Any]:
        import httpx
        try:
            r = httpx.get(self.url + "/model", timeout=2)
            r.raise_for_status()
            return {"available": True, **r.json()}
        except httpx.HTTPStatusError as exc:        # its message names the agent's internal address
            return {"available": False, "reason": f"HTTP {exc.response.status_code}"}
        except (httpx.HTTPError, ValueError) as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _credentials(self) -> Dict[str, str]:
        if CREDENTIALS.exists():
            return json.loads(CREDENTIALS.read_text())
        # a fresh email with each password, so losing this file means a new account, not a locked-out one
        creds = {"email": f"stockintel-{secrets.token_hex(4)}@local", "password": secrets.token_urlsafe(18)}
        CREDENTIALS.parent.mkdir(parents=True, exist_ok=True)
        CREDENTIALS.write_text(json.dumps(creds))
        CREDENTIALS.chmod(0o600)
        return creds

    def _login(self) -> str:
        import httpx
        creds = self._credentials()
        with httpx.Client(base_url=self.url, timeout=10) as c:
            r = c.post("/auth/login", json=creds)
            if r.status_code != 200:
                c.post("/auth/register", json=creds)
                r = c.post("/auth/login", json=creds)
            if r.status_code != 200:
                raise AgentUnavailable(f"could not log in to the self-learning agent: HTTP {r.status_code} {r.text[:120]}")
            return r.json()["token"]

    def ask(self, question: str, subject: str = "", *, session: str) -> Dict[str, Any]:
        """One question over one socket; returns the agent's 'turn' frame.

        An empty `session` starts a new conversation each time."""
        import httpx
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect
        with _unless_busy(self._lock):
            try:
                self._token = self._token or self._login()
                ws_url = self.url.replace("http", "ws", 1) + "/chat"
                with connect(ws_url, open_timeout=10, close_timeout=2, max_size=2 ** 24) as ws:
                    ws.send(json.dumps({"token": self._token, "subject": subject,
                                        "conversation": self._conversations.get(session)}))
                    first = json.loads(ws.recv(timeout=TIMEOUT_S))
                    if first.get("stage") == "refused":
                        self._token = None
                        self._conversations.clear()
                        raise AgentUnavailable(f"agent refused the session: {first.get('because')}")
                    if session:
                        self._conversations.pop(session, None)
                        self._conversations[session] = first["conversation"]
                        if len(self._conversations) > CONVERSATIONS_KEPT:
                            del self._conversations[next(iter(self._conversations))]
                    ws.send(json.dumps({"question": question}))
                    while True:
                        frame = json.loads(ws.recv(timeout=TIMEOUT_S))
                        if frame.get("stage") == "turn":
                            return frame
                        if frame.get("stage") == "refused":
                            raise AgentUnavailable(frame.get("because", "refused"))
            except (OSError, TimeoutError, WebSocketException, httpx.HTTPError) as exc:
                raise AgentUnavailable(f"self-learning agent not reachable: {exc}") from exc
