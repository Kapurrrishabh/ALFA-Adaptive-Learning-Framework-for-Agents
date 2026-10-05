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
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_URL = "http://127.0.0.1:8010"
CREDENTIALS = Path.home() / ".stockintel" / "selfagent.json"
TIMEOUT_S = 60


class AgentUnavailable(Exception):
    pass


class SelfAgentClient:
    def __init__(self, url: Optional[str] = None):
        self.url = (url or os.environ.get("SELFAGENT_URL", DEFAULT_URL)).rstrip("/")
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    def status(self) -> Dict[str, Any]:
        import httpx
        try:
            r = httpx.get(self.url + "/model", timeout=2)
            r.raise_for_status()
            return {"available": True, "url": self.url, **r.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"available": False, "url": self.url, "reason": f"{type(exc).__name__}: {exc}"}

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

    def ask(self, question: str, subject: str = "") -> Dict[str, Any]:
        """One question over one socket; returns the agent's 'turn' frame."""
        import httpx
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect
        with self._lock:
            try:
                self._token = self._token or self._login()
                ws_url = self.url.replace("http", "ws", 1) + "/chat"
                with connect(ws_url, open_timeout=10, close_timeout=2, max_size=2 ** 24) as ws:
                    ws.send(json.dumps({"token": self._token, "subject": subject}))
                    first = json.loads(ws.recv(timeout=TIMEOUT_S))
                    if first.get("stage") == "refused":
                        self._token = None
                        raise AgentUnavailable(f"agent refused the session: {first.get('because')}")
                    ws.send(json.dumps({"question": question}))
                    while True:
                        frame = json.loads(ws.recv(timeout=TIMEOUT_S))
                        if frame.get("stage") == "turn":
                            return frame
                        if frame.get("stage") == "refused":
                            raise AgentUnavailable(frame.get("because", "refused"))
            except (OSError, TimeoutError, WebSocketException, httpx.HTTPError) as exc:
                raise AgentUnavailable(f"self-learning agent not reachable at {self.url}: {exc}") from exc
