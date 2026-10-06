"""The polite session, which now lives under `backend/live/`.

Moved there rather than copied: this folder is deleted before submission and the agent's live-search path
needs the same rate limiting and the same robots.txt handling, so the surviving copy owns it. Every
source here imports from this name, which is why the module stays.
"""

from backend.live.session import (  # noqa: F401
    BACKOFF_SECONDS,
    CONTACT_VARIABLE,
    DEFAULT_REQUESTS_PER_SECOND,
    FATAL_STATUS_CODES,
    MAX_ATTEMPTS,
    TIMEOUT_SECONDS,
    BlockedByHost,
    PoliteSession,
    user_agent,
)
