"""Daily caps on language-model calls, per user and for all users together, so the free model
time is shared out. One 'lm' kind covers every Space or Claude call, so the user sees one number."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from backend.config import MARKETS
from backend.database.storage import Store

# PROVISIONAL: about 50 answers a day fit the Space's 5 free ZeroGPU minutes at ~6 s each, and 40 leaves
# headroom. Confirm on the Space's usage page after a day of traffic.
LIMITS = {"lm": 10}             # per user per day
EVERYONE_LIMITS = {"lm": 40}    # all users together per day
IST = ZoneInfo(MARKETS["NSE"].timezone)


def day_ist(now: datetime) -> str:
    return now.astimezone(IST).date().isoformat()


def resets_at(now: datetime) -> datetime:
    return datetime.combine(now.astimezone(IST).date() + timedelta(days=1), time(0), tzinfo=IST)


class QuotaExceeded(Exception):
    def __init__(self, kind: str, limit: int, scope: str, resets_at: datetime):
        self.kind, self.limit, self.scope, self.resets_at = kind, limit, scope, resets_at
        used = f"You have used your {limit}" if scope == "you" else f"All users together have used the app's {limit}"
        super().__init__(f"{used} AI answers for today. More come at {resets_at:%H:%M} India time "
                         f"on {resets_at:%d %b}.")


def charge(store: Store, who: str, kind: str, now: datetime) -> None:
    stopped = store.spend(who, kind, day_ist(now), LIMITS[kind], EVERYONE_LIMITS[kind])
    if stopped:
        limit = LIMITS[kind] if stopped == "you" else EVERYONE_LIMITS[kind]
        raise QuotaExceeded(kind, limit, stopped, resets_at(now))


def refund(store: Store, who: str, kind: str, now: datetime) -> None:
    """Pass the charge's own `now`, so a call that runs past midnight is refunded on the day it was charged."""
    store.refund(who, kind, day_ist(now))
