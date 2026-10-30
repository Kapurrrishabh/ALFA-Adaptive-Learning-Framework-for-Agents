import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from backend.api.quota import EVERYONE_LIMITS, USER_LIMITS, QuotaExceeded, charge, day_ist, refund, resets_at
from backend.database.storage import EVERYONE, Store

from .conftest import NOW

DAY = day_ist(NOW)


def test_charging_stops_exactly_at_the_limit(store):
    for _ in range(USER_LIMITS["lm"]):
        charge(store, "a", "lm", NOW)
    with pytest.raises(QuotaExceeded) as stopped:
        charge(store, "a", "lm", NOW)
    assert (stopped.value.scope, store.usage("a", DAY)) == ("you", {"lm": USER_LIMITS["lm"]})


def test_the_overall_cap_stops_another_user_without_charging_them(store):
    for i in range(EVERYONE_LIMITS["lm"]):
        charge(store, f"user{i // USER_LIMITS['lm']}", "lm", NOW)
    with pytest.raises(QuotaExceeded) as stopped:
        charge(store, "late", "lm", NOW)
    assert (stopped.value.scope, store.usage("late", DAY)) == ("everyone", {})


def test_a_zero_limit_charges_nothing(store):
    assert (store.spend("a", "lm", DAY, 0, 5), store.usage("a", DAY)) == ("you", {})


def test_a_refund_lowers_the_users_and_the_overall_count(store):
    charge(store, "a", "lm", NOW)
    charge(store, "a", "lm", NOW)
    refund(store, "a", "lm", NOW)
    assert [store.usage(who, DAY) for who in ("a", EVERYONE)] == [{"lm": 1}, {"lm": 1}]


def test_a_refund_never_takes_a_count_below_zero(store):
    charge(store, "a", "lm", NOW)
    refund(store, "a", "lm", NOW)
    refund(store, "a", "lm", NOW)
    assert [store.usage(who, DAY) for who in ("a", EVERYONE)] == [{"lm": 0}, {"lm": 0}]


def test_a_refund_with_nothing_to_refund_leaves_the_overall_count_alone(store):
    charge(store, "a", "lm", NOW)
    charge(store, "b", "lm", NOW)
    refund(store, "a", "lm", NOW)
    refund(store, "a", "lm", NOW)
    assert store.usage(EVERYONE, DAY) == {"lm": 1}


@pytest.mark.parametrize("move", [charge, refund])
def test_the_all_users_row_is_refused_as_a_user_id(store, move):
    with pytest.raises(ValueError, match="all-users usage row"):
        move(store, EVERYONE, "lm", NOW)


def test_the_day_turns_at_midnight_india_time():
    before, after = (datetime(2026, 10, 30, 18, m, tzinfo=timezone.utc) for m in (29, 31))
    assert [day_ist(before), day_ist(after)] == ["2026-10-30", "2026-10-31"]


def test_a_utc_evening_resets_at_the_next_india_midnight_not_the_next_utc_one():
    ist = timezone(timedelta(hours=5, minutes=30))
    assert resets_at(datetime(2026, 10, 30, 20, 0, tzinfo=timezone.utc)) == datetime(2026, 11, 1, 0, 0, tzinfo=ist)


@pytest.mark.parametrize("india_time", [day_ist, resets_at])
def test_a_time_without_a_zone_is_refused(india_time):
    with pytest.raises(ValueError, match="no time zone"):
        india_time(datetime(2026, 10, 30, 20, 0))


def test_the_quota_message_gives_the_limit_and_the_reset_time(store):
    for _ in range(USER_LIMITS["lm"]):
        charge(store, "a", "lm", NOW)
    with pytest.raises(QuotaExceeded, match=f"your {USER_LIMITS['lm']} AI answers .* 00:00 India time on 27 Sep"):
        charge(store, "a", "lm", NOW)


def test_usage_counts_one_user_on_one_day(store):
    store.spend("a", "lm", DAY, 5, 5)
    store.spend("a", "lm", DAY, 5, 5)
    store.spend("b", "lm", DAY, 5, 5)
    store.spend("a", "lm", day_ist(NOW + timedelta(days=1)), 5, 5)
    assert store.usage("a", DAY) == {"lm": 2}


def test_an_sqlite_older_than_3_35_is_refused(monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.34.1")
    with pytest.raises(RuntimeError, match="SQLite 3.34.1"):
        Store(":memory:")
