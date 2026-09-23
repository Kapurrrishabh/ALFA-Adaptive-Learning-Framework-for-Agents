"""C5: a conversation survives a restart, and a signed-in user sees only their own.

Both halves of the gate are here, and so is the rule that makes the second one hold: no method takes a
user id, so a cross-user read has to be asked for with someone else's token and is refused by name.
"""

import sqlite3

import pytest

from backend.agent.core import Turn
from backend.database import BUY, SELL, NotYours, Store, fold_positions
from selfagent.learn import HUMAN

ANSWER = "aapl closed at 337.00 , up 1.4 % on the day ."


def _store(tmp_path):
    return Store(tmp_path / "users.sqlite")


def _signed_in(store, email="one@example.test", password="a-password"):
    store.register(email, password)
    return store.log_in(email, password)


def _turn(question="how has aapl been doing ?", served=ANSWER, spoke=True, confidence=0.91,
          ticker="AAPL", intent="performance"):
    return Turn(question, "recent performance on aapl ?", ticker, intent, served, served,
                "aapl closed at 337.00", confidence, confidence, spoke, 0.31, (), "", "2026-09-17")


def test_a_conversation_survives_a_restart(tmp_path):
    with _store(tmp_path) as store:
        token = _signed_in(store)
        conversation = store.start_conversation(token, "AAPL")
        store.remember(token, conversation, _turn())

    with _store(tmp_path) as reopened:
        token = reopened.log_in("one@example.test", "a-password")
        (message,) = reopened.recent(token, conversation)
        assert message.served == ANSWER
        (found,) = reopened.conversations(token)
        assert (found[0], found[2], found[3]) == (conversation, "AAPL", 1)


def test_a_signed_in_user_sees_only_their_own_conversations(tmp_path):
    with _store(tmp_path) as store:
        mine = _signed_in(store, "mine@example.test")
        theirs = _signed_in(store, "theirs@example.test")
        conversation = store.start_conversation(mine)
        store.remember(mine, conversation, _turn())

        assert store.conversations(theirs) == []
        with pytest.raises(NotYours, match="does not belong"):
            store.recent(theirs, conversation)
        with pytest.raises(NotYours, match="does not belong"):
            store.remember(theirs, conversation, _turn())


def test_a_session_that_was_logged_out_is_no_longer_a_user(tmp_path):
    with _store(tmp_path) as store:
        token = _signed_in(store)
        store.log_out(token)
        with pytest.raises(NotYours, match="not signed in"):
            store.user_of(token)


def test_a_wrong_password_and_an_unknown_email_fail_the_same_way(tmp_path):
    """Two messages would turn the login form into a way to ask which emails have accounts."""
    with _store(tmp_path) as store:
        store.register("one@example.test", "a-password")
        with pytest.raises(ValueError) as wrong:
            store.log_in("one@example.test", "another-password")
        with pytest.raises(ValueError) as unknown:
            store.log_in("nobody@example.test", "a-password")
        assert str(wrong.value).replace("one@", "") == str(unknown.value).replace("nobody@", "")


def test_an_email_can_only_have_one_account(tmp_path):
    with _store(tmp_path) as store:
        store.register("one@example.test", "a-password")
        with pytest.raises(ValueError, match="already has an account"):
            store.register("one@example.test", "a-different-password")


def test_a_position_is_the_fold_of_the_trades_and_a_sell_keeps_the_average():
    folded = fold_positions([("AAPL", BUY, 10, 300.0), ("AAPL", BUY, 10, 400.0),
                             ("AAPL", SELL, 5, 500.0), ("MSFT", BUY, 1, 100.0)])
    assert folded == [("AAPL", 15, 350.0), ("MSFT", 1, 100.0)]


def test_a_closed_position_is_not_reported():
    assert fold_positions([("AAPL", BUY, 2, 300.0), ("AAPL", SELL, 2, 310.0)]) == []


def test_selling_more_than_is_held_is_refused_and_leaves_the_log_alone(tmp_path):
    with _store(tmp_path) as store:
        token = _signed_in(store)
        store.record_trade(token, "AAPL", BUY, 5, 300.0)
        with pytest.raises(ValueError, match="cannot sell 6.0 of AAPL holding 5.0"):
            store.record_trade(token, "AAPL", SELL, 6.0, 310.0)
        assert store.holdings(token) == [("AAPL", 5.0, 300.0)]
        assert len(store.trades(token)) == 1


def test_a_users_portfolio_is_their_own(tmp_path):
    with _store(tmp_path) as store:
        mine = _signed_in(store, "mine@example.test")
        theirs = _signed_in(store, "theirs@example.test")
        store.record_trade(mine, "AAPL", BUY, 5, 300.0)
        assert store.holdings(theirs) == []


def test_a_refused_turn_is_remembered_with_no_confidence_at_all(tmp_path):
    """A refusal is history: it stores a null rather than a NaN, which SQLite would keep and compare
    false against itself for ever."""
    with _store(tmp_path) as store:
        token = _signed_in(store)
        conversation = store.start_conversation(token)
        store.remember(token, conversation, _turn(served="i don't know", spoke=False,
                                                  confidence=float("nan")))
        (message,) = store.recent(token, conversation)
        assert message.spoke == 0 and message.confidence is None


def test_a_turn_the_router_never_placed_is_remembered_too(tmp_path):
    """Refused before routing, so there is no ticker and no intent -- and those columns take no null."""
    with _store(tmp_path) as store:
        token = _signed_in(store)
        conversation = store.start_conversation(token)
        store.remember(token, conversation, _turn(served="i could not understand you", spoke=False,
                                                  ticker="", intent=""))
        (message,) = store.recent(token, conversation)
        assert (message.ticker, message.intent) == ("", "")


def test_a_message_points_at_the_feedback_row_that_judged_it(tmp_path):
    """The loop keeps owning feedback; this only holds the key, and the key has to be a real one."""
    with _store(tmp_path) as store:
        token = _signed_in(store)
        conversation = store.start_conversation(token)
        judged = store.feedback.append("2026-09-17T00:00:00+00:00", "how has aapl been doing ?",
                                       "aapl closed at 337.00", ANSWER, 0.91, True, HUMAN)
        store.remember(token, conversation, _turn(), feedback_id=judged)
        (message,) = store.recent(token, conversation)
        assert message.feedback_id == judged

        with pytest.raises(sqlite3.IntegrityError):
            store.remember(token, conversation, _turn(), feedback_id=judged + 1000)
