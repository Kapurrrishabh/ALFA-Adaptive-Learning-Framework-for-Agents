"""C7: the chat loop from B7 runs over the socket.

The agent is scripted here rather than loaded. What these tests pin is the loop around it -- ask, answer,
judge, store, refit, and report what moved -- plus the two things a served surface must not get wrong: a
turn nobody has judged still gets stored, and no route hands one user another user's rows.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.agent.core import Turn
from backend.database import BUY, SELL, Store
from backend.main import create_app
from selfagent.learn import AGENT
from selfagent.learn.abstain import Abstainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chat import adapt  # noqa: E402

HAND_PICKED = 0.9980
EVIDENCE = "ticker AAPL ; close 337.00 ; rsi_14 65"
ANSWER = "its 14 day rsi is 65 , which is neither overbought nor oversold ."


class Scripted:
    """Stands in for the agent: the routes must not care what produced the turn.

    Confidence walks down over the session so a fitted cut has a spread to find, which is what makes the
    refit visible at all -- with one confidence value there is no threshold to solve for.
    """

    def __init__(self, served=ANSWER, spoke=True, written=None):
        self.served, self.spoke, self.written = served, spoke, written
        self.market = SimpleNamespace(advisor=SimpleNamespace(version="persistence@20d"))
        self.gate = SimpleNamespace(cut=0.088)
        self.abstainer, self.calibrator = Abstainer(HAND_PICKED), None
        self.asked = []

    def answer(self, question, as_of=None):
        self.asked.append(question)
        # An empty `written` is how the real agent reports a turn that stopped before the decoder: no
        # words of its own and so no confidence either.
        written = self.served if self.written is None else self.written
        confidence = 0.999 - 0.01 * len(self.asked) if written else float("nan")
        return Turn(question, question, "AAPL", "overbought", self.served, written, EVIDENCE,
                    confidence, None, self.spoke, 0.49, [],
                    "" if self.spoke else "low confidence", "2026-09-17")


def _verdicts(*labels):
    """A judge that hands back the labels given, in order, and then stops having an opinion."""
    remaining = list(labels)
    def judge(question, evidence, answer):
        if not remaining:
            return None, "no verdict yet", AGENT
        is_right = remaining.pop(0)
        return is_right, "judged in session", AGENT
    return judge


def _app(tmp_path, agent=None, judge=None):
    agent = agent or Scripted()
    store = Store(tmp_path / "served.sqlite")
    app = create_app(agent, store, judge or _verdicts(),
                     lambda feedback: adapt(feedback, 0.6, 2, AGENT),
                     Path("artifacts/advisory_combined.npz"))
    return TestClient(app), agent, store


def _signed_in(client, email="one@example.test", password="a-password"):
    client.post("/auth/register", json={"email": email, "password": password})
    token = client.post("/auth/login", json={"email": email, "password": password}).json()["token"]
    return token, {"Authorization": f"Bearer {token}"}


def _ask(socket, question):
    socket.send_json({"question": question})
    return [socket.receive_json() for _ in range(3)]


def test_the_chat_loop_runs_over_the_socket_and_the_cut_moves(tmp_path):
    """B7's gate in miniature: every turn is answered, judged, stored, and refit from the log."""
    client, agent, store = _app(tmp_path, judge=_verdicts(True, False, True, False))
    token, headers = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token, "subject": "AAPL"})
        ready = socket.receive_json()
        assert (ready["stage"], ready["answer_cut"]) == ("ready", HAND_PICKED)
        stored = [_ask(socket, f"is AAPL overbought ? {turn}")[2] for turn in range(4)]

    assert [frame["stage"] for frame in stored] == ["stored"] * 4
    assert [frame["judged"] for frame in stored] == [True, False, True, False]
    # Two rows and both labels present is the warmup this app was given, so the cut is solved from
    # turn 2 onwards and is no longer the hand-picked one.
    assert [frame["learned"] for frame in stored] == [False, True, True, True]
    assert stored[-1]["answer_cut"] != HAND_PICKED and agent.abstainer.cut == stored[-1]["answer_cut"]
    assert len(store.recent(token, ready["conversation"], limit=10)) == 4
    assert len(store.feedback.rows(labeller=AGENT)) == 4


def test_an_unjudged_turn_is_answered_and_stored_and_leaves_the_cut_alone(tmp_path):
    """A miss from the agent-as-teacher is unjudged, not guessed, and a guess here would be a label."""
    client, agent, store = _app(tmp_path, judge=_verdicts())
    token, headers = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        ready = socket.receive_json()
        working, turn, stored = _ask(socket, "is AAPL overbought ?")

    assert (working["stage"], turn["served"]) == ("working", ANSWER)
    assert (stored["judged"], stored["why"]) == (None, "no verdict yet")
    assert (stored["answer_cut"], stored["learned"]) == (HAND_PICKED, False)
    (message,) = store.recent(token, ready["conversation"])
    assert message.served == ANSWER and message.feedback_id is None
    assert store.feedback.rows(judged_only=False) == []


def test_a_turn_the_decoder_never_reached_is_stored_but_never_judged(tmp_path):
    """It has no confidence, and a 0.0 in the log would be a number the model never reported -- fitted
    into the cut as if it had."""
    agent = Scripted(served="i could not understand you", spoke=False, written="")
    client, _, store = _app(tmp_path, agent=agent, judge=_verdicts(True, True))
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        ready = socket.receive_json()
        stored = _ask(socket, "write me a poem about the sea .")[2]

    assert (stored["judged"], stored["why"]) == (None, "nothing was generated to judge")
    assert store.feedback.rows(judged_only=False) == []
    (message,) = store.recent(token, ready["conversation"])
    assert message.served == "i could not understand you" and message.confidence is None


def test_a_guarded_answer_carries_what_the_model_wrote(tmp_path):
    """The served refusal and the written answer are both in the frame, or the guard is invisible."""
    agent = Scripted(served="i do not hold that figure .", spoke=False, written="aapl rose 9.9% .")
    client, _, _ = _app(tmp_path, agent=agent)
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        socket.receive_json()
        _, turn, _ = _ask(socket, "how has AAPL been doing ?")

    assert turn["wrote"] == "aapl rose 9.9% ." and turn["spoke"] is False
    assert turn["because"] == "low confidence"


def test_a_socket_without_a_session_token_is_refused(tmp_path):
    client, _, _ = _app(tmp_path)
    with client.websocket_connect("/chat") as socket:
        socket.send_json({"question": "is AAPL overbought ?"})
        assert socket.receive_json()["stage"] == "refused"


def test_a_socket_cannot_open_another_users_conversation(tmp_path):
    client, _, _ = _app(tmp_path)
    mine, headers = _signed_in(client, "mine@example.test")
    theirs, _ = _signed_in(client, "theirs@example.test")
    conversation = client.post("/conversations", json={"subject": "AAPL"}, headers=headers).json()["id"]

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": theirs, "conversation": conversation})
        assert "does not belong" in socket.receive_json()["because"]


def test_every_read_needs_a_bearer_token(tmp_path):
    client, _, _ = _app(tmp_path)
    assert client.get("/conversations").status_code == 401
    assert client.get("/portfolio", headers={"Authorization": "made-up"}).status_code == 401


def test_holdings_come_back_from_the_trades_and_an_over_sell_is_refused(tmp_path):
    client, _, _ = _app(tmp_path)
    _, headers = _signed_in(client)
    trade = {"symbol": "AAPL", "side": BUY, "quantity": 5, "price": 300.0}
    assert client.post("/trades", json=trade, headers=headers).status_code == 201

    refused = client.post("/trades", json={**trade, "side": SELL, "quantity": 6}, headers=headers)
    assert refused.status_code == 400 and "cannot sell 6.0 of AAPL" in refused.json()["detail"]
    assert client.get("/portfolio", headers=headers).json() == [
        {"symbol": "AAPL", "quantity": 5.0, "average_price": 300.0}]


def test_the_model_route_names_what_is_answering(tmp_path):
    """A client showing an answer can say which checkpoint and which advisor produced it."""
    client, _, _ = _app(tmp_path)
    assert client.get("/model").json() == {"generator": "advisory_combined.npz",
                                           "outlook": "persistence@20d",
                                           "routing_cut": 0.088, "answer_cut": HAND_PICKED}


@pytest.mark.parametrize("email", ["one@example.test"])
def test_an_email_can_only_register_once(tmp_path, email):
    client, _, _ = _app(tmp_path)
    assert client.post("/auth/register", json={"email": email, "password": "a"}).status_code == 201
    again = client.post("/auth/register", json={"email": email, "password": "b"})
    assert again.status_code == 400 and "already has an account" in again.json()["detail"]
