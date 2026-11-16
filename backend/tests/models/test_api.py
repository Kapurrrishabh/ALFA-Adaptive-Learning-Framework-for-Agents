"""C7: the chat loop from B7 runs over the socket.

The agent is scripted here rather than loaded. What these tests pin is the loop around it -- ask, answer,
judge, store, refit, and report what moved -- plus the two things a served surface must not get wrong: a
turn nobody has judged still gets stored, and no route hands one user another user's rows.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.models import serving as models
from backend.models.agent import PARAPHRASED, QUOTED, REFERENCE, finance
from backend.models.agent.client import SelfAgentClient
from backend.models.agent.core import BUILT, Turn
from backend.database import BUY, SELL, Store
from backend.api.agent_server import create_app
from backend.models.learning import AGENT, ORACLE, FeedbackLog
from backend.models.learning.abstain import Abstainer


from backend.models.agent.run.chat import adapt  # noqa: E402

HAND_PICKED = 0.9980
EVIDENCE = "ticker AAPL ; close 337.00 ; rsi_14 65"
ANSWER = "its 14 day rsi is 65 , which is neither overbought nor oversold ."
# The served generator as `chat.generator_of` names it: a candidate, so only rows carrying its digest count.
SERVED = {"digest": "f10a270801dd7164", "since": None}


class Scripted:
    """Stands in for the agent: the routes must not care what produced the turn.

    Confidence walks down over the session so a fitted cut has a spread to find, which is what makes the
    refit visible at all -- with one confidence value there is no threshold to solve for.
    """

    def __init__(self, served=ANSWER, spoke=True, written=None):
        self.served, self.spoke, self.written = served, spoke, written
        self.market = SimpleNamespace(
            advisor=SimpleNamespace(version="persistence@20d", describe=self._describe),
            paths={"AAPL": Path("AAPL.csv")}, snapshot=self._snapshot, outlook=self._outlook,
            scenarios=SimpleNamespace(describe=lambda: {"name": "garch"}), draw_paths=self._draw_paths)
        self.gate = SimpleNamespace(cut=0.088)
        self.abstainer, self.calibrator = Abstainer(HAND_PICKED), None
        self.asked = []

    @staticmethod
    def _snapshot(ticker, as_of=None):
        return EVIDENCE, {"close": "337.00", "rsi_14": "65"}, "2026-09-17"

    @staticmethod
    def _outlook(ticker, as_of=None):
        return ["2026-09-16", "2026-09-17"], [330.0, 337.0], [0.2, 0.5, 0.3]

    @staticmethod
    def _draw_paths(ticker, steps, count, rng, as_of=None):
        return "2026-09-17", 337.0, rng.normal(0.0, 0.01, (count, steps))

    @staticmethod
    def _describe():
        return {"serving": "gru", "horizon": 5, "edges": [0.011, 0.019], "accuracy": 0.49}

    def answer(self, question, as_of=None, subject=""):
        self.asked.append((question, subject))
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
                     lambda feedback: adapt(feedback, 0.6, 2, AGENT, SERVED),
                     Path("artifacts/advisory_combined.npz"), SERVED["digest"])
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


def test_the_conversations_instrument_is_carried_only_into_a_question_pointing_back_to_it(tmp_path):
    """What makes "and its volatility ?" a question. The first turn names one, so a follow-up is asked under it, and
    general questions in between neither take it nor clear it: "how is your mom doing" was answered with its prices."""
    client, agent, store = _app(tmp_path, judge=_verdicts())
    token, headers = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        socket.receive_json()
        for question in ("is AAPL overbought ?", "how is your mom doing ?", "what is a stop loss ?", "and its volatility ?"):
            _ask(socket, question)

    assert [subject for _, subject in agent.asked] == ["", "", "", "AAPL"]


@pytest.mark.parametrize("reconnected", [False, True])
@pytest.mark.parametrize("refusal", [finance.NO_PRICES, finance.NO_SUBJECT])
def test_a_share_alfa_could_not_fetch_or_find_clears_the_subject_rather_than_keep_the_one_before(tmp_path, reconnected,
                                                                                                 refusal):
    """"should i buy it ?" after a failed fetch of nykaa's prices was asked about AAPL, the share before it, and so it
    was after a share or index ALFA has no file for at all ("bank nifty", or any listed share without --live)."""
    agent = Scripted()
    asked = agent.answer
    agent.answer = lambda question, as_of=None, subject="": (
        asked(question, as_of, subject)._replace(ticker="", served=refusal, spoke=False)
        if "nykaa" in question else asked(question, as_of, subject))
    client, _, _ = _app(tmp_path, agent=agent, judge=_verdicts())
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        conversation = socket.receive_json()["conversation"]
        for question in ("is AAPL overbought ?", "how is nykaa doing ?") + (() if reconnected else ("should i buy it ?",)):
            _ask(socket, question)
    if reconnected:
        with client.websocket_connect("/chat") as socket:
            socket.send_json({"token": token, "conversation": conversation})
            socket.receive_json()
            _ask(socket, "should i buy it ?")

    assert agent.asked[-2:] == [("how is nykaa doing ?", ""), ("should i buy it ?", "")]


class _Socket:
    """The test client's socket behind the two calls the StockIntel client makes on a `websockets` one."""

    def __init__(self, session):
        self.session = session

    def send(self, text):
        self.session.send_text(text)

    def recv(self, timeout=None):
        return self.session.receive_text()


def test_the_stockintel_client_keeps_one_conversation_per_session(tmp_path, monkeypatch):
    """"Analyze RELIANCE" and then a follow-up lost the instrument, because every question opened a new conversation."""
    client, agent, _ = _app(tmp_path, judge=_verdicts())
    token, _ = _signed_in(client)

    @contextmanager
    def connect(url, **_):
        with client.websocket_connect("/chat") as session:
            yield _Socket(session)

    monkeypatch.setattr(SelfAgentClient, "_login", lambda self: token)
    monkeypatch.setattr("websockets.sync.client.connect", connect)
    asking = SelfAgentClient("http://agent.test")
    asking.ask("is AAPL overbought ?", session="one")
    asking.ask("why is it overbought ?", session="one")
    asking.ask("why is it overbought ?", session="")
    assert [subject for _, subject in agent.asked] == ["", "AAPL", ""]
    # the oldest session is forgotten past the cap, so the map cannot grow without end
    monkeypatch.setattr("backend.models.agent.client.CONVERSATIONS_KEPT", 1)
    asking.ask("is AAPL overbought ?", session="two")
    asking.ask("why is it overbought ?", session="one")
    assert [subject for _, subject in agent.asked[3:]] == ["", ""]
    # the stock the app found in this question is what it is asked under, not the one its conversation carries
    asking.ask("how has nykaa been doing ?", "NYKAA.NS", session="one")
    assert agent.asked[-1] == ("how has nykaa been doing ?", "NYKAA.NS")


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


def _serving(tmp_path, monkeypatch, *labellers):
    """serve.app_from's app over a log of twelve rows of the served generator judged by each of `labellers`, built
    with the cut fitted as ask.build fits it."""
    from backend.models.agent.run import serve

    def build(artifacts, checkpoint, prices, gate, log, wanted, warmup, *_, labeller, **__):
        agent = Scripted()
        with FeedbackLog(log) as feedback:
            agent.abstainer, agent.calibrator, learned = adapt(feedback, wanted, warmup, labeller, SERVED)
        return agent, agent.abstainer, learned, Path("advisory_combined.npz")

    monkeypatch.setattr(serve, "build", build)
    monkeypatch.setattr(serve, "generator_of", lambda path: SERVED)
    monkeypatch.setattr(serve, "listed_names", dict)
    (tmp_path / serve.SOURCES).write_text("{}")
    with FeedbackLog(tmp_path / "served.sqlite") as feedback:
        for labeller in labellers:
            for index in range(12):
                feedback.append("2026-09-24T10:00:00+00:00", f"q{index}", EVIDENCE, ANSWER, 0.99 - index / 100,
                                index % 2 == 0, labeller, digest=SERVED["digest"])
    args = SimpleNamespace(artifacts=str(tmp_path), checkpoint="", prices="", gate="",
                           store=str(tmp_path / "served.sqlite"), wanted=0.6, warmup=10, price_head="",
                           reference_index="", reference_checkpoint="", paraphrase=False, live=False, dense=False,
                           symbols="", scenarios="", phrase=False, router="pretrained",
                           verdicts=str(tmp_path / "verdicts.jsonl"))
    return serve.app_from(args)[0]


# ORACLE: a newly promoted generator's log holds only the oracle rows reseed_feedback appended.
@pytest.mark.parametrize("labeller", [AGENT, ORACLE])
def test_the_server_fits_its_first_cut_with_the_judge_it_refits_with(tmp_path, monkeypatch, labeller):
    """A fresh server refused its first turn at the hand-picked cut and answered later ones at 0.971: it fitted
    at start on the oracle's rows and after every turn on the agent's."""
    client = TestClient(_serving(tmp_path, monkeypatch, labeller))
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        ready = socket.receive_json()
        *_, stored = _ask(socket, "is AAPL overbought ?")

    assert stored["judged"] is None and stored["learned"]
    assert ready["answer_cut"] == stored["answer_cut"] != HAND_PICKED


def test_a_server_whose_log_cannot_teach_its_generator_a_cut_will_not_start(tmp_path, monkeypatch):
    """It served the hand-picked 0.998 instead, so a deployment that forgot the reseed looked like a working one."""
    with pytest.raises(LookupError, match="reseed_feedback"):
        _serving(tmp_path, monkeypatch)


def test_the_user_questions_eval_will_not_measure_at_a_cut_the_server_would_not_start_with(monkeypatch):
    """It threw `learned` away, so with no rows of its generator it scored answers at the hand-picked 0.998."""
    from backend import cli
    from backend.models.agent.run import ask, chat
    from backend.models.external import eval_user_questions

    for name, stand_in in (("build", lambda *_, **__: (Scripted(), Abstainer(HAND_PICKED), False, Path("x.npz"))),
                           ("writer_from", lambda phrase: None), ("reference_from", lambda *_, **__: None)):
        monkeypatch.setattr(ask, name, stand_in)
    monkeypatch.setattr(chat, "generator_of", lambda path: SERVED)
    monkeypatch.setattr(cli, "_orchestrator", lambda *_: pytest.fail("measured at the hand-picked cut"))
    monkeypatch.setattr(sys, "argv", ["main"])
    with pytest.raises(LookupError, match="reseed_feedback"):
        eval_user_questions.main()


def test_the_user_questions_eval_answers_at_the_cut_the_server_serves(tmp_path, monkeypatch):
    """It fitted its cut from chat.sqlite's oracle rows, 0.818, while the server served 0.971 from the agent's."""
    from backend.models.agent.run import ask, serve
    from backend.models.external import eval_user_questions

    class Built(Exception):
        pass

    def build(artifacts, checkpoint, prices, gate, log, *_, labeller, **__):
        raise Built(log, labeller)

    fitted = []
    for module, main, argv in ((serve, serve.main, ["--verdicts", str(tmp_path / "verdicts.jsonl")]),
                               (ask, eval_user_questions.main, [])):
        for name, stand_in in (("build", build), ("writer_from", lambda phrase: None),
                               ("reference_from", lambda *_, **__: None)):
            monkeypatch.setattr(module, name, stand_in)
        monkeypatch.setattr(sys, "argv", ["main", *argv])
        with pytest.raises(Built) as built:
            main()
        fitted.append(built.value.args)
    assert fitted[0] == fitted[1] == (fitted[0][0], AGENT)


def test_the_user_questions_eval_never_judges_our_codes_words_as_a_language_models(tmp_path, monkeypatch):
    """It counted every answer whose `phrased_by` was set, our code's sentences and facts replies included."""
    from backend import cli
    from backend.models.agent.run import ask, chat
    from backend.models.external import eval_user_questions

    built = Turn("is AAPL overbought ?", "", "AAPL", "overbought", ANSWER, "", EVIDENCE, float("nan"), None, True, 0.49,
                 [], "", "2026-09-17", BUILT)
    for name, stand_in in (("build", lambda *_, **__: (SimpleNamespace(answer=lambda q: built), None, True, Path("x"))),
                           ("writer_from", lambda phrase: None), ("reference_from", lambda *_, **__: None),
                           ("listed_names", dict)):
        monkeypatch.setattr(ask, name, stand_in)
    monkeypatch.setattr(chat, "generator_of", lambda path: SERVED)
    app = SimpleNamespace(hub=type("Hub", (), {})(), handle=lambda *_: SimpleNamespace(intent="any", text="Fine."))
    monkeypatch.setattr(cli, "_orchestrator", lambda *_: app)
    (tmp_path / "reference_sources.json").write_text("{}")
    (tmp_path / "asked.json").write_text(json.dumps([{"q": built.question, "agent": "any", "app": "any"}]))
    for name, value in (("ARTIFACTS", tmp_path), ("QUESTIONS", tmp_path / "asked.json"), ("OUT", tmp_path / "out.json")):
        monkeypatch.setattr(eval_user_questions, name, value)
    judge = SimpleNamespace(predict=lambda pairs: [0.0] * len(pairs))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForSequenceClassification=SimpleNamespace(
        from_pretrained=lambda *_, **__: SimpleNamespace(eval=lambda: judge))))
    monkeypatch.setattr(sys, "argv", ["main"])
    eval_user_questions.main()
    assert json.loads((tmp_path / "out.json").read_text())["summary"]["LM-worded answers"] == 0


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


class Quoting(Scripted):
    """A reference turn: an answer the advisory generator never wrote, with its confidence or none."""

    def __init__(self, how, confidence):
        super().__init__()
        self.how, self.confidence = how, confidence

    def answer(self, question, as_of=None, subject=""):
        return super().answer(question, as_of, subject)._replace(intent=f"{REFERENCE} {self.how}",
                                                                 confidence=self.confidence)


@pytest.mark.parametrize("how,confidence", [(QUOTED, float("nan")), (PARAPHRASED, 0.5)])
def test_a_reference_turn_is_stored_but_never_judged_into_the_advisory_log(tmp_path, how, confidence):
    """A quote with a verdict waiting crashed the socket on its NaN confidence, and a paraphrase's confidence
    is another model's, so either one in the log would fit the advisory cut on numbers it never reported."""
    client, _, store = _app(tmp_path, agent=Quoting(how, confidence), judge=_verdicts(True))
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        ready = socket.receive_json()
        stored = _ask(socket, "what is a stop loss order ?")[2]

    assert stored["judged"] is None and store.feedback.rows(judged_only=False) == []
    assert len(store.recent(token, ready["conversation"])) == 1


def test_a_turn_answering_two_questions_is_stored_with_its_least_confidence_but_never_judged(tmp_path):
    """Its text is two answers, so no verdict on it relates one generation's confidence to being right."""
    agent = Scripted()
    asked = agent.answer
    agent.answer = lambda question, as_of=None, subject="": asked(question, as_of, subject)._replace(
        intent="performance and buy", confidence=0.97)
    client, _, store = _app(tmp_path, agent=agent, judge=_verdicts(True))
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        ready = socket.receive_json()
        stored = _ask(socket, "how is AAPL performing and should i buy it")[2]

    assert stored["judged"] is None and store.feedback.rows(judged_only=False) == []
    (message,) = store.recent(token, ready["conversation"])
    assert (message.intent, message.confidence) == ("performance and buy", 0.97)


def test_a_facts_turn_is_stored_with_its_intent_and_only_price_answers_are_judged_into_the_advisory_log(tmp_path):
    """The cut refitted from this log gates price answers: facts rows near 0.95 judged wrong moved it toward 0.97-0.99."""
    agent = Scripted()
    asked = agent.answer
    agent.answer = lambda question, as_of=None, subject="": asked(question, as_of, subject)._replace(
        **({"intent": finance.FACTS, "confidence": 0.95} if "debt" in question else {}))
    client, _, store = _app(tmp_path, agent=agent, judge=_verdicts(False, True))
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        ready = socket.receive_json()
        facts = _ask(socket, "how much debt does AAPL have ?")[2]
        _ask(socket, "is AAPL overbought ?")

    assert facts["judged"] is None
    assert [row["extra"]["intent"] for row in store.feedback.rows(judged_only=False)] == ["overbought"]
    assert [message.intent for message in store.recent(token, ready["conversation"])] == [finance.FACTS, "overbought"]


def test_a_turn_our_code_answered_is_judged_as_the_answer_of_alfas_it_replaced_which_did_not_speak(tmp_path):
    """The log relates ALFA's confidence to whether ALFA was right, so our sentence spoken in its place is not ALFA's."""
    agent = Scripted()
    asked = agent.answer
    agent.answer = lambda question, as_of=None, subject="": asked(question, as_of, subject)._replace(
        served="AAPL reads 65 on the 14 day rsi.", because="names another share: msft", phrased_by=BUILT)
    client, _, store = _app(tmp_path, agent=agent, judge=_verdicts(False))
    token, _ = _signed_in(client)

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": token})
        ready = socket.receive_json()
        _, turn, stored = _ask(socket, "is AAPL overbought ?")

    (row,) = store.feedback.rows()
    assert (stored["judged"], row["answer"], row["extra"]["spoke"]) == (False, ANSWER, False)
    assert (turn["spoke"], turn["phrased_by"], store.recent(token, ready["conversation"])[0].spoke) == (True, BUILT, 1)


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


@pytest.mark.parametrize("route", ["/instruments", "/instruments/aapl", "/instruments/aapl/outlook",
                                   "/instruments/aapl/scenarios"])
def test_a_well_formed_token_that_is_no_session_is_refused(tmp_path, route):
    """The instrument routes checked only the header's shape, so "Bearer x" drew 2,000 paths."""
    client, _, _ = _app(tmp_path)
    assert client.get(route, headers={"Authorization": "Bearer made-up"}).status_code == 403


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


def test_an_instrument_serves_the_snapshot_and_an_unknown_one_is_a_404(tmp_path):
    """The dashboard reads this, and it must show the figures the agent was given, not others."""
    client, _, _ = _app(tmp_path)
    _, headers = _signed_in(client)
    assert client.get("/instruments", headers=headers).json() == ["AAPL"]
    assert client.get("/instruments/aapl", headers=headers).json() == {
        "symbol": "AAPL", "as_of": "2026-09-17", "facts": {"close": "337.00", "rsi_14": "65"},
        "evidence": EVIDENCE}

    missing = client.get("/instruments/NOSUCH", headers=headers)
    assert missing.status_code == 404 and "no price file for NOSUCH" in missing.json()["detail"]


def test_the_outlook_route_carries_the_bands_and_refuses_to_imply_a_direction(tmp_path):
    """The forecast panel draws this. It may show what the head predicts -- a volatility class -- and the
    measurement that says a direction call is worse than guessing, and it may not be handed a direction."""
    client, _, _ = _app(tmp_path)
    _, headers = _signed_in(client)
    drawn = client.get("/instruments/aapl/outlook", headers=headers).json()

    assert drawn["closes"] == [330.0, 337.0] and drawn["probabilities"] == [0.2, 0.5, 0.3]
    assert [band["name"] for band in drawn["bands"]] == list(models.CLASSES)
    assert drawn["direction"]["accuracy"] < drawn["direction"]["baseline"]
    assert "direction" not in drawn["model"] and drawn["model"]["serving"] == "gru"

    missing = client.get("/instruments/NOSUCH/outlook", headers=headers)
    assert missing.status_code == 404


def test_learned_counts_only_the_asking_users_own_turns(tmp_path):
    """The feedback table has no user column, so the join is the only thing keeping these apart."""
    client, _, _ = _app(tmp_path, judge=_verdicts(True, False, True))
    mine_token, mine = _signed_in(client, "mine@example.test")
    theirs_token, theirs = _signed_in(client, "theirs@example.test")

    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": mine_token})
        socket.receive_json()
        _ask(socket, "is AAPL overbought ?")
        _ask(socket, "and now ?")
    with client.websocket_connect("/chat") as socket:
        socket.send_json({"token": theirs_token})
        socket.receive_json()
        _ask(socket, "is AAPL overbought ?")

    ours = client.get("/learned", headers=mine).json()
    assert (ours["asked"], ours["spoke"], ours["right"], ours["wrong"]) == (2, 2, 1, 1)
    # Their one turn is in the same log the cut was fitted from, and still not in our counts.
    assert client.get("/learned", headers=theirs).json()["asked"] == 1
    assert ours["log_rows"] == 3 and ours["answer_cut"] != HAND_PICKED


@pytest.mark.parametrize("email", ["one@example.test"])
def test_an_email_can_only_register_once(tmp_path, email):
    client, _, _ = _app(tmp_path)
    assert client.post("/auth/register", json={"email": email, "password": "a"}).status_code == 201
    again = client.post("/auth/register", json={"email": email, "password": "b"})
    assert again.status_code == 400 and "already has an account" in again.json()["detail"]


def test_the_scenarios_route_is_seeded_names_its_model_and_never_makes_a_call(tmp_path):
    """The same request has to draw the same picture, or a user comparing two looks at one chart sees
    noise and reads it as news. And the sentence saying these are ranges travels with every response."""
    client, _, _ = _app(tmp_path)
    _, headers = _signed_in(client)
    first = client.get("/instruments/aapl/scenarios?steps=10&paths=100&seed=3", headers=headers).json()
    again = client.get("/instruments/aapl/scenarios?steps=10&paths=100&seed=3", headers=headers).json()
    other = client.get("/instruments/aapl/scenarios?steps=10&paths=100&seed=4", headers=headers).json()

    assert first["fan"] == again["fan"] and first["fan"] != other["fan"]
    assert len(first["fan"]["0.5"]) == 10 and len(first["examples"]) == 5
    assert first["model"]["name"] == "garch" and "not which way" in first["direction"]
    assert client.get("/instruments/aapl/scenarios?steps=999", headers=headers).status_code == 400
    assert client.get("/instruments/NOSUCH/scenarios", headers=headers).status_code == 404


def test_the_agent_loads_the_checkpoint_where_the_registry_says_it_is(tmp_path, monkeypatch):
    """A generator promoted from alfa_v2_data was loaded as artifacts/<its stem>.npz, a file that is not there."""
    from backend.models.agent.run import ask
    from backend.models.serving import registry

    class Loaded(Exception):
        pass

    promoted = tmp_path / "alfa_v2_data" / "alfa_v2.step2000.npz"
    monkeypatch.setattr(registry, "serving", lambda role: promoted)
    monkeypatch.setattr(ask, "load_model", lambda artifacts, dataset, checkpoint: (_ for _ in ()).throw(Loaded(checkpoint)))
    for checkpoint, expected in (("", promoted), ("alfa_v2_data/alfa_v2.step2000.npz", promoted)):
        with pytest.raises(Loaded) as loaded:
            ask.build(tmp_path, checkpoint, "", "", "", 0.6, 10, "", labeller=AGENT)
        assert loaded.value.args[0] == expected
