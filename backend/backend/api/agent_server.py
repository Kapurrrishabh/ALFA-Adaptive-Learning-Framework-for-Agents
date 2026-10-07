"""The HTTP and WebSocket surface. Routes only: everything they call was built and measured elsewhere.

`create_app` is given the agent, the store, the judge and the refit rather than paths to them. The same
reason `Agent` takes its model instead of loading one: a served agent and a measured one have to be the
same object, and a factory that loaded its own would make "which checkpoint answered this" unanswerable.
It is also what lets the tests drive every route without a 30 MB checkpoint.

**The conversation streams by stage, not by token.** A frame goes out when its fact becomes true -- the
question was received, the turn finished, it was stored, the adaptation moved -- and no frame pretends to
be progress. `model.generate` returns a whole sequence, so cutting a finished string into fake tokens
would be theatre, and this is the one place where a UI detail could quietly misrepresent the model.

**Two threads, and only one of them touches SQLite.** A connection belongs to the thread that opened it,
so every store call happens on the event loop and only the decoder -- seconds of NumPy, no database --
goes to a threadpool. One lock around it, because there is one set of weights and one rng behind it.
"""

import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.models.core.backend import default_rng
from backend.models.data import advisory

from backend.models import serving as models
from backend.database import NotYours

# The frontend is served from somewhere else in development, so the browser will not talk to this at all
# without it. Kept to localhost: a wildcard here would let any page a user visits spend their session.
DEVELOPMENT_ORIGINS = ("http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173")


class Credentials(BaseModel):
    email: str
    password: str


class NewConversation(BaseModel):
    subject: str = ""


class NewTrade(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: float


def create_app(agent, store, judge, refit, served):
    """The app. `judge(question, evidence, answer)` returns (is_right or None, why, labeller); `refit`
    takes the feedback log and returns (abstainer, calibrator, learned)."""
    app = FastAPI(title="ALFA", description=__doc__)
    app.add_middleware(CORSMiddleware, allow_origins=list(DEVELOPMENT_ORIGINS),
                       allow_methods=["*"], allow_headers=["*"])
    generating = asyncio.Lock()

    @app.exception_handler(NotYours)
    async def _not_yours(_, refused):
        return JSONResponse({"detail": str(refused)}, status_code=403)

    @app.exception_handler(ValueError)
    async def _bad_request(_, refused):
        return JSONResponse({"detail": str(refused)}, status_code=400)

    @app.post("/auth/register", status_code=201)
    async def register(credentials: Credentials):
        return {"user": store.register(credentials.email, credentials.password)}

    @app.post("/auth/login")
    async def log_in(credentials: Credentials):
        return {"token": store.log_in(credentials.email, credentials.password)}

    @app.post("/auth/logout")
    async def log_out(token=Header(default="", alias="Authorization")):
        store.log_out(_bearer(token))
        return {"signed_out": True}

    @app.get("/conversations")
    async def conversations(token=Header(default="", alias="Authorization")):
        return [{"id": row[0], "started": row[1], "subject": row[2], "messages": row[3]}
                for row in store.conversations(_bearer(token))]

    @app.post("/conversations", status_code=201)
    async def start(new: NewConversation, token=Header(default="", alias="Authorization")):
        return {"id": store.start_conversation(_bearer(token), new.subject)}

    @app.get("/conversations/{conversation}/messages")
    async def messages(conversation: int, token=Header(default="", alias="Authorization")):
        return [message._asdict() for message in store.recent(_bearer(token), conversation)]

    @app.get("/portfolio")
    async def portfolio(token=Header(default="", alias="Authorization")):
        return [position._asdict() for position in store.holdings(_bearer(token))]

    @app.post("/trades", status_code=201)
    async def trade(new: NewTrade, token=Header(default="", alias="Authorization")):
        return {"trade": store.record_trade(_bearer(token), new.symbol, new.side, new.quantity,
                                            new.price)}

    @app.get("/instruments")
    async def instruments(token=Header(default="", alias="Authorization")):
        """What can be asked about, from the price files on disk. There is no other universe."""
        _bearer(token)
        return sorted(agent.market.paths)

    @app.get("/instruments/{symbol}")
    async def instrument(symbol: str, as_of: str = None,
                         token=Header(default="", alias="Authorization")):
        """The same snapshot the agent is given for this instrument, and nothing the agent cannot see.

        Deliberately the figures only: putting a generated read here would be a second answering path
        beside `/chat`, measured by nothing, and the guard that screens figures runs in that one.
        """
        _bearer(token)
        held = symbol.upper()
        if held not in agent.market.paths:
            raise HTTPException(404, f"no price file for {symbol}; ask /instruments for the universe")
        evidence, shown, taken_at = agent.market.snapshot(held, as_of)
        return {"symbol": held, "as_of": str(taken_at), "facts": shown, "evidence": evidence}

    @app.get("/instruments/{symbol}/outlook")
    async def outlook(symbol: str, as_of: str = None,
                      token=Header(default="", alias="Authorization")):
        """The week-ahead call behind the risk answer, with the history and bands a client can draw.

        Same bars, same as-of rule and same advisor as `/chat`, so the picture cannot disagree with the
        answer. No direction: it measured below its own baseline, and `direction` carries that measurement
        so a client shows the reason rather than a badge.
        """
        _bearer(token)
        held = symbol.upper()
        if held not in agent.market.paths:
            raise HTTPException(404, f"no price file for {symbol}; ask /instruments for the universe")
        advisor = agent.market.advisor
        if advisor is None:
            raise HTTPException(503, "no price head is loaded, so there is no outlook to draw")
        dates, closes, probabilities = agent.market.outlook(held, as_of)
        card = advisor.describe()
        return {"symbol": held, "as_of": str(dates[-1]), "dates": [str(day) for day in dates],
                "closes": closes, "probabilities": probabilities, "horizon": card["horizon"],
                "bands": models.price_bands(closes[-1], card["edges"], card["horizon"]),
                "model": card, "direction": advisory.DIRECTION_SKILL}

    @app.get("/instruments/{symbol}/scenarios")
    async def scenarios(symbol: str, steps: int = 20, paths: int = 200, as_of: str = None, seed: int = 0,
                        token=Header(default="", alias="Authorization")):
        """Possible price paths from the as-of date, as a fan of quantiles plus a few whole paths.

        Seeded, so the same request draws the same picture twice. Which model drew it and what it measured
        travel with the answer, and so does `direction`: these are ranges, never calls.
        """
        _bearer(token)
        held = symbol.upper()
        if held not in agent.market.paths:
            raise HTTPException(404, f"no price file for {symbol}; ask /instruments for the universe")
        if agent.market.scenarios is None:
            raise HTTPException(503, "no return model is loaded, so there are no paths to draw")
        if not (1 <= steps <= 250 and 10 <= paths <= 2000):
            raise HTTPException(400, f"steps must be 1-250 and paths 10-2000; got {steps} and {paths}")
        taken, close, drawn = await run_in_threadpool(
            agent.market.draw_paths, held, steps, paths, default_rng(seed), as_of)
        return {"symbol": held, "as_of": str(taken), "close": close, "steps": steps, "paths": paths,
                "fan": models.scenarios.fan(close, drawn),
                "examples": models.scenarios.priced(close, drawn[:5]).tolist(),
                "model": agent.market.scenarios.describe(), "direction": models.scenarios.DIRECTION}

    @app.get("/learned")
    async def learned(token=Header(default="", alias="Authorization")):
        """What this user's feedback has done to the agent, and what it was fitted from.

        `answer_cut` and `log_rows` are the deployment's, not this user's: one cut is fitted from the
        whole log. Saying so here is the point -- the page has to show which part is yours.
        """
        mine = store.learning(_bearer(token))
        agreed, of = store.feedback.agreement()
        return {**mine._asdict(), "answer_cut": agent.abstainer.cut, "routing_cut": agent.gate.cut,
                "log_rows": len(store.feedback), "judges_agreed": agreed, "judges_compared": of}

    @app.get("/model")
    async def model():
        """What is answering, so a client can show it rather than trusting the deployment."""
        advisor = agent.market.advisor
        return {"generator": served.name,
                "outlook": advisor.version if advisor else None,
                "routing_cut": agent.gate.cut,
                "answer_cut": agent.abstainer.cut}

    @app.websocket("/chat")
    async def chat(socket: WebSocket):
        """B7's loop over one socket: ask, answer, judge, adapt, and say what moved."""
        await socket.accept()
        opening = await socket.receive_json()
        try:
            token = opening["token"]
            conversation = (opening.get("conversation")
                            or store.start_conversation(token, opening.get("subject", "")))
            # Reading it is the ownership check: someone else's conversation raises here rather than
            # after the first answer has already been written into it. It also carries the subject over a
            # reconnect, which is the case a variable in this loop cannot cover.
            spoken = store.recent(token, conversation)
        except (KeyError, NotYours) as refused:
            await socket.send_json({"stage": "refused", "because": str(refused) or
                                    "the first frame has to carry a session token"})
            # 1008 rather than a normal close: the socket is being refused, not finished with.
            await socket.close(code=1008)
            return

        await socket.send_json({"stage": "ready", "conversation": conversation,
                                "answer_cut": agent.abstainer.cut})
        subject = next((message.ticker for message in reversed(spoken) if message.ticker), "")
        while True:
            try:
                frame = await socket.receive_json()
            except WebSocketDisconnect:
                return
            question = (frame.get("question") or "").strip()
            if not question:
                await socket.send_json({"stage": "refused", "because": "no question in that frame"})
                continue
            await socket.send_json({"stage": "working", "question": question})
            async with generating:
                turn = await run_in_threadpool(agent.answer, question, frame.get("as_of"), subject)
            # Only ever replaced, never cleared: a general question in the middle of a conversation about
            # one instrument does not mean the next one has changed the subject.
            subject = turn.ticker or subject
            await socket.send_json({"stage": "turn", **_as_frame(turn)})
            await socket.send_json({"stage": "stored", **_store_turn(store, token, conversation, turn,
                                                                    judge, refit, agent)})

    def _bearer(header):
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "send the session token as 'Authorization: Bearer <token>'")
        return token

    return app


def _as_frame(turn):
    """A turn as a client reads it. `wrote` appears only when a guard changed what was served."""
    return {"question": turn.question, "asked": turn.asked, "ticker": turn.ticker,
            "intent": turn.intent, "served": turn.served, "evidence": turn.evidence,
            "spoke": turn.spoke, "because": turn.because, "as_of": turn.as_of,
            "margin": _number(turn.margin), "confidence": _number(turn.confidence),
            "stated": _number(turn.stated), "unsupported": list(turn.unsupported),
            "wrote": turn.answer if turn.answer != turn.served else None,
            "phrased_by": turn.phrased_by, "unphrased_because": turn.unphrased_because}


def _store_turn(store, token, conversation, turn, judge, refit, agent):
    """Judge the turn, store it with the verdict that judged it, and refit from the log as it stands.

    The order is B7's and it matters: the cut that decided this turn was fitted before it, and the one
    this returns decides the next. A cut fitted on the turn it judges would report its own answer back.

    A turn the decoder never reached is remembered but not judged. The log exists to relate confidence
    to correctness, and that turn has no confidence -- storing a 0.0 would put a number the model never
    reported into the fit. Routing is gated on its own measurement, not on this log.
    """
    if turn.answer:
        is_right, why, labeller = judge(turn.question, turn.evidence, turn.answer)
    else:
        is_right, why, labeller = None, "nothing was generated to judge", None
    judged = None
    if is_right is not None:
        judged = store.feedback.append(datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       turn.question, turn.evidence, turn.answer,
                                       _number(turn.confidence), is_right, labeller, why,
                                       spoke=turn.spoke)
    message = store.remember(token, conversation, turn, feedback_id=judged)
    was = agent.abstainer.cut
    abstainer, calibrator, learned = refit(store.feedback)
    agent.abstainer, agent.calibrator = abstainer, calibrator
    return {"message": message, "judged": is_right, "why": why, "labeller": labeller,
            "answer_cut": abstainer.cut, "moved": abstainer.cut != was, "learned": learned}


def _number(value):
    """None for a NaN, because JSON has no way to write one and a client would read it as a string."""
    return None if value is None or value != value else float(value)
