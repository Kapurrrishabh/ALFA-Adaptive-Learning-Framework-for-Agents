#!/usr/bin/env python3
"""Start the API: assemble the agent exactly as `ask.py` does, then hand it to the routes.

The assembly lives in one place and both entry points call it, so the agent behind the socket is the
agent the C1 gate demonstrates -- same checkpoint from the registry, same routing cut, same answer cut
refitted from the same log. Nothing here chooses a model.

**Who judges a typed question.** The oracle cannot: it needs the gold answer the built dataset carries,
and a question a person types has none. So the judge is `AgentTeacher` -- Claude, in session -- which
replays verdicts it has already written and reports a miss as *unjudged* rather than guessing. An
unjudged turn is still stored and still answered; it just does not move the cut. `--requests` writes the
pending pairs out for judging, and the next run replays them.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from ask import build  # noqa: E402
from backend.database import Store  # noqa: E402
from backend.main import create_app  # noqa: E402
from chat import adapt  # noqa: E402
from selfagent.learn.teacher import AgentTeacher  # noqa: E402


def app_from(args):
    """The app, the store it writes to, and the teacher holding the verdicts that shaped the cut."""
    # One file for the accounts and the feedback, so a stored message can point at the row that judged
    # it, and the cuts are refitted from that same log.
    agent, _, _, served = build(Path(args.artifacts), args.checkpoint, args.prices, args.gate,
                                args.store, args.wanted, args.warmup, args.price_head)
    teacher = AgentTeacher(args.verdicts)
    store = Store(args.store)
    app = create_app(agent, store,
                     lambda question, evidence, answer:
                         teacher.judge(question, evidence, answer) + (teacher.name,),
                     lambda feedback: adapt(feedback, args.wanted, args.warmup, teacher.name),
                     served)
    return app, store, teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--checkpoint", default="",
                        help="a candidate to serve instead of the promoted one")
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--gate", default="artifacts/route_gate.json")
    parser.add_argument("--price-head", default="artifacts/price_head.npz")
    parser.add_argument("--store", default="artifacts/served.sqlite",
                        help="accounts, conversations and the feedback this deployment learns from")
    parser.add_argument("--verdicts", default="artifacts/served_verdicts.jsonl")
    parser.add_argument("--requests", default="artifacts/served_requests.jsonl",
                        help="where the pairs needing a verdict are written when the server stops")
    parser.add_argument("--wanted", type=float, default=0.6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app, store, teacher = app_from(args)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        pending = teacher.write_requests(args.requests)
        store.close()
        print(f"{pending} answers are waiting for a verdict in {args.requests}")


if __name__ == "__main__":
    sys.exit(main())
