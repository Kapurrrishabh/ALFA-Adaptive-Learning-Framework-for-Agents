#!/usr/bin/env python3
"""B7's loop, driven over C7's socket instead of in-process. Start `serve.py` first.

Same session this prints in chat.py -- what was asked, what was said or withheld, and where the cut went
-- with two differences that are the point of C7 rather than details of it.

  the questions are typed, not drawn from a dataset. So no gold answer comes with them and the oracle
  cannot judge: the verdicts are the agent's, replayed from the file serve.py was started with. A
  question nobody has judged is still asked, answered and stored; it just cannot move the cut.

  the client sees only frames. It never imports the model, the store or the cut, so anything it reports
  had to survive being serialised, which is the part an in-process loop cannot check.

Run it once against an empty verdict file to collect the pending pairs, judge those, then run it again:
the rng is seeded per process, so asking the same questions in the same order reproduces the same answers
and the replayed verdicts land on the rows they were written for.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask import DEMO  # noqa: E402

# DEMO's six reach every stage that can stop a turn, including the poem, which is refused before the model
# runs and so has nothing to judge. The rest are the same intents typed the way people actually type them,
# across tickers the price files cover.
TYPED = DEMO + (
    "should i buy NVDA ?",
    "how volatile is AMZN at the moment ?",
    "how far is GOOGL off its recent high ?",
    "what did META earn last quarter ?",
    "is nvda overbought",
    "how has MSFT been doing over the last month ?",
    "how risky is AMD over the next week ?",
    "hows tsla lookin",
    "how far off its high is AAPL ?",
    "should i buy msft",
    "how volatile is TSLA right now ?",
    "is AMZN overbought right now ?",
    "how has NVDA been doing lately ?",
    "how risky is GOOGL over the next week ?",
)


def session(base, token, questions):
    """Ask each question over one socket, printing each turn as it lands. Returns the frames."""
    stored = []
    with connect(f"{base.replace('http', 'ws', 1)}/chat", open_timeout=30) as socket:
        socket.send(json.dumps({"token": token, "subject": "typed"}))
        ready = json.loads(socket.recv())
        print(f"conversation {ready['conversation']}, answer cut {ready['answer_cut']:.5f}")

        for number, question in enumerate(questions, start=1):
            socket.send(json.dumps({"question": question}))
            frames = {}
            while "stored" not in frames:
                # Read until the turn is stored: the stages arrive in order and the last one is the
                # only one that knows whether the cut moved.
                frame = json.loads(socket.recv(timeout=600))
                frames[frame["stage"]] = frame
            turn, done = frames["turn"], frames["stored"]
            print(f"\nturn {number}  {question}")
            print(f"  {'answered:' if turn['spoke'] else 'stayed quiet:'} {turn['served']}"
                  + (f"  [{turn['because']}]" if not turn["spoke"] else ""))
            if turn["wrote"]:
                print(f"  wrote     {turn['wrote']!r}")
            verdict = "unjudged" if done["judged"] is None else f"judged {done['judged']}"
            print(f"  {verdict} ({done['why']});  cut now {done['answer_cut']:.5f}"
                  + ("  <- moved" if done["moved"] else ""))
            stored.append(done)
    return stored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--email", default="viva@example.test")
    parser.add_argument("--password", default="viva-demo")
    parser.add_argument("--turns", type=int, default=20)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base, timeout=30) as client:
        credentials = {"email": args.email, "password": args.password}
        client.post("/auth/register", json=credentials)
        signed_in = client.post("/auth/login", json=credentials)
        signed_in.raise_for_status()
        token = signed_in.json()["token"]
        print(json.dumps(client.get("/model").json()))

    stored = session(args.base, token, TYPED[: args.turns])

    judged = [frame for frame in stored if frame["judged"] is not None]
    cuts = [frame["answer_cut"] for frame in stored]
    print(f"\n{len(stored)} turns over the socket, {len(judged)} judged "
          f"({sum(frame['judged'] for frame in judged)} right), "
          f"{sum(frame['moved'] for frame in stored)} of them moved the cut")
    print(f"  cut {cuts[0]:.5f} -> {cuts[-1]:.5f}"
          + ("" if stored[-1]["learned"] else " — never left the hand-picked value, too little feedback"))
    print(f"  stored as messages {stored[0]['message']}..{stored[-1]['message']}")


if __name__ == "__main__":
    sys.exit(main())
