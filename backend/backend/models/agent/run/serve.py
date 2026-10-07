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
from backend.paths import ARTIFACTS, DATA

import argparse
import json
import sys
from pathlib import Path
import uvicorn  # noqa: E402

from backend.models.agent.run.ask import ROUTERS, build, listed_names, reference_from, writer_from  # noqa: E402
from backend.database import Store  # noqa: E402
from backend.api.agent_server import create_app  # noqa: E402
from backend.models.agent.run.chat import adapt  # noqa: E402
from backend.models.learning.teacher import AgentTeacher  # noqa: E402
from backend.models.agent.present import present  # noqa: E402

# {document key: source URL} for the reference index, built once from the data manifest (present.source_map)
SOURCES = "reference_sources.json"


def app_from(args):
    """The app, the store it writes to, and the teacher holding the verdicts that shaped the cut."""
    # One file for the accounts and the feedback, so a stored message can point at the row that judged
    # it, and the cuts are refitted from that same log.
    artifacts = Path(args.artifacts)
    writer = writer_from(args.phrase)
    agent, _, _, served = build(artifacts, args.checkpoint, args.prices, args.gate, args.store,
                                args.wanted, args.warmup, args.price_head,
                                reference_from(artifacts, args.reference_index,
                                               args.reference_checkpoint, args.paraphrase, args.live, writer,
                                               args.dense),
                                args.symbols, args.scenarios, writer=writer, listed=listed_names(),
                                router=args.router)
    teacher = AgentTeacher(args.verdicts)
    store = Store(args.store)
    names, sources = listed_names(), json.loads((artifacts / SOURCES).read_text())
    app = create_app(agent, store,
                     lambda question, evidence, answer:
                         teacher.judge(question, evidence, answer) + (teacher.name,),
                     lambda feedback: adapt(feedback, args.wanted, args.warmup, teacher.name),
                     served, lambda turn: present(turn, names, sources))
    return app, store, teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    parser.add_argument("--checkpoint", default="",
                        help="a candidate to serve instead of the promoted one")
    parser.add_argument("--prices", default=str(DATA / "prices"))
    parser.add_argument("--router", choices=sorted(ROUTERS), default="alfa",
                        help="ALFA's encoder, or the pretrained sentence encoder that routes more questions")
    parser.add_argument("--gate", default="", help="a routing cut other than the one fitted for --router")
    parser.add_argument("--price-head", default=str(ARTIFACTS / "price_head.npz"))
    parser.add_argument("--scenarios", default=str(ARTIFACTS / "returns.npz"),
                        help="the return generator; empty serves no sampled price paths")
    parser.add_argument("--reference-index", default="reference_index.npz",
                        help="empty serves the seven routed intents and refuses everything else")
    parser.add_argument("--reference-checkpoint", default="generator.npz")
    parser.add_argument("--paraphrase", action="store_true",
                        help="serve the reference generator's words rather than the passage it read")
    parser.add_argument("--live", action="store_true",
                        help="search SEC EDGAR for a company that files there and Wikipedia otherwise, "
                             "before reading the corpus for the first and after it for the second; off by "
                             "default, and off is what makes this deployment offline")
    parser.add_argument("--symbols", default=str(DATA / "raw/sec_edgar/company_tickers.json"),
                        help="SEC's symbol table: an instrument's company name and filing number")
    parser.add_argument("--phrase", action="store_true",
                        help="reword each answer with the language model (the model Space's when one is set)")
    parser.add_argument("--dense", action="store_true",
                        help="find reference passages by meaning and rerank them with a cross-encoder")
    parser.add_argument("--store", default=str(ARTIFACTS / "served.sqlite"),
                        help="accounts, conversations and the feedback this deployment learns from")
    parser.add_argument("--verdicts", default=str(ARTIFACTS / "served_verdicts.jsonl"))
    parser.add_argument("--requests", default=str(ARTIFACTS / "served_requests.jsonl"),
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
