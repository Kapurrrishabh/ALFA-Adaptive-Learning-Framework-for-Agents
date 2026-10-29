"""Does the app answer what people actually ask? Realistic questions through the agent and the app chat.

Each question in models/training/user_questions.json names what a right turn looks like: the agent's intent
(an advisory intent, "recent", "lesson", "answer", "refuse" or "any") and the app chat's handler. Three numbers
come out of it, two of them against outside standards:

- how many questions got the kind of turn they should, per path;
- reading level of what was served, as a Flesch-Kincaid grade; plain-language guidance asks for 9 or below;
- the hallucination rate of every answer the language model worded, judged against the sources it was given by
  Vectara's HHEM-2.1-Open, the model behind the public hallucination leaderboard (a score under 0.5 is a
  hallucination there, and leading models sit near 1-3%).

    python -m backend.models.external.eval_user_questions
"""
import argparse
import json
import re
import time
from argparse import Namespace
from pathlib import Path

from backend.paths import ARTIFACTS, DATA

QUESTIONS = Path(__file__).resolve().parents[1] / "training" / "user_questions.json"
OUT = ARTIFACTS / "external" / "user_questions_eval.json"
HHEM = "vectara/hallucination_evaluation_model"
# pinned: the judge runs code from its own repo, so a later push must not change it under a past measurement
HHEM_REVISION = "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"


def plain(markdown):
    """The prose a reader reads: no links, headings, figure lines or source lists."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
    text = "\n".join(line for line in text.splitlines()
                     if not line.startswith(("###", "**Key figures**", "**Sources**", "- ", "**Source**", "**Related**")))
    return " ".join(text.replace("**", "").split())


def grade(text):
    """Flesch-Kincaid grade level, with syllables counted by vowel groups."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    sentences = max(1, len(re.findall(r"[.!?]+(\s|$)", text)))
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower())) - (w.lower().endswith("e") and len(w) > 3)) for w in words)
    return round(0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59, 1)


def agent_kind(turn):
    if not turn.spoke:
        return "refuse"
    if turn.intent == "reference lesson":
        return "lesson"
    if turn.intent == "reference recent":
        return "recent"
    if turn.intent.startswith("reference"):
        return "answer"
    return turn.intent


def agent_ok(expected, kind):
    return expected == "any" or expected == kind or (expected == "answer" and kind != "refuse")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-hhem", action="store_true")
    args = ap.parse_args()
    from backend.cli import DEFAULT_DB, _orchestrator
    from backend.models.agent.present import present
    from backend.models.agent.run.ask import build, listed_names, reference_from, writer_from
    asked = json.loads(QUESTIONS.read_text())
    writer = writer_from(True)
    reference = reference_from(ARTIFACTS, "reference_index.npz", "generator.npz", False, True, writer, dense=True)
    agent, *_ = build(ARTIFACTS, "", str(DATA / "prices"), "", str(ARTIFACTS / "chat.sqlite"), 0.6, 10,
                      str(ARTIFACTS / "price_head.npz"), reference, str(DATA / "raw/sec_edgar/company_tickers.json"),
                      str(ARTIFACTS / "returns.npz"), writer=writer, listed=listed_names(), router="pretrained")
    names, sources = listed_names(), json.loads((ARTIFACTS / "reference_sources.json").read_text())
    orch = _orchestrator(Namespace(db=DEFAULT_DB, demo=False))
    type(orch.hub).writer = lambda self: writer
    rows = []
    for item in asked:
        started = time.time()
        turn = agent.answer(item["q"])
        agent_seconds = time.time() - started
        shown = present(turn, names, sources)
        started = time.time()
        reply = orch.handle(item["q"], f"eval-{len(rows)}")
        kind = agent_kind(turn)
        row = {**item, "agent_kind": kind, "agent_ok": agent_ok(item["agent"], kind), "agent_shown": shown,
               "agent_grade": grade(plain(shown)) if turn.spoke else None, "agent_seconds": round(agent_seconds, 1),
               "phrased_by": turn.phrased_by, "unphrased_because": turn.unphrased_because,
               "premise": f"{turn.answer} {turn.evidence}" if turn.phrased_by else "",
               "app_intent": reply.intent, "app_ok": item["app"] in ("any", reply.intent),
               "app_text": reply.text, "app_grade": grade(plain(reply.text)), "app_seconds": round(time.time() - started, 1)}
        rows.append(row)
        print(f"{'ok ' if row['agent_ok'] else 'BAD'} agent {kind:12} | {'ok ' if row['app_ok'] else 'BAD'} app "
              f"{reply.intent:18} | {item['q']}", flush=True)
    if not args.skip_hhem:
        from transformers import AutoModelForSequenceClassification
        judge = AutoModelForSequenceClassification.from_pretrained(HHEM, revision=HHEM_REVISION, trust_remote_code=True).eval()
        worded = [r for r in rows if r["phrased_by"]]
        scores = judge.predict([(r["premise"], plain(r["agent_shown"])) for r in worded]) if worded else []
        for r, score in zip(worded, scores):
            r["hhem"] = round(float(score), 3)
    summary = summarise(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


def summarise(rows):
    def share(flags):
        flags = list(flags)
        return f"{sum(flags)}/{len(flags)}" if flags else "0/0"
    scored = [r for r in rows if "gap" not in r]
    worded = [r for r in rows if "hhem" in r]
    grades = [r["agent_grade"] for r in rows if r["agent_grade"] is not None] + [r["app_grade"] for r in rows]
    return {"agent right kind of turn": share(r["agent_ok"] for r in scored),
            "app right handler": share(r["app_ok"] for r in scored),
            "agent answered (spoke)": share(r["agent_kind"] != "refuse" for r in scored if r["agent"] not in ("refuse",)),
            "LM-worded answers": len(worded),
            "hallucination rate (HHEM < 0.5)": share(r["hhem"] < 0.5 for r in worded),
            "median reading grade": sorted(grades)[len(grades) // 2] if grades else None,
            "answers above grade 9": share(g > 9 for g in grades),
            "gaps (not yet supported)": [r["q"] for r in rows if "gap" in r]}


if __name__ == "__main__":
    main()
