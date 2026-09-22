# Build queue

One task at a time, in order, each with a gate that says whether it is actually done. This file is the
state of the build: it survives a lost session, so read it first and trust it over memory.

## Loop protocol

Every pass does the same five things. Do not start a task while an earlier one is still `open`.

1. Read this file. Take the first task whose status is `open`. A task marked `waiting` is blocked on a
   run finishing rather than on anything to decide, so step over it and come back — but never step over
   an `open` one.
2. Do it. Write the test that would catch the failure the task names.
3. Run the gate. A gate is a number or a passing test, never an opinion.
4. Update the row: status `done`, and paste the measured number next to it. If the gate failed, leave it
   `open` and write what the number actually was — a failed gate is a result, not a reason to move on.
5. Commit locally only. Nothing is pushed until stage D.

**Re-entry prompt**, for a new session or after the context is lost:

> Read `docs/BUILD_QUEUE.md`. Take the first `open` task. Do it, run its gate, record the measured
> number in the row, commit locally. Then take the next. Do not push to GitHub until stage D.

## Rules that hold for every task

- **Free and local only.** No paid service, no API key, nothing that phones home. SQLite over Postgres,
  an in-process index over a hosted vector DB, RSS over a news vendor.
- **No look-ahead.** Any feature read at time *t* uses only bars up to *t*. This is the one bug that
  makes every number meaningless, and it is invisible in a loss curve.
- **The agent is the teacher.** There is no human analyst, so preference labels come from Claude in
  session plus a programmatic oracle where ground truth exists. Log which one labelled each row; a
  result that only holds under one labeller is not a result.
- **Frozen foundation weights.** The thesis is improvement *without* retraining the base model. Every
  adaptation in stage B changes a threshold, a weight, or a ranking — not the transformer. LoRA is a
  last resort and needs its own control.
- **Every learning curve needs a flat control** beside it, same data, adaptation switched off. A curve
  that goes up on its own proves nothing.

---

## Stage A — finish the measurement already running

| id | task | gate | status |
|---|---|---|---|
| A1 | Chat-register arm completes; measure the `casual` split on its best and final checkpoints | casual-split loss and exact match recorded next to `unseen` | waiting — step 23,200/28,163, ~0.3h left |
| A2 | Combined arm: `--freeze-encoder` on the casual dataset | unseen loss beats 0.1717, or say plainly that the two fixes do not add | waiting — `advisory_combined` launched, ~1.6h |
| A3 | Write A1/A2 into `PLAN_OF_ACTION.md` and `learnme.md` | both files quote the same numbers as this file | open |

Already measured, for reference: frozen encoder cut unseen loss 0.2821 → **0.1717** and unsupported
figures on unseen 4.7% → **0.7%**, closing S14 on both splits for the first time. Selecting on loss
picked a checkpoint that generates worse (81.0% vs 94.5% exact match), so stage B selects on generation.

## Stage B — interactive training, the thesis

| id | task | gate | status |
|---|---|---|---|
| B1 | `selfagent/learn/store.py`: append-only feedback log — question, evidence, answer, label, labeller, timestamp. SQLite. | a round trip test, and the log replays into the same training set twice | **done** — 8 tests in `tests/test_learn.py` |
| B2 | `selfagent/learn/teacher.py`: two labellers behind one interface — programmatic oracle (exact match, guardrail, intent) and agent-as-teacher (Claude, writing judgments to disk for replay) | 200 rows labelled both ways; agreement rate reported, not assumed | **done** — agreement **156/198 (78.8%)** |
| B3 | `selfagent/learn/calibrate.py`: map self-confidence to the probability the answer is right, fit on the feedback log | calibration error beats the raw confidence's on a held-out slice | open |
| B4 | `selfagent/learn/abstain.py`: learned threshold replacing the hand-picked 0.9980 | beats 53.8% correct @ 40% coverage on `unseen`, threshold fit on other rows | open |
| B5 | `selfagent/learn/rank.py`: generate k candidates, rank with frozen weights using the calibrator | exact match on `unseen` beats single-sample 31.0% | open |
| B6 | `scripts/learning_curve.py`: accuracy against number of feedback rows, adaptation on vs off | two curves; the gap is the thesis, and if there is no gap say so | open |
| B7 | `scripts/chat.py`: one conversation turn at a time — ask, answer, label, adapt, show what moved | a scripted session of 20 turns runs end to end and the store grows by 20 | open |

**B2 measured**, on 200 `unseen` rows from `advisory_frozen.npz` (198 distinct question/answer pairs;
the sample drew two pairs twice). Oracle **62/200 right (31.0%)**, matching `check_answers.py`'s unseen
exact match exactly, which is the consistency check that the labeller and the scorer are one rule. Agent
**103/198 (52.0%)**. Agreement **156/198 (78.8%)**, and the 42 disagreements all run one way: the oracle
said *differs from the gold answer* and the agent said right. **Zero** rows where the oracle said right
and the agent said wrong — so blind judging never rejected an answer the oracle could verify, and the
whole gap is exact match refusing a correct answer in different words. Exact match understates this
model by 21 points, so B5's gate should be read against the agent's number too, not only 31.0%.

What the agent's 95 wrong rows are, which sets stage B's targets: **intent misrouting dominates** —
a buy-advice template for overbought or outlook questions (14), performance for an outlook or volatility
question (13), drawdown for an outlook question (6). Then **false refusals** (30): refusing a question
the evidence plainly answers. Then two the guardrail cannot see: four **wrong comparative claims**
("above both its 20 and 50 day averages" when the price is below one) where every digit is supported,
and four **digit-duplication inventions** (`242804.00` for a close of `24280.00`) which it does catch.
A figure guard is not an intent guard, and C1's router is where the first class gets fixed.

## Stage C — the architecture in the diagram

Source of truth for the shape: `docs/architecture.png`. Reuse from `Financial-Analysis-Upgrade`:
`utils/indicators.py` + `feature_columns.json`, `preprocessing/ticker_mapping.py`, the `main.py` FastAPI
shape, the `complete_pipeline.py` fan-out.

### Layout

One package per concern, and dependencies point one way: `backend/` imports `selfagent/`, never the
reverse. `selfagent/` stays the research artifact it is — a backend that grew a copy of the model would
give the thesis two sets of weights to defend — and `dataforge/` stays deletable.

```
backend/
  main.py          FastAPI app and routes only, no logic                       C7
  agent/           intent router -> context assembler -> generator -> guards   C1
  retrieval/       chunker, embedder, local index, the as-of filter            C2
  models/          registry, version-pinned loaders, the eval gate             C3, C6
  news/            RSS pull, dedupe, ticker tag, sentiment                     C4
  database/        SQLite schema, migrations, one repository per table         C5
  logging/         structured logs, request ids, the error handlers
frontend/          adapted from Financial-Analysis-Upgrade                     C8
selfagent/         unchanged: the model, the tokenizer, the learning loop
dataforge/         unchanged, and still deletable on its own
```

| id | task | gate | status |
|---|---|---|---|
| C1 | Agent core: intent router → context assembler → generator → guardrails, domain agnostic | one call answers a free-text question end to end, no network | open |
| C2 | Retriever: hybrid lexical + vector, as-of filtered, local index; chunk and embed worker | recall@5 beats the lexical baseline, or the vector half is dropped and that is recorded | open |
| C3 | Price advisory endpoint: version-pinned GRU + point-in-time feature builder | a served feature vector matches one rebuilt from bars up to that date, exactly | open |
| C4 | News ETL (RSS, dedupe, ticker tag) + sentiment scorer, version pinned | a scored article traces to its source URL and licence in the manifest | open |
| C5 | Portfolio and memory service: SQLite for users, conversations, holdings, trades, feedback; session buffer. The reference repo already has login/auth and a portfolio UI, so the schema follows what that UI asks for rather than being invented here | a conversation survives a restart, and a logged-in user sees only their own | open |
| C6 | Model registry + eval gate + promote: nothing ships that fails S14 | a deliberately bad model is refused promotion by the gate | open |
| C7 | FastAPI REST + WebSocket, streaming a turn | the chat loop from B7 runs over the socket | open |
| C8 | Frontend from the chosen repo, wired to C7 | a question typed in the browser returns a grounded answer with its evidence shown | open |

## Stage D — publish

| id | task | gate | status |
|---|---|---|---|
| D1 | Full suite green, every gate above recorded | `pytest tests/ dataforge/ -q` passes and no row is `open` | open |
| D2 | One commit series pushed to GitHub | remote matches local | open |

## Reference repo, settled

Frontend design and model input shapes both come from
[Financial-Analysis-Upgrade](https://github.com/Kapurrrishabh/Financial-Analysis-Upgrade) — its
`frontend/` for the design and `backend/` for the input shapes. Cloned read-only to
`reference/financial-analysis-upgrade/`, which is git-ignored: it is a thing to read, not a dependency,
and `selfagent/` must not import from it.

It already carries login, auth and a portfolio UI, which is why it is worth adapting rather than
starting over — those are the parts nobody enjoys rebuilding. The design is ours to change to fit a
chat-first agent; the diagram's entry point is a conversation, not a dashboard.
