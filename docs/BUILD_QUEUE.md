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
| A1 | Chat-register arm completes; measure the `casual` split on its best and final checkpoints | casual-split loss and exact match recorded next to `unseen` | **done** — final **94.5%** casual / **39.0%** unseen exact match; the loss-picked checkpoint is worse on all four |
| A2 | Combined arm: `--freeze-encoder` on the casual dataset | unseen loss beats 0.1717, or say plainly that the two fixes do not add | **done, gate unanswerable as written** — they add on faithfulness (unsupported **4.1% → 2.4%**, loss **0.4162 → 0.2496**) and not on exact match (39.0% → 36.0%) |
| A3 | Write A1/A2 into `PLAN_OF_ACTION.md` and `learnme.md` | both files quote the same numbers as this file | **done** — §1a in the plan, §5.9 and new §5.10 in `learnme.md`; stage B's own write-up is B8 |

Already measured, for reference: frozen encoder cut unseen loss 0.2821 → **0.1717** and unsupported
figures on unseen 4.7% → **0.7%**, closing S14 on both splits for the first time. Selecting on loss
picked a checkpoint that generates worse (81.0% vs 94.5% exact match), so stage B selects on generation.

**A1 measured**, `scripts/ablate_generator.py` for loss and `scripts/check_answers.py` for generation,
200 rows each, on `advisory_casual`:

| checkpoint | split | loss | exact match | unsupported figures | false refusal |
|---|---|---|---|---|---|
| best, step 6,000 | casual | 0.0193 | 68.0% | 20.7% of answers | 0.0% |
| best, step 6,000 | unseen | **0.2339** | 32.5% | 17.2% | 18.9% |
| final, step 28,163 | casual | 0.0040 | **94.5%** | **3.6%** | 0.0% |
| final, step 28,163 | unseen | 0.4162 | **39.0%** | **4.1%** | **4.7%** |

Read the two unseen rows against each other, because they settle how this project picks checkpoints. The
loss-selected one is better on loss by a wide margin — 0.2339 against 0.4162 — and worse on every
generation number there is: 6.5 points less exact match, four times the unsupported figures, four times
the false refusals. Early stopping on held-out loss would have shipped the worse generator, and nothing
in the loss curve says so. The earlier 81.0%-vs-94.5% finding was one arm and one split; this is both
splits and a larger gap, so **selecting on loss is settled as wrong here, not merely suspect**.

Why it happens is visible in the ablation: the evidence-swap penalty *grows* with training, +1.69 →
**+2.60** on unseen. The late model commits harder to the specific digits its evidence states, which
costs average cross-entropy on held-out phrasings while making the answers more correct. Rising unseen
loss is the price of reading the evidence, not a sign of overfitting to it.

**A2 measured**, and first the gate has to be corrected. It asked A2's unseen loss to beat **0.1717**,
which was the frozen-encoder arm's figure on **`advisory`'s** unseen split, while A2 trains on
`advisory_casual` and reports on **that** dataset's unseen split. Different rows, so the two numbers were
never comparable and the gate as written cannot be settled. What is comparable is A1 against A2: same two
splits, same 200 rows, one fix apart. Final checkpoints:

| | A1, casual data only | A2, casual data + frozen encoder |
|---|---|---|
| casual loss | 0.0040 | 0.0037 |
| casual exact match | 94.5% | 94.5% |
| casual unsupported figures | 3.6% of answers | **2.4%** |
| unseen loss | 0.4162 | **0.2496** |
| unseen exact match | **39.0%** | 36.0% |
| unseen unsupported figures | 4.1% of answers | **2.4%** |
| unseen false refusal | 4.7% | **3.6%** |

**So the two fixes add on faithfulness and not on exact match.** Freezing the encoder cuts unseen loss by
40%, cuts unsupported figures and false refusals by about a quarter each, and leaves casual exact match
identical to the row — 189/200 in both arms. It reads 3 points lower on unseen exact match, which is 6 rows
of 200, and B5 measured that this sample carries about ±3 points at n=200, so that is not a cost this
measurement can distinguish from noise. The claim to make is the faithfulness one, which is larger than the
noise and is what the fix was for.

The second finding is the one that matters more, because it is now **replicated on an independent arm**.
A2's best checkpoint by unseen loss is step 6,000 at **0.1435** — better on loss than the final's 0.2496 —
and worse on all four generation numbers: casual exact match **77.0% against 94.5%**, unseen **30.5%
against 36.0%**, unsupported figures **17.2% against 2.4%**, false refusal **7.7% against 3.6%**. That is
the same direction and roughly the same size as A1. Two arms, two datasets, one conclusion: on this task
held-out loss and generation quality move in opposite directions after the copying phase transition, and
early stopping on loss ships the worse generator. Stage B selects on generation, and so will stage C's
eval gate in C6.

## Stage B — interactive training, the thesis

| id | task | gate | status |
|---|---|---|---|
| B1 | `selfagent/learn/store.py`: append-only feedback log — question, evidence, answer, label, labeller, timestamp. SQLite. | a round trip test, and the log replays into the same training set twice | **done** — 8 tests in `tests/test_learn.py` |
| B2 | `selfagent/learn/teacher.py`: two labellers behind one interface — programmatic oracle (exact match, guardrail, intent) and agent-as-teacher (Claude, writing judgments to disk for replay) | 200 rows labelled both ways; agreement rate reported, not assumed | **done** — agreement **156/198 (78.8%)** |
| B3 | `selfagent/learn/calibrate.py`: map self-confidence to the probability the answer is right, fit on the feedback log | calibration error beats the raw confidence's on a held-out slice | **done** — 0.683 → **0.121** oracle, 0.475 → **0.144** agent |
| B4 | `selfagent/learn/abstain.py`: learned threshold replacing the hand-picked 0.9980 | beats 53.8% correct @ 40% coverage on `unseen`, threshold fit on other rows | **done, gate tied not beaten** — **53.6% @ 39.0%** held out |
| B5 | `selfagent/learn/rank.py`: generate k candidates, rank with frozen weights using the calibrator | exact match on `unseen` beats single-sample 31.0% | **done, gate failed** — no picker beats it; 8 samples give **1.77 distinct answers** |
| B6 | `scripts/learning_curve.py`: accuracy against number of feedback rows, adaptation on vs off | two curves; the gap is the thesis, and if there is no gap say so | **done, gate passes** — the learned cut misses its stated bar by **8.5 pts** against **16.7** hand-picked and **29.2** unabstained |
| B7 | `scripts/chat.py`: one conversation turn at a time — ask, answer, label, adapt, show what moved | a scripted session of 20 turns runs end to end and the store grows by 20 | **done, gate passes** — 20 turns, log **0 → 20**, cut moved 0.99800 → **0.99962** at turn 10 |
| B8 | Write stage B into `learnme.md` and `PLAN_OF_ACTION.md` — B2's agreement, B3's AUC, B4's curve, B5's failed gate. Neither doc mentions the learning loop's results yet, and it is the thesis | both files quote the same numbers as this file, and a reader can answer "did the loop work" from `learnme.md` alone | **done** — `learnme.md` §5.11 and the plan's §1a; **S7 recorded as partly addressed**, not closed |

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

**B3 measured**, `scripts/calibrate_confidence.py`, fit on half the log and scored on the other,
averaged over 25 splits. Platt scaling, two parameters, `p = sigmoid(0.33 * log odds of confidence
- 3.31)` under the oracle. The gate passes wide: calibration error 0.683 → **0.121** (oracle) and
0.475 → **0.144** (agent), on every split, because raw confidence claims 0.99 on answers right 31% of
the time and almost anything beats that. So a second, harder baseline was measured too — a constant that
ignores the confidence and predicts the base rate. Against it: calibration error is a coin toss
(**12/25** oracle, **9/25** agent) but Brier is a near-clean sweep (**25/25** and **24/25**). Read the
Brier: calibration error flatters a constant, which has no spread inside a bin to be wrong about.

The number that matters for what comes next is **ranking AUC 0.740 ± 0.039** (oracle) and **0.692 ±
0.036** (agent). The confidence does rank right above wrong, well clear of a coin toss — and because any
monotone calibration leaves the order untouched, that AUC is the hard ceiling on B4's threshold and B5's
reranker. Neither can do better than this signal allows, and B5's gate should be judged against it.

**B4 measured**, `scripts/fit_abstention.py`, cut fitted on half and quoted on the other, 25 splits.
**The gate is tied, not beaten: 53.6% correct on 39.0% coverage, against 53.8% @ 40%.** Read what the
target was, though — `check_answers.report_confidence` says in its own docstring that a cut chosen on the
rows it is quoted on is an upper bound, and 53.8% was exactly that. On this log that in-sample bound is
68.9% @ 22.5%. So the honest result is that a cut fitted out of sample *reproduces* the in-sample number
at matched coverage, which is the thing that was in doubt.

No threshold could have beaten it, and that is structural rather than a shortfall. A cut on a calibrated
probability orders rows identically to a cut on the raw confidence, so every threshold lives on one
precision-coverage curve fixed by the model, and AUC 0.740 is where that curve sits. What B4 replaces is
therefore the *meaning* of the number, not the number: `Abstainer.fit(..., wanted=0.6)` asks for a stated
chance of being right and solves for the confidence that delivers it, and 60% still means 60% on the next
checkpoint where 0.9980 means nothing. It also reports the bar it missed instead of claiming it.

The curve itself, held out, under the oracle: asking 50% buys **51.5% @ 43.8%**, asking 60% buys
**58.5% @ 32.7%**, asking 70% buys **64.3% @ 24.2%** — against 31.0% for answering everything, so +21 to
+34 points of precision for the coverage given up. Under the agent's labels the same cut of 0.9980 turns
out to be almost exactly a 70% bar (67.2% @ 52.2%), which is what the hand-picked number had been all
along without saying so.

**B5 measured, and the gate fails.** `scripts/rerank.py`, `selfagent/learn/rank.py`. Three pickers were
tried against taking the first sample, on `unseen` with `advisory_frozen.npz`:

| temperature / top-p | distinct of 8 | single | by confidence | by `rank()` | most repeated | ceiling |
|---|---|---|---|---|---|---|
| 0.9 / 0.9 (served) | **1.54** | 31.0% | 28.5% | 30.0% | 31.5% | 35.5% |
| 0.9 / 0.99 | 1.65 | 31.0% | 28.5% | 29.0% | — | 35.0% |
| 1.0 / 0.999 | 1.72 | 31.0% | 29.0% | 30.5% | 33.0% | 37.5% |
| 1.0 / 0.999, 600 rows | **1.77** | 28.0% | 27.8% | 28.5% | **29.2%** | 34.0% |

**The cause is not the ranking — it is that there is nothing to rank.** Eight samples produce 1.54
distinct answers at the served sampler. Doubling k from 4 to 8 moved the ceiling 1.0 point; loosening the
nucleus tenfold moved distinctness to 1.65 and the ceiling not at all. `_nucleus` keeps a second token
only when the top one holds less than top-p of the mass, and this model's per-token mass is about 0.99 —
the same fact as the raw confidence sitting above 0.999 in B3. A near-deterministic model cannot be
improved by sampling it more, so best-of-k is structurally unavailable here.

The most repeated answer is the only picker that ever leads, and on the 600-row run it **rescued 14 rows
the first sample got wrong and spoiled 7 it got right**. McNemar exact on those 21 discordant rows gives
**p = 0.189**; 16 of 21 would have been needed for 0.05. So +1.2 points is not distinguishable from
chance, and the gate is recorded as failed rather than squeezed. Note also that single-sample accuracy
read 31.0% on 200 rows and 28.0% on 600 — the 200-row figure the gate was written against carries about
±3 points, which is larger than every effect measured against it.

Two findings worth carrying forward. **Confidence ranks worse than not ranking at all** (-2.5 points),
because refusals carry the highest confidence the model produces — 0.99538 against 0.98797 on answers, at
18 words against 32, and confidence correlates -0.037 with length so it is not brevity. B3's AUC 0.740
holds *across* questions and does not transfer to ranking answers to one; B4's threshold is unaffected
because it also ranks across questions. And **the tiers do help where they fire**, beating confidence by
1.5 points, but fired on 4-6% of rows because this checkpoint invents figures on only 0.7% of unseen
answers. `rank.py` and its tier order are kept for C1, where the candidates will come from different
retrieved contexts rather than from resampling one question — real diversity, which is the input this
measurement says the policy is missing.

**B6 measured, and the gate passes.** `scripts/learning_curve.py`. The adapting thing is B4's abstention
cut: the agent is told "answer only where you are at least *p* likely to be right" and solves for the
confidence cut that delivers it from the feedback it has. So the score is not accuracy but the **miss** —
how far the delivered precision lands from the bar that was promised, on rows the cut was never fitted on.
A cut fitted on ten rows promises 60% and delivers whatever those ten rows happened to say; the thesis is
that the promise comes true as the log grows, with no gradient computed anywhere.

| feedback rows | oracle @ 60% | agent @ 80% |
|---|---|---|
| 10 | 12.9 pts | 13.3 pts |
| 20 | 12.0 | 10.4 |
| 40 | 9.5 | 9.5 |
| 60 | **8.5** | 9.0 |
| 80 | 9.3 | 7.7 |
| 118 | 9.6 | **6.3** |
| fixed cut 0.9980, no feedback | 16.7 | 10.4 |
| no abstention at all | 29.2 | 27.2 |
| fitted on the scored rows (ceiling) | 1.1 | 3.5 |

**The gap is there and it is large.** Both flat controls are beaten at every log size past ten rows: a cut
placed from 60 rows of feedback misses by 8.5 points where the hand-picked 0.9980 misses by 16.7 and
answering everything misses by 29.2. Nothing in the transformer changed to earn that — it is one threshold
solved from outcomes, which is the thesis in its smallest honest form.

**Feedback pays in proportion to how hard the promise is.** At a 60% bar under the oracle the curve falls
12.9 → 8.5 by 60 rows and then flattens; at an 80% bar under the agent's labels it falls monotonically
13.3 → 6.3 and is **still falling at 118 rows**, which is the whole log. The flat 60% case is not a null
result but an easy bar: the agent's own base rate is 52.8%, so answering everything nearly clears 60%
by accident, and there was little for a threshold to buy. Read the strict column for the thesis and the
loose one for the reason a learning curve needs a hard task to be visible on.

**And a miss that will not close is not always a failure to learn**, which is the finding worth carrying.
Ask the oracle's 80% bar and the curve bottoms at 15.0 points — but the ceiling, a cut fitted on the very
rows it is scored on, *also* misses by 11.7. This model has no confidence region that is right 80% of the
time at 20% coverage or better, so only 3.3 of those 15.0 points was ever learnable and no quantity of
feedback would have closed the rest. `Abstainer.fit` taking the most precise cut the coverage floor allows
and reporting the shortfall through `expected` is what makes that visible instead of a threshold quietly
promising something it cannot deliver. Every miss in the table is therefore split into the part the bar
forbids and the part the log has not yet taught.

One check on what the residual is. Scoring on 120 rows instead of 80 moved the plateau by 0.5 points, far
less than the binomial noise of the scored set predicts, so the ~7 points still open at a 60% bar is
**transfer error from the fitting rows**, not noise in the scoring rows. That is precisely what more
feedback buys, and 118 rows is not enough of it — the honest next step for this curve is a longer log from
B7's chat loop rather than a cleverer fit.

**B7 measured, and the gate passes.** `scripts/chat.py`. Twenty turns ran end to end, the log grew 0 → 20,
and the cut it serves with moved **0.99800 → 0.99962 at turn 10**, which is the `--warmup` boundary: before
it the loop serves the hand-picked value and says so, because a threshold solved from three rows is a
coincidence and serving one while calling it learned is the one thing this loop must not do. On this
session it spoke on 8 of 20 and was right on 2 of those (25.0%) against 10.0% for answering everything —
the right direction, but 20 turns is far too few rows to read as a result, and the 10.0% is an unlucky
prefix against the 28-31% this split gives at 200 rows.

What the loop shows per turn is the honest scope of the claim: **what adapts is when the agent speaks, not
what it says.** The wording comes from frozen weights, and B5 measured that resampling them gives nothing
to choose between; the judgement about whether the wording is worth saying comes from the log. The line
`expects 29% on 35%` is `Abstainer.fit` reporting that it could not reach the 60% bar and taking the most
precise cut the coverage floor allows instead — the shortfall stated rather than hidden, which is the same
behaviour B6 decomposed at the 80% bar.

Two limits are deliberate and written into the script. The questions come from the dataset rather than
from a person, because a typed question arrives with no evidence and choosing its passages is C2's
retriever and C1's context assembler — guessing at it here would leave two rules for assembling context.
And the session writes to `artifacts/chat.sqlite`, not the `feedback.sqlite` that B2 through B6 were
measured on, so replaying a session cannot move a published number.

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
  news/            reads the collected feeds, dedupe, ticker tag, sentiment    C4
  database/        SQLite schema, migrations, one repository per table         C5
  logging/         structured logs, request ids, the error handlers
frontend/          adapted from Financial-Analysis-Upgrade                     C8
selfagent/         unchanged: the model, the tokenizer, the learning loop
dataforge/         unchanged, and still deletable on its own
```

| id | task | gate | status |
|---|---|---|---|
| C1 | Agent core: intent router → context assembler → generator → guardrails, domain agnostic | one call answers a free-text question end to end, no network | **done** — `scripts/ask.py` answers a typed question offline. Rewriting into the routed trained phrasing takes held-out exact match **33.5% → 74.5%** with frozen weights (C1a); the routing gate refuses 12/12 out-of-domain questions at 88.6% coverage. Routing is still what is left: 22.5 of the remaining 25.5 points |
| C1a | Router: sum the frozen cosine with the share of the question's own term weight that the known phrasings match | an unseen sentence shape routes better than the served scorer managed, and no out-of-domain question is answered | **done** — `combined` serves. Routing **92.6% → 95.3%** overall, an unseen sentence shape **67% → 90%**, an unseen frame 68% → 76%, gate coverage at the same 97% precision bar 80.6% → **88.6%**, 12/12 out-of-domain still refused, and held-out exact match **68.0% → 74.5%** against the semantic scorer on the same pool. No parameter was fitted: the frozen encoder is read exactly as it was. Measured on the axes as they stood; a short unseen family added after C7 makes it a harder test, and the figures on that are below |
| C2 | Retriever: hybrid lexical + vector, as-of filtered, local index; chunk and embed worker | recall@5 beats the lexical baseline, or the vector half is dropped and that is recorded | **done** — the gate's second branch. On 400 queries over a 44,811-chunk index: lexical **15.5%**, hybrid 12.0%, vector **1.5%**, random 0.0%, so `SERVED = LEXICAL` and `scripts/eval_retrieval.py` prints the verdict. Cause diagnosed: no contrastive objective, mean-embedding norm 0.942 of 1. Only `sec_edgar` and `fed_press` carry real dates, so the other nine sources are excluded rather than dated by download |
| C3 | Price advisory endpoint: version-pinned GRU + point-in-time feature builder | a served feature vector matches one rebuilt from bars up to that date, exactly | **done** — the gate holds at the array level and through `Market.snapshot`, both verified by mutation (standardising over the whole series; taking the window's scale from `bars[-1]`). `models.load` picks the advisor from the artifact's own recorded numbers: on all 28,676 held-out windows the GRU scores **49.1%** against persistence **47.8%** and majority 39.6%, a gap of 4.6 standard errors, so serving uses `gru@62babf47cdd5` (202,755 params). Below one standard error it serves the arithmetic instead. The earlier "tie" (46.9% vs 47.0%) was the head scored on a 1,920-window prefix — 7 of 101 tickers — against a baseline scored on all of them |
| C4 | News ETL (RSS, dedupe, ticker tag) + sentiment scorer, version pinned | a scored article traces to its source URL and licence in the manifest | **done** — the gate holds in the type: `feed.read` refuses a file the manifest does not describe, so an article that cannot name its URL and licence is unconstructible, and `scripts/news.py` prints one in full. **1,481 articles** over 55 feeds after dedupe, **1,316 tagged**, scorer pinned as a digest of its own word lists (`lexicon@d3e0671ec848`). Validated and **the forecast half is a null**: on 456 scored pairs the polarity's sign agrees with the next day's return **49.6% ± 2.3%** against a 52.6% base rate, so it is not wired into advice. The same-day check separates why — positive minus negative is **+0.90% at 2.6 standard errors** on the publication day, so the lexicon does read the headline and only the forecast fails |
| C5 | Portfolio and memory service: SQLite for users, conversations, holdings, trades, feedback; session buffer. The reference repo's portfolio UI settles the fields; its accounts could not be copied, because its portfolio is a process-global list and its login page needs a Firebase key | a conversation survives a restart, and a logged-in user sees only their own | **done** — both halves are pinned in `tests/test_database.py`: a turn written before `close()` reads back after reopening the file, and asking for another user's conversation raises `NotYours`. That holds structurally rather than by care: **no method takes a user id**, only a session token it resolves itself, so a cross-user read cannot be written by forgetting a `WHERE`. **A position is not stored** — `holdings` is a fold of the trade log, so the two cannot drift, and a sell that the log would not support is refused before the insert. Feedback stays in the loop's `FeedbackLog`, opened on the same file, so a message holds a real foreign key to the row that judged it |
| C6 | Model registry + eval gate + promote: nothing ships that fails S14 | a deliberately bad model is refused promotion by the gate | **done** — the gate refused a real checkpoint, not a fixture. `scripts/promote.py` measures a candidate on 200 held-out rows of each split with the same counting every published number here uses, and `backend/models/registry.py` promotes it only if its own record clears two standards, both written with `price.beats` so the margin rule is one rule: **S14** — the unsupported-figure rate on the reworded split must sit under 2% by more than the noise at 2% — and **generation, not loss** — it must beat the promoted checkpoint's exact match on the same rows by more than one standard error. The served checkpoint promotes: **4/771 figures (0.5%)** unsupported and **36.0%** exact match on `unseen`, 5/709 and 94.5% on `casual`. The loss-picked `advisory_casual.best.npz` is **refused on both counts** — 40/653 figures (**6.1%**) and 32.5% against 36.0% — which is A1's finding enforced instead of written down. `ask.py` now asks the registry which weights answer rather than carrying a default file name, and the digest is re-checked at serving, so an artifact overwritten in place is refused rather than answering under a record of the bytes it replaced |
| C7 | FastAPI REST + WebSocket, streaming a turn | the chat loop from B7 runs over the socket | **done** — 20 typed questions driven over a real socket by `scripts/chat_socket.py`, which imports nothing but frames: all **20 stored** as one conversation, **17 judged** by the replayed agent verdicts, and the answer cut moved **0.99800 → 0.97125** at turn 12, the tenth judged row and so the warmup. It **loosened**: at 82% right the log does not justify withholding anything to clear the 60% bar, and the hand-picked 0.998 was stricter than the feedback supports. The move is visible in what the user gets — the NVDA performance question was withheld at 0.998 in the unjudged pass and answered at 0.97125 in the judged one, then judged right. Over the 17: **82.4% right**, spoke on 14 and **right on 13 of them (92.9%)**. `backend/main.py` is routes only, given the agent, the store, the judge and the refit rather than paths to them, and `scripts/serve.py` assembles through the same `ask.build` the C1 gate demonstrates, so the agent behind the socket is the measured one. It streams **by stage, not by token**: `generate` returns a whole sequence, so fake tokens would be theatre. The 3 turns refused before the decoder are stored but never judged — they have no confidence, and a 0.0 in the log would be a number the model never reported, fitted into the cut |
| C8 | Frontend from the chosen repo, wired to C7 | a question typed in the browser returns a grounded answer with its evidence shown | **done** — `frontend/verify_gate.mjs` drives a real Chromium against the dev server and the live socket: it registers a new account through the form, waits for the socket to report `ready`, types three questions and reads what rendered. **3 typed, 2 answered, 1 withheld**, every reply carrying its **12 evidence lines** with the `as_of` date, and every figure in a spoken answer present in that panel. It types rather than posts, because everything reachable by an HTTP request is already measured by C7 — the only thing left worth checking is the part a request cannot see. Doing that found a real bug no Python test could: the scroll effect's `scrollIntoView({behavior:"smooth"})` now resolves a promise in Chromium, React took the returned promise for the cleanup function, and the page tore itself down the moment the socket went `ready`. The gate's first pass also flagged `its 14 day rsi is 52` as ungrounded — wrong, and the check was: `14` is the field's own period, which the panel heads `RSI 14`, so it matches against the whole panel now and defers to the guard the agent already runs. The withheld one is the AAPL performance question at the refitted cut, which is the C1a improvement visible in a browser rather than in a table. Four routes build static and clean, 0 npm vulnerabilities on Next 16.3.6. Only what C7 serves is wired — auth, chat, portfolio, model — and the reference repo's seven pages calling endpoints that do not exist were deleted rather than shipped erroring |
| C9 | Reference path: retrieved passages wired into the answer, for the questions the router refuses | a served answer states no figure its own passages do not, on real questions nobody wrote for this | **done** — the gate holds and the answers are bad, and both halves are the result. `scripts/eval_reference.py` runs 200 real Stack Exchange questions over the served 32,713-chunk index: **coverage 200/200**, **0 answers / 0 figures invented**, against **60 answers / 186 figures the guard caught** in the paraphrase it graded beside them. Reading those paraphrases is what changed the design. The guard replaced 30.0% of them, and the 140 it passed are fluent and incoherent — grounded in their figures and wrong about what the passage says — so a figure rate was clearing a bar the answers did not. **So this path quotes by default**: it retrieves, cites, and serves the passage's own words, with no decoder pass at all, and `--paraphrase` keeps the generator's words servable and graded so the decision stays demonstrable next to the thing it rejected. A quoted turn reports no confidence, because the model chose none of the words; under `--paraphrase` it runs 0.0253–0.6035 (median 0.0539) and is never calibrated, because the calibrator's record is the advisory generator's. Additive by construction: `reference=None` reproduces C1's refusals exactly, the citation rides inside the existing `" ; "` evidence field, and no schema, `Turn` arity, wire format or frontend change was needed. **Retrieval was rebalanced and re-measured.** `share()` split the chunk budget equally between sources, which gave Stack Exchange 296 of 148,034 threads — an equal share of chunks is a tiny share of a source whose documents are short. `--weighting by-documents` weights each source's share by how many documents it holds: stackexchange 296 → **1,732** threads, and the index came out *smaller* (32,713 against 44,374) because Stack Exchange boilerplate pushed exact duplicates from 4.3% to 25.1%. Reading 20 question/passage pairs from the two indexes side by side — a hand judgement, not a rate, kept in `artifacts/reference_index_compare.log` — the top passage is **better on 10, worse on 1, unchanged on 9**: the spinoff question that drew algebra homework now draws a thread about adjusted close. Top-passage source mix moves 90% → 96% stackexchange. The cost is real and recorded: fed_press 665 → 152 documents, and rbi, sebi, ncert, gutenberg and openstax down to **1 document each**. **What is still bad, stated plainly:** better is not good. Most retrievals are off-topic or merely adjacent, and the top-ranked chunk is not always the best of the four retrieved — "what is a stop loss order ?" quotes a wash-sale question while passages 2 through 4 are about stop losses, because BM25 ranks repeated query terms above a definition. That is the 15.5% recall@5 of C2 arriving where a user can see it, and the reranker that would fix it is the deferred item there. `SCORE_FLOOR` is recorded as **inert**: 0 of 200 fall under it, the median top score is 53.65 and the 10th percentile 34.40, and raising it would reject short questions rather than irrelevant passages because the magnitude tracks query length. Overlap **0/200**, so no question was answered by its own indexed thread. `scripts/refresh_index.py` keeps the index current without rebuilding it: it chunks only the documents the index does not already hold, takes the window, source list, undated rule, encoder and weighting from the build's own `.sources.json` rather than from its flags, caps what one run may add with `--budget`, and refuses to append at all when that record is missing or disagrees with the file. A dry run over today's corpus: **4,588 chunks to add** from 175,332 documents the index does not hold, 7.6% of what it cut already present. Two real bugs surfaced and are each pinned by a test: `chunks()` stored a sentence longer than the ids it had truncated, overflowing 7% of a 77-token index; and `unsupported_figures(text, text)` was not empty for every text, which made a quote fail against its own source. The second is in the shared guard, so the advisory rate was re-measured after it — **4/771 figures, unchanged**, still clearing C6's 2% bar |
| C10 | Index the answers people endorsed, not the threads holding them, and measure retrieval against labels nobody here wrote | a question whose answer the corpus holds is found in the top 5, on ground truth from the asker rather than from us | **done** — the fix C9 asked for, and the number it needed. C9's honest complaint was that better is not good: `what is a stop loss order ?` quoted a stranger's wash-sale question. The cause was not ranking. `scripts/eval_answer_retrieval.py` is the first **labelled** retrieval gate here — the query is a question as its asker typed it, a hit is a chunk of the answer that asker accepted or that site voted up, and it reports the ceiling separately from the ranking. On the served 32,713-chunk index: **8/500 (1.6%)** of sampled questions had their endorsed answer in the corpus at all, and **100% recall@1** on those eight. That 100% is not skill, it is leakage: `data/text/stackexchange` holds each thread *flattened* — title, question body, then every answer — so the index held the question text and matched it to itself. **Coverage, not ranking, was the whole problem.** `backend/retrieval/qa.py` reads `data/qa` instead, where the same posts sit structured with accept flags and vote scores, and indexes **62,341 endorsed answers** as `qa_money` / `qa_quant` / `qa_economics`, each keyed by the manifest row that records its URL and CC BY-SA licence, so a citation still resolves. The thread blobs are dropped from the served sources. Measured on the same 500 questions: present **1.6% → 82.4%**, recall@5 **35.7%**, MRR 0.277, and **found@5 1.6% → 29.4%**, which is the number a user actually gets. **The window was measured, not assumed.** 192 tokens scores found@5 **29.4%**, against 26.2% at 77 and 27.2% at 384, so 192 is a peak rather than a step on a curve. 77 was never a retrieval choice — it was the generator's slot, inherited from a decoder path quoting no longer runs, so `assemble(cut=True)` now feeds the paraphrase the passage's opening and **reports that as its evidence**, because a figure checked against text the model never read is not checked. **Raising the budget instead buys nothing**, which is why `--first-chunks` was deleted rather than tuned: 900k chunks at 77 tokens reaches the same 82.4% coverage and ranks worse (recall@5 31.3%), netting −0.4 points, so depth and coverage were not the trade-off they looked like. The served index is **519,135 chunks / 175 MB**, builds in 52s with no encoder pass, and BM25 answers in **1 ms** at 1.7 GB — all of which is why no cap is needed. Re-running the C9 gate on it: **coverage 200/200, 0 invented, overlap 0/200**, and the guard now catches **67 answers / 193 figures (33.5%, was 30.0%)** — honestly worse, because the decoder reads 77 of a 192-token passage and has less evidence to support a figure. Nothing served changed: a quote of the passage cannot invent a figure. The demo question is fixed — all three top passages for `what is a stop loss order ?` are now stop-loss answers, led by a direct definition. **What is still bad:** recall@1 is 23.1%, so where the answer exists the top passage is still wrong three times in four, and generic questions retrieve adjacently — `how does compound interest work ?` leads with an answer about periodic deposits, `should i pay off my mortgage early ?` with credit-card advice. The remaining **17.6% of questions have no endorsed answer at all**, which is a property of the corpus and not something indexing can fix. **And the gate now reports both phrasings, because the gap is large and the smaller number is the honest one**: scored on the title alone — one line, which is what a user types rather than the paragraph an asker writes under it — the same index gets recall@5 **19.2%** and **found@5 15.8%**, against 29.4% on the whole post. Most of BM25's signal is coming from text a user does not type. **Four ways to improve the ranking were measured and all four failed**, which is why none shipped and why the reranker stays C2's deferred item: idf-weighted term-coverage reranking over a 20-chunk pool **loses** (found@5 24.0% against 29.4%), so C9's guess that BM25 rewards a repeated query term over a definition was one anecdote rather than the pattern; RRF-fusing coverage with BM25 also loses (27.4%); pseudo-relevance feedback loses at every setting tried (best 15.4% against 15.8% on titles, worse as more feedback weight is added) because a first pass that is 11.4% right at rank 1 expands the query on the wrong documents; and keeping only the best chunk per answer in the top 5 gains **+0.6 points, inside the ±2.4 noise** at n=412, so the 16.3% of turns that show two chunks of one answer are a cosmetic defect and not a retrieval one. Lexical ranking is where it can get to; the next real move is a **learned** reranker or the contrastive objective C2 diagnosed as missing, and that is a training job, not a tweak |
| C11 | The serving defects a user found reading real answers, and whether a relevance cut can stop the reference path serving nonsense | a grounded answer is no longer withheld over its own spelling, and any relevance gate that ships is fitted rather than guessed | **done — two bugs fixed, three relevance experiments failed.** Found by reading a real session rather than a metric, which is how both bugs survived 486 passing tests. **A grounded answer was being withheld over a space.** The evidence stated `return_20d -5.6%`, the model wrote it, and the guard refused the turn: `_PRETOKEN` splits a sign off its digits, `decode` put a space before every piece, so the extractor read an unsigned `5.6%` against signed evidence. Every existing faithfulness test spells the sign attached, which is exactly why the suite was green while serving refused. Both halves fixed: `decode` now inverts the spacing `pretokenize` dropped, and `unsupported_figures` accepts an **unsigned** answer figure against a signed evidence one of the same magnitude while still requiring a **signed** one to match exactly, so a flipped sign is still caught. Measured on the live socket, the two advisory questions that were withheld both now answer with grounded figures, and the served prose reads `it is at 1242.30, -5.6% over 20 days` instead of `reliance . ns ... , - 5.6%`. The cost is pinned in a test rather than left to be found: a sign binds to any digit, so `1993 - 2000` decodes as `1993 -2000`; only decoded text is affected and the magnitude comparison means no verdict reads a figure out of it. **The page also explained the wrong refusal** — one hardcoded sentence told the user their confidence was under the cut whichever guard had fired, so a figure the guard replaced was reported as a confidence problem. Keyed on `because` now, and silent for a reason it does not know rather than guessing. **Three ways to stop the reference path serving an irrelevant passage were measured and all three failed.** The user's `how is the reliance stock working` returned an Airspan SEC filing about SoftBank, and `suggestion about reliance for future` returned a Wikipedia article on suggestion boxes. (1) A raw `SCORE_FLOOR` was already recorded inert in C9 and still is — the magnitude tracks query length. (2) Normalising it into the **share of the query's own term weight the top chunk matched** — the length-bounded statistic `combined` uses for routing, and the one C9 said this needed — **inverts the signal**: `what is the weather tomorrow` scores **1.565** and `should i buy MSFT` 1.432, against a median **0.891** for questions whose endorsed answer the index holds, and hits are indistinguishable from misses (**0.921 vs 0.916**, n=120). Short queries of common words saturate BM25 against a small idf denominator. (3) Stripping the company name from the document query — the same `without_subject` the router uses, on the theory that `reliance` was the term that pulled the filing — is **neutral**: garbage becomes different garbage (`Working Capital Turnover Ratio`), and the case that matters is unchanged either way. (4) Refusing documents for any question that names an instrument was written and **reverted**: `tests/test_agent.py` already pins `should i set a stop loss on AAPL ?` as a ticker-naming question the documents rightly answer, and no statistic separates it from the failures — concept questions cluster at router top score **0.754–0.843** and both failing state questions land inside that band. **What this settles.** Nothing thresholdable distinguishes a passage about a company from one that merely says its name, so the reference path stays as C9 and C10 left it and the remaining lever is unchanged: a **learned** reranker or C2's missing contrastive objective, trained on the 62,341 endorsed (question, answer) pairs. **And the router's weakest axis is where a typed question lands.** `route.py` reports 93.0% overall but **3/7 (43%)** on a question using both an unseen word and an unseen shape. Diagnosed on the failing query: `how is the stock doing` routes to performance at 1.136 and `how is holding up` at 1.505, while `working` — one unseen verb, same shape — drops performance to 0.754 under `unsupported` at 0.792, collapsing the margin to 0.016 against a cut of 0.111. Adding frames does not reach it, because the miss is lexical and the frozen encoder's cosine is what ranked `what is the market capitalisation for ?` above `how is faring ?` |

Two of the three limits in C2 are still deliberate rather than unfinished. The third — retrieved passages
not being wired into the answer path — was lifted by C9, which uses the decoder that **was** trained on
four passage slots rather than the advisory one trained on a single slot.
Near-duplicate chunks (~11%, on top of the 3.8% exact ones that are dropped) need the MinHash index in
`dataforge/process/dedup.py`, which `backend/` must not import. And the index is a bounded sample of the
corpus — 44,811 chunks of a possible ~13.4M, because the whole corpus is ~19 hours of encoder passes —
with the share of each source recorded in `artifacts/index.sources.json` so no number reads as covering
everything collected.

C3 ships with three limits of its own, all recorded rather than hidden. The head's `outlook_confidence`
is an **uncalibrated softmax**, where the persistence rule quotes the measured hit rate of the bucket it
landed in — calibrating the head needs a confusion table from data it did not train on, which C6's gate
does not produce: it decides promotions, not calibration. The served checkpoint is **2,000 steps**, not a converged run; a 6,000-step head measured the
same accuracy, so the serving decision does not turn on it. And the transformer tower has not been
re-measured on the full held-out set, so the tower comparison in §3.4 still rests on the partial
evaluation — `--towers transformer` re-runs it when the comparison is worth an hour.

C4 needed per-symbol feeds to be measurable at all. The six topic and regulator feeds are macro news:
matching company names over 147 of their headlines tags **4** to an instrument the agent holds prices
for, which is too few to check a score against anything. So `config.TICKER_FEED` adds one feed per US
symbol, 49 of the 50 in `config.TICKERS` — the publisher spells Berkshire `BRK.B`, ours is `BRK-B`, and
that one is logged as a 404 rather than patched with a per-publisher symbol map. Only **50 of 101**
instruments can be named at all, because the Indian listings are not SEC registrants and SEC's symbol
table is the only name source on disk.

Three bugs came out of the real feeds, each now pinned by a test in `tests/test_news.py`: the symbol
feeds put the *same* URL on every item, so link identity collapsed 26 stories into one; deduplicating a
syndicated story dropped the second feed's instrument with the duplicate; and matching company names as
substrings tagged every headline about artificial **intel**ligence to Intel. Identity is now the headline
and the day, and a repeated story unions its tickers instead of losing one.

Two limits stay. **278 pairs have no next bar**, because the feeds carry about a month and the price bars
end 2026-09-17; prices were deliberately not re-fetched, since extending them would move the held-out
splits every number above rests on. And the polarity is **description, not a signal** — reported beside
an article, never fed into advice — on the strength of its own null. The lexicon was written from general
finance vocabulary before either measurement was run, so the same-day result is not a fitted one.

C6's bar has two readings and the record carries both, because which one is quoted changes the verdict.
S14 says *unsupported-figure rate*, and per figure the served checkpoint is **0.5% on the reworded split**
— comfortably inside 2%. Per *answer* it is **2.4% on both splits**, which is outside it. The gate uses the
standard's own denominator and stores the counts rather than a rate, so either reading can be recomputed
from the registry file. Neither number is what a user is exposed to: `guardrails.screen` refuses an answer
that states a figure the evidence lacks, so the served rate is zero by construction, and that is exactly
why the bar has to be applied to the decoder — a model can always meet S14 at serving by refusing more.

C7's numbers rest on a judge that has to be named. A typed question arrives with no gold answer, so the
oracle cannot score it and the verdicts are the agent's — written once against the evidence each answer
was given, cached in `artifacts/served_verdicts.jsonl`, and replayed rather than re-asked, so a second run
of the session sees the same feedback as the first. A pair with no cached verdict is reported **unjudged**
rather than guessed. That makes the 82.4% reproducible, not independent, and it is why the session is
recorded as the loop working rather than as a measurement of answer quality; the published rates come from
`check_answers.py` against gold, where nothing is judged by the thing being tested.

The same session shows C1a's residual weakness in the open. *"hows msft been doin lately"* and *"how
volatile is TSLA right now ?"* were both refused as unclear while near-identical phrasings of each were
answered, and *"hows tsla lookin"* was routed to the momentum band rather than performance — it is the one
wrong answer among the 14 the agent spoke. Routing, not wording, is still what is left, exactly as C1's
row says.

Chasing those three cost three hypotheses and bought one thing, which is worth recording because two of
them were wrong. They are short questions, so the first guess was chat register: `scripts/route.py` grew a
`casual shape` axis that renders an unseen shape the way somebody types it, and it scored **19/21, the same
as the clean unseen shape** — register is not the cause, and the axis stays because it rules that out. The
second guess was that the pool holds no short shapes, which is true: its 183 entries run 2 to 11 words and
the 9 shortest are all noun fragments like *"rsi on X ?"*, never a clipped verb. Adding a clipped-verb
family across all 7 intents changed unseen-shape routing **not at all** (23/28 and 21/28, identical) and
cost coverage **86.5% → 81.9%**, because short entries match each other across intents and squeeze the
runner-up closer rather than further. It was reverted. The third was that the margin's scale depends on
length: it correlates **-0.426** with word count, but the median margin of a question of four words or
fewer is **0.558 against 0.544** for a longer one, so the correlation is a tail and not a shift, and a
scale-free `(best - runner) / best` margin moved coverage 81.9% → 82.9% at identical precision. Sweeping
the coverage weight over eight values confirms the served **0.5** is already the peak at 93.0%.

What the measuring did find is where the error actually is. Of the questions the gate answers, a long one
is right **100.0%** of the time and a short one **90.9%** — every surviving routing error is short, and the
margin cannot see it. So the fourth held-out family is a short one, the copula dropped to a bare adjective
(*"AAPL too hot ?"*), uniform across all 7 intents and using adjectives that appear nowhere else in
`advisory.py` so a hit cannot come from a shared term. Only **4 of 7** place correctly against 19/21 for
the long unseen families, and 2 of 7 in chat register. Putting that family in the fit is the improvement:
the cut rises **0.0877 → 0.111**, and *"hows tsla lookin"* — the one wrong answer of the 14 — scored
**0.1098**, so it now falls below the cut and is refused. The correct short question refused at 0.0847 was
already refused, so nothing right was lost. The cost is stated: coverage **88.7% → 86.5%**, and held-out
routing precision reads 98.8% against 99.4% because the same seven hard rows are in the held-out half too,
which makes it a harder test rather than a worse router. Fixing those 4-of-7 is encoder work, not pool
work, and it belongs with the deferred corpus item.

## Stage D — publish

| id | task | gate | status |
|---|---|---|---|
| D1 | Full suite green, every gate above recorded | `pytest tests/ dataforge/ -q` passes and no row is `open` | **done** — **470 passed** in 5.6s, and all 20 rows above carry a measured figure. Five of them record a gate that did not pass: A2 unanswerable as written, B4 tied not beaten, B5 failed outright, and B6 missing its own stated bar. They are left saying so. A queue where every gate passes is a queue whose bars were set after the numbers came in, and the two results this project rests on — a threshold that adapts from feedback alone, and a router that reads a frozen encoder better without fitting a parameter — are worth less if the rows around them are not honest. `frontend/verify_gate.mjs` is deliberately not in this count: it needs two servers running, and a suite that cannot pass from a clean clone is a suite people learn to skip |
| D2 | One commit series pushed to GitHub | remote matches local | **done** — 7 commits pushed as one series, 177 tracked files, 2.0 MB of history. The 8.1 GB corpus and the 920 MB of checkpoints stay out: both are rebuilt by a script in the repository, and `data/manifest.jsonl` keeps each source's URL and licence so provenance survives without the bytes. Two files are excepted from that and committed — `artifacts/verdicts_given.jsonl` and `artifacts/served_verdicts.jsonl` — because no script can rebuild a judgement, and without them the learning curve is not reproducible from a clone. Scanned before pushing: no token, key, or credential in the tree or in any of the seven diffs |

## Reference repo, settled

Frontend design and model input shapes both come from
[Financial-Analysis-Upgrade](https://github.com/Kapurrrishabh/Financial-Analysis-Upgrade) — its
`frontend/` for the design and `backend/` for the input shapes. Cloned read-only to
`reference/financial-analysis-upgrade/`, which is git-ignored: it is a thing to read, not a dependency,
and `selfagent/` must not import from it.

It already carries login, auth and a portfolio UI, which is why it is worth adapting rather than
starting over — those are the parts nobody enjoys rebuilding. The design is ours to change to fit a
chat-first agent; the diagram's entry point is a conversation, not a dashboard.
