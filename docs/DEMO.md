# Viva demo — the runbook

Every command below was run on 2026-09-22 and the numbers quoted are what it printed. All of it is
offline, single process, no API key and no network. Nothing takes longer than seven seconds except where
noted, so the whole demo fits in ten minutes.

Run everything from the repo root. `python3`, not `python`.

---

## 0. The suite is green (3 s)

```bash
python3 -m pytest tests/ dataforge/ -q
```

> `407 passed`

Say: every claim in the demo has a test that fails if the behaviour changes, and five of them were
verified by deliberately breaking the code first.

---

## 1. The agent answers, and refuses (2 s)

```bash
python3 scripts/ask.py
```

Six questions chosen to hit each way a turn can stop. Expect **2 answers and 4 refusals**:

| question | what happens |
|---|---|
| `how has AAPL been doing over the last month ?` | writes a correct answer, then **stays quiet** — its stated chance of being right was 9%, under the bar |
| `hows msft been doin lately` | `[unclear question]` — the router's margin is under the gate |
| `is AAPL overbought right now ?` | **answers** — rsi 65, neither overbought nor oversold |
| `how risky is AAPL over the next week ?` | **answers** with the price head's outlook and its confidence |
| `how is TSLA doing ?` | `[low confidence]` |
| `write me a poem about the sea .` | `[no subject]` |

The three lines to point at:
- `routed to risk (margin 0.297), asked as '...'` — the question the decoder actually reads is the
  trained phrasing, not the user's words. That rewrite is worth **33.5% → 73.5%** exact match.
- `evidence ticker AAPL ; close 337.00 ; ... ; outlook normal ; outlook_confidence 38%` — every figure
  the answer is allowed to say. Arithmetic computed it, not the model.
- `QUIET i do not have that in the evidence ...` — a refusal names its stage. None of them is a generic
  apology.

## 2. Your own question (1.5 s)

```bash
python3 scripts/ask.py "how risky is AAPL over the next week ?"
python3 scripts/ask.py "what is the drawdown on RELIANCE ?"
python3 scripts/ask.py "should i buy TSLA ?"
```

101 tickers are on disk, Indian ones included — `RELIANCE` resolves to `RELIANCE.NS` without the suffix.
Ask about a ticker that is not there (`how is NVDA doing ?` if it is missing) and it refuses with
`no subject` rather than guessing.

**If asked about look-ahead:** the same question as of any past date, reading no bar after it.

```bash
python3 scripts/ask.py "how risky is AAPL over the next week ?" --as-of 2024-06-03
```

> `close 194.03 ; ... ; outlook calm ; outlook_confidence 52%` — a different world, built only from bars
> up to that date. `tests/test_agent.py` pins that a window served live equals one rebuilt from a file
> truncated to that date, exactly.

---

## 3. The learning loop, live — the thesis (2 s)

```bash
python3 scripts/chat.py --dataset advisory --checkpoint advisory_combined.npz \
  --split validation --turns 20 --log /tmp/demo.sqlite
```

Twenty turns, each one judged, and the abstention cut refitted after every turn. The last line is the
demo:

> `20 turns, log 0 → 20 rows (+20)`
> `spoke on 16, right on 16 (100.0%); answering all 20 would have been right on 17 (85.0%)`
> `stayed quiet on 4, and the cut moved 0.99800 → 0.96338`

Say plainly: **the transformer's weights are loaded once and never touched.** What changed between turn 1
and turn 20 is the cut and the calibrator, both solved from the log. `--log /tmp/demo.sqlite` keeps the
real feedback log clean, so this is re-runnable as often as you like.

For the harder, honest version — reworded questions the decoder has never seen — drop the flags:

```bash
python3 scripts/chat.py --turns 6
```

This one mostly refuses and is right on 0. That is the measured paraphrase weakness (90.5% on trained
wording, 28.5% on reworded), and the refusals are the system behaving correctly on it.

---

## 4. The learning curve — does more feedback help? (0.5 s)

```bash
python3 scripts/learning_curve.py
```

The agent is told *"answer only where you are at least 60% likely to be right"* and solves for the cut
from its own logged outcomes, scored on rows the cut never saw, averaged over 50 seeds:

| feedback rows | delivered | miss vs the 60% bar |
|---|---|---|
| 10 | 53.5% | 12.9 pts |
| 60 | 59.9% | **8.5 pts** |
| 118 | 59.7% | 9.6 pts |
| *fixed hand-picked cut* | 43.3% | 16.7 pts |
| *answer everything* | 30.8% | 29.2 pts |

Two flat controls, beaten at every log size past ten rows. **It learns when to speak, not what to say** —
and the second half of that is measured too: eight samples of this model give 1.77 distinct answers, so
there is nothing for a reranker to choose between.

## 5. The confidence, and whether it is worth anything (0.2 s)

```bash
python3 scripts/calibrate_confidence.py
```

> raw confidence calibration error **0.683** → calibrated **0.121**, Brier 0.676 → 0.183
> ranking auc **0.740 ± 0.039**

The honest part: the base rate — a constant that ignores the confidence — ties on calibration error
(12/25 splits) and loses on Brier (25/25). So the confidence carries *some* signal and not much, and the
ranking AUC of 0.740 is the ceiling on everything built on top of it.

## 6. The cut at every bar you could ask for (0.5 s)

```bash
python3 scripts/fit_abstention.py
```

> asked for 60% → cut 0.9998 → **58.5% correct on 32.7%** of questions, +27.8 pts over answering all
> asked for 80% → cut 0.9999 → 64.8% correct on 22.5%

Fitted on half the log and quoted on the other half. The in-sample number (68.9%) is printed next to it
as the upper bound it is.

---

## 7. The price head, and the rule it had to beat (1 s)

```bash
python3 -c "
from pathlib import Path
from backend import models
from selfagent import pretrained
m = pretrained.metadata(Path('artifacts/price_head.npz'))
print('held out     %.1f%% on %d windows' % (100*m['accuracy'], m['evaluated']))
print('persistence  %.1f%%   majority %.1f%%' % (100*m['persistence'], 100*m['majority']))
print('margin needed %.2f pts, measured gap %.2f pts' % (100*(m['accuracy']*(1-m['accuracy'])/m['evaluated'])**0.5, 100*(m['accuracy']-m['persistence'])))
print('serving uses', models.load(Path('artifacts/price_head.npz')).version)
"
cat artifacts/price_head.log
```

> held out **49.1%** on 28,676 windows; persistence **47.8%**, majority 39.6%
> margin needed 0.30 pts, measured gap 1.36 pts → `serving uses gru@62babf47cdd5`

The artifact's own recorded numbers choose which advisor answers: the head has to clear one line of
arithmetic by more than one standard error of its own accuracy, or the arithmetic serves instead. **Two
findings here, and one of them is a mistake I found:** 5-day direction is unpredictable (35.7% against a
36.0% baseline) so the agent never gives directional advice; and the earlier reading that the head merely
*ties* the baseline came from scoring it on the first 1,920 windows, which is 7 of the 101 tickers, while
the baseline was scored on all of them. See §5.14 of `learnme.md`.

Retraining it is ~20 min, so do not do it live. If asked:
`python3 scripts/train_price_head.py --towers gru --steps 2000 --save price_head.npz`

## 8. Routing, measured before it is served (2 s)

```bash
python3 scripts/route.py
```

> semantic **94.3%** on intent (chance 14%); 100% on chat-register questions; **68%** on a novel sentence
> shape — the weakest axis in the project, and the one worth fixing next
> at a 97% bar it refuses **12/12** out-of-domain questions at 88.5% coverage, where BM25's margin serves
> *"write me a poem about the sea"* as a performance question

## 9. Proof the model reads its evidence (7 s)

```bash
python3 scripts/ablate_generator.py --checkpoint advisory_combined.npz --batches 40
```

> as given loss **0.0041** · evidence blanked **+1.2945** · evidence swapped **+2.9318**

The single most important measurement in the project. The first generator scored **+0.018** when its
evidence was blanked — fluent and completely ignoring its input, which only ablation revealed.

---

## If the examiner asks for the weakest point

Say it before they find it:

1. **Paraphrase generalisation.** A novel word is fine (15/15); a novel sentence *shape* is 68%, and
   fine-tuning made it worse (68% → 32%). Routing on unseen shapes is 23.5 of the remaining 26.5 points
   of error, so it is the one number left worth moving.
2. **The price head's `outlook_confidence` is an uncalibrated softmax**, where the persistence rule quotes
   a measured hit rate. Named limitation, not shipped silently — the fix belongs with the eval gate.
3. **Unsupported figures are 1.7% on trained wording and 4.7% on reworded**, so the 2% target closes on
   one and fails on the other. Both numbers travel together.
4. **No FastAPI backend or frontend yet**, and retrieved passages are deliberately not wired into the
   answer path: the decoder has exactly one passage slot, so five would be out of distribution.
