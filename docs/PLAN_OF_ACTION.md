# Plan of Action: Self-Learning Financial Agent

Companion to `RESEARCH.md` (design rationale, literature, risks). This document is the build plan:
module layout, phases, exit gates, and the experiment that produces the headline result.

Date: 2026-09-06. Model configuration updated 2026-09-14 with measured step times (§4). Target
architecture revised 2026-09-20 to the conversational advisory design in §3a.

---

## 1. Objective in one paragraph

Build, in NumPy only, a **conversational finance advisory agent**: it answers a user's question in
sentences, grounds every answer in retrieved documents and computed price facts rather than in its own
weights, and gets better from use. The language backbone is a Transformer encoder pretrained with
masked language modelling on a finance corpus; answers are written by a decoder that **cross-attends
to retrieved passages**, so the model paraphrases text it can see instead of recalling facts it
cannot. A price tower reads OHLCV windows for direction and confidence. Around all of it sits the
learning loop that is the actual research contribution: the agent selects its own uncertain cases,
takes user feedback and realized market outcomes, and updates a **per-user LoRA adapter** with replay
and EWC — improving for that user **without retraining the foundation model**. Proven by measuring
answer quality and accuracy against rounds of interaction, versus a no-feedback and a random-query
control.

**No external LLM and no API keys anywhere in the system.** The model we build *is* the language
model. Where that costs us fluency, we pay it and say so.

---

## 1a. Status

| Phase | State |
|---|---|
| 0 — environment, benchmark | **Done.** §4 measured; corpus audited in place of the missing-data audit |
| 1 — autograd engine | **Done.** 46 gradient checks green |
| 2 — layers, encoder, optimizer | **Done.** Overfit gate green |
| 3 — tokenizer, corpus, MLM pretraining | **Done.** 100.8M tokens; validation loss 3.03 vs uniform 8.99 (S3 closed) |
| 3a — generator stack (decoder, grounded seq2seq) | **Done, trained.** Causal-mask, grounding and overfit gates green |
| 4 — retrieval + generation training | **Grounding done, paraphrase open.** BM25 only; no dense retriever, no reranker |
| 5 — price advisory, both heads | **Both built and compared.** Direction is unpredictable here; volatility loses to persistence |
| 6–9 | Not started |

309 tests pass (`python3 -m pytest tests/ dataforge/ -q`). Gates closed: S1, S2, S3, S11, S15. **S14
closes on trained phrasings only.** Re-measured on the current model over 200 generated answers: 1.7%
unsupported figures on validation, under the 2% gate and down from 10.5% on the previous model — but
4.7% on phrasings training never saw. The gate is met where the wording is familiar and missed where it
is not, which is the same paraphrase failure below, reaching the number the product is judged on.

**Phase 3 results, measured.** Corpus 454 MB of cleaned text over nine sources → 100,828,672 training
tokens and 1,018,368 validation tokens at `max_text_length` 128. MLM validation loss fell 6.19 → 3.03
over 48,000 steps (perplexity 20.7) against 8.99 for a uniform guess, so the backbone learned finance
language rather than memorising a prior. Remaining weakness on probes: number pieces score 29.8%,
which is real learning of magnitude and format against 10% chance, not a bug to fix — the digits are
already in the character vocabulary.

**Phase 4 results: grounding is proven, not assumed.** The advisory dataset is 321,279 rows over 7
intents and 190 question phrasings built from frames × interchangeable words (19,183 distinct
questions). Ablation on the converged checkpoint — loss 0.0073 as the evidence is given, 1.0493 with it
blanked (**+1.0420**), 1.9537 with another row's numbers swapped in (**+1.9464**). Swapping costing
*more* than blanking is the ordering that proves reading: a model reciting a memorised shape still
scores well with no evidence, but not while staring at wrong numbers. The old Stack Exchange task cost
+0.018 on the same test, so this is ~58× its grounding. S14/S15 close on the guardrail plus the slot
path, which makes faithfulness structural — a slot answer names fields and states no figure of its own.

**What is still open is paraphrase.** Loss on phrasings training never saw bottoms at 0.2821 while
trained phrasings reach 0.0057. Four times more phrasing variety cut that floor from 0.70 to 0.28, so
the wording of the data was the binding constraint, not its size. `scripts/probe_intents.py` says why:
the MLM encoder places a word it has never read 15/15 given a familiar sentence shape, an unfamiliar
shape built from familiar words only 28/41, and both-novel 2/7. Task fine-tuning then *halves* shape
handling, 28/41 → 13/41 — catastrophic forgetting, measured. So the ~600M-token general-English corpus
is **not** the blocker and is deliberately not being collected; `--freeze-encoder` tests the other half.

**Held-out loss is the wrong way to pick a checkpoint here, and that is now measured rather than
suspected.** On the chat-register arm the loss-selected checkpoint (step 6,000, unseen loss 0.2339) loses
to the final one (unseen loss 0.4162) on every generation number: exact match 32.5% → **39.0%**,
unsupported figures 17.2% → **4.1%** of answers, false refusals 18.9% → **4.7%**. On the register's own
split the same comparison is 68.0% → **94.5%** exact match. Early stopping would have shipped the worse
generator with a better loss curve. The ablation says why: the evidence-swap penalty *grows* with
training, +1.69 → **+2.60** on unseen, so the late model commits harder to the digits its evidence
states — which costs average cross-entropy on unfamiliar phrasings while making answers more correct.
Every checkpoint decision from stage B onward is therefore made on generated answers, not on loss. The
combined arm then **replicated it independently**: loss-picked 0.1435 against final 0.2496, and the
loss-picked one worse on all four generation numbers again (30.5% against 36.0% exact match, 17.2% against
2.4% unsupported, 7.7% against 3.6% false refusal). Two arms, one conclusion.

**Stacking the frozen encoder onto the richer phrasings adds on faithfulness and not on exact match.** Same
splits, same 200 rows, one fix apart: unseen loss 0.4162 → **0.2496**, unsupported figures 4.1% → **2.4%**
of answers, false refusals 4.7% → **3.6%**, and own-split exact match identical at 94.5% (189/200 in both).
Unseen exact match reads 39.0% → 36.0%, which is 6 rows and sits inside the ±3 points this sample carries
at n=200, so it is not a cost the measurement can distinguish from noise. So `--freeze-encoder` stays on
for the served model: it buys the faithfulness the S14 gate is written against, for no measurable
accuracy.

**The real-human-Q&A dataset was re-scored, both directions, and it is still the harder task.** The
Stack Exchange model on its own validation: loss 4.1227 (perplexity 61.7), blanking the evidence costs
**+0.0191** and swapping it +0.0302 — 55-65× weaker grounding than the advisory task, the original
diagnosis reproduced. Generated answers are fluent and unfaithful: 0% exact match, **57.1% of figures
unsupported**, repetition healthy at 2.6% against the humans' 2.1%. The advisory model scored on Stack
Exchange reaches loss 11.96, i.e. nothing transfers across answer distributions. Exact match is not a
usable score on human prose — there is no single right wording — so loss plus the blank/swap deltas are
the measurement, and they say the same thing they said before: 30.7% of answer content words are absent
from the input, so most rows are unanswerable from their evidence and the decoder correctly learns to
ignore it. Fixing this is a retrieval-coverage problem, not a model problem.

**Retrieval is lexical by decision, not by omission.** BM25 only. An oracle ablation showed a perfect
reranker would not fix what was actually costing answers, so the dense retriever and cross-encoder in
the Phase 4 list are deprioritised rather than pending, and S16 stays open.

**Phase 5 results, and two are negative.** Direction over the next week is not predictable from these
windows — measured against persistence, which is the honest baseline here, never majority class.
Volatility is the label that carries signal and it still loses to persistence. The GRU
(`RecurrentPriceTower`) beat the patch transformer, so the tower question is settled. The consequence is
written into the output contract: the agent reports risk and abstains on direction.

## 2. Success criteria

| # | Criterion | Measurement | Gate |
|---|---|---|---|
| S1 | Autograd is correct | Finite-difference check, every op | max rel. error < 1e-5 |
| S2 | Model can learn at all | Overfit 100 examples | train loss → ~0 |
| S3 | Pretraining learned something | MLM loss vs uniform (`ln(vocab)`) | clearly below; report perplexity |
| S4 | Text head beats a dumb baseline | Macro-F1 vs majority-class and TF-IDF+logreg, walk-forward | beats both |
| S5 | Numeric head beats a dumb baseline | Macro-F1 / MCC vs momentum and DLinear | beats momentum, or reported honestly if not |
| S6 | Fusion earns its complexity | Fused vs best single modality | improves, or fusion is cut |
| S7 | **Human feedback measurably helps** | Accuracy vs feedback rounds, 3 arms (§6) | uncertainty-query arm > random-query > no-feedback |
| S8 | **No catastrophic forgetting** | Global held-out metric before/after user adaptation | drop < 2 points absolute |
| S9 | **Users stay isolated** | User A's metric unchanged after user B trains | bit-identical backbone; A's metric unchanged |
| S10 | No leakage | Point-in-time assertion test | passes |
| S11 | Constraint honoured | Import-guard test | passes |
| S12 | Reproducible | One command + config regenerates every reported number | passes |
| S13 | Retrieval finds the answer | Recall@5 on held-out Q&A vs BM25-style lexical baseline | beats lexical, or lexical is kept |
| S14 | **Generation is faithful** | Every number and named entity in an answer traced to a retrieved passage or a computed fact | unsupported-figure rate < 2% |
| S15 | The agent abstains | Questions with no supporting passage | says so; never fabricates an answer |
| S16 | Generation earns its place | Grounded answer vs returning the top retrieved passage verbatim, human-rated | preferred, or we ship the passage |

S7, S8, S9 are the project. S1–S6 and S13–S16 are the platform that makes them measurable.

**S14 is the one that can sink the product.** An advisory that invents a number is worse than no
advisory, so faithfulness is measured as a hard rate, not a vibe. S16 is its honest counterweight: if
our own generated sentence is not preferred to simply showing the retrieved paragraph, then the
decoder is decoration and we say so in the report.

---

## 3a. Runtime architecture

What happens to one user question. Each box is a module in §3; the arrows are the only paths.

```
question
   |
   v
intent router  ----> which machine answers this? three kinds, three machines:
   |                   mechanism-why   ("what is a 10-K?")        -> retrieval
   |                   state/direction ("is RELIANCE falling?")   -> price advisory
   |                   event-why       ("why did it fall?")       -> event join
   v
context assembler ---- retriever (as-of filtered) -> reranker -> top-k passages
   |              \___ price advisory  -> direction, sigma, confidence
   |              \___ computed facts  -> close, 5d move, volume vs average, 52w high
   |              \___ portfolio + memory service -> what this user holds and asked before
   v
generator (encoder reads question + passages + facts; decoder cross-attends and writes)
   |
   v
guardrails ---- every figure in the answer must appear in the context, or it is cut (S14)
   |            no supporting passage at all -> abstain (S15)
   |            never a buy/sell instruction; direction plus uncertainty, and that is all
   v
answer + citations  ---->  feedback log
                              |
                        outcome labeler (realized forward return, no human needed)
                              |
                        training set -> retrain -> eval gate -> model registry -> promote
```

**Two rules this architecture exists to enforce.**

*Retrieval is the only source of facts.* The generator never emits a figure that is not in its
context. This is why the decoder cross-attends rather than generating freely, and why guardrails
re-check the output against the context even after training.

*The learning loop is the only path that changes a model.* Nothing is promoted without passing the
eval gate, and every endpoint is version-pinned, so a bad update is a rollback rather than an
incident. Feedback never edits weights in place.

**Three kinds of "why" need three different machines, and conflating them is how systems
hallucinate.** Mechanism-why is retrieval over textbooks and Q&A, and it works today. State and
direction is the price model. Event-why needs a point-in-time event-to-price join, which we do not
have yet: there is no CIK-to-ticker map in the repo and the mega-caps hold only 1–5 filings each
across ten years, so this path must abstain until Phase 6 builds the join.

---

## 3. Module layout

One responsibility per module; dependencies point strictly downward (`agent` → `learn` → `models` →
`nn` → `autograd` → `backend`). Nothing lower imports anything higher.

`[x]` exists and is tested; `[ ]` is planned. Planned files are listed for shape only — none are
created empty ahead of need.

```
selfagent/
[x] backend.py              # xp = numpy, dtype switch, finite-value checking. Only numpy importer
[x] config.py               # ModelConfig: validated, JSON round-trip, fingerprint

  autograd/
[x]   tensor.py             # Tensor, iterative topological sort, backward(), no_grad
[x]   ops.py                # add sub mul div neg matmul sum mean exp log sqrt tanh
                            #   reshape transpose concat getitem; broadcasting-safe accumulation
[x]   functional.py         # sigmoid softmax gelu layer_norm cross_entropy dropout embedding

  nn/
[x]   module.py             # Module + ModuleList: parameters, train/eval, state_dict
[x]   init.py               # INIT_STD in one place
[x]   layers.py             # Linear, Embedding, LayerNorm, Dropout
[x]   attention.py          # MultiHeadAttention (self and cross), padding_mask, causal_mask
[x]   encoder.py            # TransformerBlock (pre-LN) + TransformerEncoder  <- shared by D1
[x]   decoder.py            # pre-LN decoder block: causal self-attn + cross-attn + FFN
[ ]   recurrent.py          # GRU cell and stack, for the price head comparison in Phase 5
[ ]   lora.py               # LoRALinear: frozen W + B@A, B zero-init

  tokenizer/
[x]   vocab.py              # special tokens and ids, fixed order
[x]   wordpiece.py          # trainer and encoder
[x]   normalize.py          # SEC forms, currency, grouped numbers; fingerprinted

  optim/
[x]   adamw.py              # decoupled decay, skips 1-D parameters
[x]   schedule.py           # linear warmup then cosine
[x]   clip.py               # global grad-norm clip

  data/
[x]   features.py           # log returns + per-window standardising; forward_return labels
[x]   masking.py            # whole-word MLM masking
[ ]   qa_pairs.py           # data/qa/*.jsonl -> (question, accepted answer) and score rankings
[ ]   news.py  ohlcv.py     # loaders enforcing the two schemas (pandas)
[ ]   align.py              # POINT-IN-TIME news->price join, market hours, label horizon
[ ]   splits.py             # walk-forward with purge + embargo

  models/
[x]   embeddings.py         # TextEmbedding; AnswerEmbedding; PricePatchEmbedding (PatchTST)
[x]   heads.py              # tied-weight LM head, ClassificationHead, pool_cls
[x]   agent.py              # TextTower, PriceTower, MaskedLanguageModel, FinanceAgentModel
[x]   generator.py          # GroundedGenerator: encoder + cross-attending decoder + generate()
[ ]   retriever.py          # dual encoder over the pretrained backbone, as-of filtered
[ ]   reranker.py           # cross-encoder over (question, passage), trained on accepted answers
[ ]   fusion.py             # cross-attention fusion, once late-concat has a number to beat

[ ] learn/                  # the self-learning layer: uncertainty, feedback_store, outcome_labeler,
                            #   index, reward_model, policy, dpo, replay, ewc, user_profile, registry
[ ] eval/                   # THE ONLY package allowed to import sklearn/scipy
[ ] agent/                  # router.py, compose.py, guardrails.py, loop.py, explain.py, cli.py

tests/
[x] conftest.py             # numeric checking on for the whole suite
[x] test_gradcheck.py       # S1 - 46 finite-difference and graph-mechanics tests
[x] test_model.py           # S2 - overfit gates, wiring, causal mask, grounding, masking, features
[x] test_import_guard.py    # S11 - library constraint, one test per source file
[ ] test_leakage.py         # S10 - highest-value test in the repo
[ ] test_faithfulness.py    # S14 - no figure in an answer that is not in the context
[ ] test_user_isolation.py  # S9      test_forgetting.py  # S8

[x] configs/tiny.json  configs/target.json
[x] scripts/benchmark_step.py  prepare_pretrain.py  train_pretrain.py  export_model.py
[ ] scripts/train_generator.py  build_index.py  train_supervised.py  run_experiment.py
```

**Two module-level decisions worth recording.** The decoder is a separate class from
`TransformerBlock` rather than a flag on it, so the encoder path carries no branch it never takes.
And the token embedding is shared three ways — encoder input, answer input, output matrix — passed
through `forward()` instead of stored, because registering it twice would make AdamW apply every
update twice; at vocab 8,000 × dim 256 a second copy would also be 2.05M parameters, 39% of the model.

**Conventions:** every layer is a `Module` with `forward()` and `parameters()`; no layer touches disk
or the clock; all randomness flows from one seeded generator passed in from `config.py`.

---

## 4. Model configuration — measured

Step times below are **measured**, not estimated: `python3 scripts/benchmark_step.py` on the
development laptop (macOS, NumPy 2.0.2 on Accelerate BLAS), batch 32, sequence 128, vocab 8,000,
full MLM forward + backward + AdamW step. "Hours" projects a 15M-token pretraining run.

| config | layers / d_model / heads / FFN | params | s/step | tokens/s | hours for 15M tokens |
|---|---|---|---|---|---|
| tiny | 2 / 128 / 4 / 512 | 1.46M | 0.42 | 9,697 | 0.4 |
| **small** | **4 / 256 / 4 / 1024** | **5.32M** | **1.02** | **4,013** | **1.0** |
| medium | 6 / 384 / 6 / 1536 | 13.93M | 2.52 | 1,623 | 2.6 |
| large | 8 / 512 / 8 / 2048 | 29.65M | 4.64 | 883 | 4.7 |

**The finding that matters: compute is not the constraint — corpus size is.** Even `large` pretrains
in under five hours on a laptop. But 15M tokens against a 5.3M-parameter model is roughly 3 tokens per
parameter, where BERT-base had about 30. Scaling the model up on a small corpus buys overfitting, not
quality. So the lever to pull is **more finance text**, not more layers.

**Updated 2026-09-20, and it changes the conclusion.** Collection reached **100.8M training tokens**,
not 15M — 19 tokens per parameter at `small`, near BERT-base's 30. The corpus is no longer the binding
constraint on model size, so `medium` (13.9M params, ~7 tokens/param, 2.6 h) is now defensible and
should be measured against `small` rather than ruled out. What stayed true is the *composition*
problem, which matters more than the count: `sec_edgar` is 2.3 GB of the 2.8 GB of text and is held at
a 50% share cap, Indian sources are ~2%, and `news_rss` is 24 KB. More filings would add nothing. The
registers the corpus is short of are news and Indian material, and the fix for news is press-release
**archives**, not RSS feeds — a re-collect moved `news_rss` from 87 to 143 articles, and reaching even
5% of the corpus that way would take about 13 years of daily collection.

**Committed configs** (`configs/`, both round-trip through `ModelConfig`):

| | `tiny.json` | `target.json` |
|---|---|---|
| purpose | unit tests, CI — seconds per run | reported results |
| vocab / max_len | 1,000 / 32 | 8,000 / 128 |
| layers / d_model / heads / FFN | 2 / 64 / 4 / 128 | 4 / 256 / 4 / 1024 |
| price window / patch / stride → patches | 32 / 8 / 8 → 4 | 128 / 16 / 8 → 15 |
| price layers | 1 | 2 |
| fingerprint | `2b1d1625dddd` | `0842fa9e2fa8` |

`target.json` is `small`, chosen for the tokens-per-parameter reason above rather than for speed.
Revisit it upward only if the corpus grows past roughly 50M tokens. Note that the token embedding
(8,000 × 256 = 2.05M) is 39% of the model, so cutting vocab is the cheapest way to shrink it.

Remaining hyperparameters: LoRA rank 8 (~0.3% of params per user), batch 32, LR 3e-4 with 5% linear
warmup then cosine decay, AdamW weight decay 0.01 skipping 1-D parameters, gradient clip 1.0.

---

## 5. Phases

Each phase has deliverables and an **exit gate**. Do not start a phase before its predecessor's gate
passes; a failing gate is information, not an obstacle to route around.

### Phase 0 — Ground the plan in measurement — DONE except the data audit
- Install NumPy; confirm BLAS backend and thread count.
- **Data audit:** load the actual news and OHLCV files. Report row counts, date range, tickers, label
  distribution, missing data, and how many news items align to a valid price window. Write it into
  `docs/DATA_AUDIT.md`.
- `scripts/benchmark_step.py`: time a forward+backward for 3–4 candidate sizes.
- Fill in `configs/target.json` from those timings.
- **Gate:** measured step times recorded; `target.json` justified by a number, not a preference; data
  audit written.

> If the data turns out too small (e.g. <2MB text or <200 labelled examples), that is a Phase-0
> finding that changes the plan — surface it immediately rather than pretraining on it.

### Phase 1 — Autograd engine — DONE
- `Tensor` with reverse-mode autodiff over a topologically sorted graph; `zero_grad`; no-grad context.
- Ops from §3, each with a hand-derived backward.
- **Gate (S1):** `test_gradcheck.py` passes for every op — analytic vs central finite differences,
  max relative error < 1e-5. **Nothing else is built until this is green.**

### Phase 2 — Layers and the Transformer encoder — DONE
- `Module` base, Linear, Embedding, LayerNorm, Dropout, attention (with mask), pre-LN block, encoder.
- AdamW, warmup+cosine schedule, grad clipping.
- **Gate (S2):** encoder + head overfits 100 examples to ~zero loss on `configs/tiny.json`;
  gradcheck extended to composite modules.

### Phase 3 — Tokenizer, corpus, MLM pretraining — DONE
- WordPiece trainer + encoder; finance normalisation, every rule counted in the corpus before it was
  kept (measurement dropped more rules than it kept, and overrode intuition three times).
- Whole-word masking; MLM head with tied embedding weights.
- Pretrained on the finance corpus; loss curve logged; checkpointing survives a crash mid-run.
- **Gate (S3): passed.** Validation loss 3.03 against `ln(8000) = 8.99`, perplexity 20.7, on 100.8M
  tokens over 48,000 steps.

### Phase 3a — Generator stack — DONE, TRAINED
- `nn/attention.causal_mask`; `nn/decoder.py` pre-LN block with causal self-attention, cross-attention
  to the encoder output, and FFN; `models/generator.py` with `generate()` and greedy/temperature
  sampling.
- **Gates: green.** Decoder-block gradcheck covers the memory input as well as the sequence input,
  because a decoder that passes no gradient back through cross-attention leaves the encoder untrained
  while the loss still falls — it just learns an unconditional prior. A causal-leakage test confirms
  an earlier position cannot see a later one, verified by removing the mask and watching the test
  fail: the leak is only ~1e-2, which is exactly the size that trains to a plausible loss curve and
  then generates nonsense.

### Phase 4 — Retrieval and grounded generation (~3 weeks)
- `data/qa_pairs.py` over `data/qa/*.jsonl`: (question → accepted answer) for the generator,
  score-ordered answers for the reranker and for Phase 7's preference pairs.
- `models/retriever.py` dual encoder initialised from the pretrained backbone; `learn/index.py` for
  the chunk/embed/index worker, with **as-of filtering** so a passage published after the question's
  date is never retrievable.
- `models/reranker.py` cross-encoder; `scripts/train_generator.py` teacher-forced on retrieved
  passages, encoder warm-started from `artifacts/pretrained.npz`.
- `agent/guardrails.py`: figures cross-checked against the context, abstention when nothing is
  retrieved.
- **Gates (S13, S14, S15, S16):** recall@5 beats a lexical baseline; unsupported-figure rate under 2%;
  abstains rather than inventing; the generated answer is preferred to the raw top passage — and if it
  is not, we ship the passage and report that the decoder did not earn its place.

### Phase 5 — Price advisory: both heads, honest baselines (~2 weeks)
- `data/align.py` point-in-time join; `data/splits.py` walk-forward with purge + embargo.
- **Two price heads, built and compared, not chosen in advance:** the existing PatchTST-style
  transformer tower and a GRU (`nn/recurrent.py`, gradchecked). Both against momentum and DLinear.
- Direction plus calibrated uncertainty, never an instruction to buy or sell. A move inside one sigma
  of a stock's own volatility is reported as noise — most "why is it down" questions are noise, so the
  agent checks whether there is anything to explain before explaining.
- All baselines in `eval/baselines.py`; metrics incl. calibration (ECE).
- **Gates (S4, S5, S6, S10):** leakage test green; text beats majority and TF-IDF+logreg; the better
  price head beats momentum, or the result is reported honestly as a loss; fusion kept only if it
  improves on the best single modality.

### Phase 6 — Human-in-the-loop learning (~2 weeks)
- `learn/uncertainty.py`, `feedback_store.py`, `replay.py`, `ewc.py`, `online_update.py`.
- **`learn/outcome_labeler.py`: the cheapest signal in the system.** A direction call is scored by the
  realized forward return with no human in the loop, so the training set grows while nobody is
  watching. `forward_return` already exists in `data/features.py`.
- `learn/registry.py`: a retrain is promoted only through the eval gate, endpoints are version-pinned,
  and feedback never edits weights in place. A bad update is a rollback, not an incident.
- `nn/lora.py` injected into attention and FFN projections; backbone frozen.
- `agent/loop.py` + `cli.py`: show the answer, its citations, confidence and a short
  attention-based explanation; accept a correction; update immediately; persist.
- **Gates (S7 partial, S8):** LoRA zero-init test; one correction changes the prediction on that
  example; 20 corrections do not drop the global held-out metric by more than 2 points.
- **Also builds the event-why join** that Phase 4 had to abstain on: `data.sec.gov/submissions/` plus
  `www.sec.gov/files/company_tickers.json` give a keyless CIK↔ticker map. Indian company-level events
  are **not obtainable** under the no-API-key rule — NSE and BSE are bot-protected — so Indian
  event-why stays macro-only (RBI press releases), and the agent says so rather than guessing.

### Phase 7 — Preference learning and policy optimization (~2 weeks)
- Bradley-Terry reward model on frozen backbone features.
- **Preference signal is taken on selection, not on generation.** A thumbs-down on ~80 generated
  tokens is one bit over a long sequence; a pick among k candidate answers is a clean
  `(chosen, rejected)` pair, which is exactly what Bradley-Terry and DPO want. Stack Exchange scores
  and accepted-answer flags give the same shape of pair for free, as pretraining data for the reward
  model before any user has clicked anything.
- REINFORCE + learned baseline + KL-to-reference; DPO as the comparison arm.
- **Gate:** reward rises **and** held-out supervised macro-F1 does not degrade; KL stays inside its
  bound. Reward up + F1 down is reward hacking and must be reported as such, not as success.

### Phase 8 — Multi-user adaptation (~1.5 weeks)
- `learn/user_profile.py`: create / load / save / delete a user's adapter, reward model and buffer.
- Simulate 3 users with deliberately different preferences (e.g. risk-averse vs risk-tolerant
  labelling of the same headlines).
- **Gates (S9, S8):** backbone bytes identical after any user trains; each user's metric improves on
  their own labels; user A's metric is unchanged by user B's training; forgetting within bound.

### Phase 9 — Experiments, report, packaging (~2 weeks)
- Run the full experiment matrix (§6), 3+ seeds, produce all tables and plots.
- README, install and run instructions, architecture diagram, final report.
- **Gates (S11, S12):** import-guard green; every reported number regenerable by one command.

**Provisional total: ~18 weeks part-time**, up from 14: retrieval and grounded generation (Phase 4)
are new work that the original plan did not contain, because the original plan only ever promised a
3-class classifier. Phases 1–3 are done and off the critical path. Phase 4 is now the critical path —
nothing downstream can be evaluated conversationally until the generator has retrieval to stand on.

---

## 6. The headline experiment

This is what the project is judged on. Fixed budget of `K` feedback rounds, `n` items per round.

**Arms (identical seeds, identical data, identical budget):**

| Arm | Query selection | Update |
|---|---|---|
| A0 | — | none (frozen control) |
| A1 | random | LoRA + replay + EWC |
| A2 | **uncertainty (margin/entropy)** | LoRA + replay + EWC |
| A3 | uncertainty | reward model + REINFORCE |
| A4 | uncertainty | DPO |
| A5 | uncertainty | LoRA **without** replay/EWC (ablation: shows forgetting) |

**Reported per round:** macro-F1 on the user's held-out set, macro-F1 on the **global** held-out set
(the forgetting metric), calibration error, and labels consumed.

**Expected shape of the result:** A2 > A1 > A0 in user-set F1 at equal label budget; A5 shows the same
user-set gain as A2 but a visibly falling global metric — which is exactly the evidence that replay
and EWC are doing the job the literature says they should. A3 vs A4 answers whether the reward model
is worth its noise at low feedback volume.

**Ablations:** LoRA rank; fusion on/off; replay size; EWC λ; MLM vs no pretraining (this one directly
measures what pretraining bought us).

---

## 7. Immediate next steps

Rewritten 2026-09-21. The previous list — re-collect Stack Exchange, collect the archives, re-tokenize
and pretrain, then Phase 4 — is done. Everything below is ordered by what the measurements now say.

1. **Stop the fine-tune destroying the encoder.** The probe puts the paraphrase failure in the encoder's
   handling of sentence shape, and shows task training halving it. `--freeze-encoder` holds the four
   encoder layers at their pretrained values (3.2M of 7.5M parameters) while the shared token matrix
   stays trainable, because that matrix is also the decoder's output layer. Run it against the current
   baseline with nothing else changed, and compare the unseen-phrasing floor, not the final loss —
   `{dataset}.best.npz` now keeps the turning point, since the unseen split starts rising while
   validation falls forever. If freezing loses too much, the next step is a lower encoder learning rate,
   which needs per-parameter rates in `AdamW`.
2. **Train the slot variant** (`--dataset advisory_slots --extra-split unseen`) and compare
   unsupported-figure rates against the figure-carrying model. The dataset is built and tested; a slot
   answer cannot state a figure at all, so if it costs little fluency it should become the default.
3. **One fair rematch for the price head, then stop regardless of outcome.** A 20-day scale channel
   instead of 128-day, plus the `warmup_cosine` schedule the generator does not use either. If it still
   loses to persistence, that is the reported result and Phase 5 closes as a negative.
4. **Phase 4's remaining substance is tools, not models.** Indicators, retrieval and a persistence-based
   risk estimator supply every fact; the generator only phrases them. This is what makes the small model
   viable and it needs no further training.
5. **Then Phase 6, which is the thesis** — calibration from outcomes, a learned abstention threshold,
   retrieval weighting and candidate ranking, all with frozen weights; LoRA only as a last resort. It
   needs a learning curve **and** a no-adaptation control curve, or it shows nothing.
6. Deferred with a reason, not forgotten: the ~600M-token general-English corpus (the probe says
   vocabulary is not the constraint), the neural reranker (the oracle ablation says it would not pay),
   and the CIK↔ticker map for Phase 6's event-why join (still blocked, and `sec_edgar` is already at its
   50% corpus-share cap so more filings add no text).
