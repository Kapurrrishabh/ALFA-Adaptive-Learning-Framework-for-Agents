# Research: A Self-Learning Financial Agent Built From Scratch

Status: research findings and design rationale. Companion document: `PLAN_OF_ACTION.md`.
Date: 2026-09-06.

---

## 1. What we are actually building

A single agent that:

1. reads **finance news text** and **OHLCV price series**,
2. produces a decision-support output (sentiment, risk flag, direction view) with a calibrated
   confidence,
3. **asks the user** about the cases it is least sure of,
4. **updates itself** from the user's answers and ratings, and
5. keeps a **separate learned profile per user**, so two users teaching it different things do not
   overwrite each other.

Point 5 is the part that makes this a "self-learning agent" rather than a model with a fine-tune
script. It is also the part with the least off-the-shelf precedent, so most of the design risk lives
there.

### Constraints (fixed by the user)

| Constraint | Consequence |
|---|---|
| NumPy, pandas, matplotlib allowed | We write autograd, tokenizer, attention, optimizers, RL loop; pandas does the data layer and matplotlib the plots |
| SciPy/sklearn permitted **for evaluation only** | Enforced by a test, not by good intentions (§6.4) |
| No TensorFlow / Keras / PyTorch / HuggingFace | No pretrained weights — we pretrain our own, small |
| No API keys, no hosted models | All data must be local files; no live market or LLM calls |
| Modular coding standards | Layout in `PLAN_OF_ACTION.md` §3 |
| Laptop CPU + Colab/GPU available | **NumPy does not use the GPU** — see §5.3, this is a trap |

---

## 2. Literature grounding

### 2.1 The encoder

**BERT** (Devlin et al., 2019) is the right base: bidirectional Transformer encoder, masked-language-
model (MLM) pretraining, `[CLS]` pooling for classification. We drop Next-Sentence Prediction —
**RoBERTa** (Liu et al., 2019) showed it does not help, and removing it removes a whole data pipeline.

Small-scale pretraining is where MLM is weakest: only ~15% of tokens produce a learning signal per
step. Two alternatives are more sample-efficient at our scale:

- **ELECTRA** (Clark et al., 2020) — replaced-token detection over *all* positions; reported
  substantially better results at small compute. Cost: a second (generator) network.
- **Whole-word / span masking** (**SpanBERT**, Joshi et al., 2020) — cheap to implement, helps the
  model learn multi-token finance entities ("earnings per share", "Federal Reserve").

**Decision:** implement MLM with whole-word masking. Keep ELECTRA as a documented Phase-3 upgrade if
MLM loss plateaus. Rationale: one network, one loss, and the masking change is ~20 lines.

**Architecture details worth copying, all cheap:**
- **Pre-LayerNorm** blocks (Xiong et al., 2020) — trains stably without a warmup-heavy schedule.
  Post-LN (original BERT) diverges easily at small scale, which we cannot afford to debug.
- **GELU** activation (Hendrycks & Gimpel, 2016) — standard in this family.
- **Learned absolute positions** for v1. Rotary (RoPE, Su et al., 2021) is better but only matters
  for length extrapolation we don't need at 128 tokens.

**Finance-domain precedent:** **FinBERT** (Araci, 2019; and Yang et al., 2020) established that
domain pretraining on financial text beats general BERT on financial sentiment. It also sets our
honest expectation: FinBERT started from a fully pretrained BERT. We are starting from random on a
much smaller corpus, so **we should not expect to match published FinBERT numbers** (§4).

### 2.2 The numeric (OHLCV) branch

The naive approach — one token per timestep — wastes attention on a very long, very smooth sequence.
Two findings matter:

- **PatchTST** (Nie et al., 2023): split the series into **patches** of consecutive timesteps and
  embed each patch as one token. Cuts sequence length by the patch size, improves accuracy, and is
  channel-independent (each of O/H/L/C/V handled as its own series). This is a ~40-line embedding
  layer in front of the *same* Transformer encoder we already wrote — maximum reuse.
- **Are Transformers Effective for Time Series Forecasting?** (Zeng et al., 2023): a one-layer linear
  model (DLinear) beat many Transformer forecasters. Read as a warning, not a veto: **we must run
  the linear/momentum baseline and report it honestly**, or our Transformer numbers mean nothing.

**Decision:** patch embedding + reused encoder, and a mandatory DLinear-style and
momentum/moving-average baseline in `eval/baselines.py`.

### 2.3 Fusing text and prices

Precedent for news+price fusion: **StockNet** (Xu & Cohen, 2018), **HAN** (Hu et al., 2018) for
hierarchical attention over news, and the broader multimodal pattern from **ViLBERT** (Lu et al.,
2019) / **Perceiver** (Jaegle et al., 2021).

Three fusion options, cheapest first:

| Option | Mechanism | Cost | Verdict |
|---|---|---|---|
| Late concat | `[CLS]_text ‖ [CLS]_price` → MLP head | trivial | **v1 baseline** |
| Cross-attention | price tokens attend to news tokens | one extra block | **v2 target** |
| Full early fusion | one shared sequence, modality embeddings | rewrite of batching | not worth it |

**Decision:** ship late concat first so the whole pipeline works end-to-end, then add cross-attention
and *measure whether it actually helps*. If it doesn't, delete it.

### 2.4 Learning from human interaction

Three distinct mechanisms, often conflated. We need all three and they serve different purposes.

**(a) Active learning — deciding *what* to ask.**
Settles (2009) survey; the workhorse methods are uncertainty sampling: least-confidence, **margin**
(gap between top-2 class probabilities), and **entropy**. Deep-model variants add **BALD** /
MC-dropout (Gal & Ghahramani, 2016) for epistemic uncertainty, and **core-set** selection (Sener &
Savarese, 2018) for batch diversity.

**Decision:** margin + entropy for v1 (a dozen lines, no extra forward passes); MC-dropout BALD as an
upgrade since we already own the dropout layer. Diversity filter only if we see the agent asking ten
near-identical questions in a row — a real failure mode of pure uncertainty sampling.

Why this matters for the project story: it converts "the user labels data" into "the agent decides
what it needs to know", which is the difference between a labelling tool and an agent.

**(b) Preference learning — turning ratings into a reward.**
Christiano et al. (2017) established learning a reward model from human *pairwise preferences*; this
became the RLHF recipe in **InstructGPT** (Ouyang et al., 2022): supervised fine-tune → reward model
→ PPO with a KL penalty to the reference policy. The reward model is a **Bradley-Terry** model:

```
P(a preferred over b) = sigmoid( r(a) - r(b) )
loss = -log sigmoid( r(a) - r(b) )
```

Crucially, this only needs *relative* judgements — much easier for a user to give ("this call was
better than that one") than an absolute score, and less noisy.

**Policy optimization choice.** PPO (Schulman et al., 2017) is the published default and also the
most fragile thing in the RLHF stack (clipping, GAE, value function, KL coefficient, all interacting).
Our action space is small and episodes are one step, which changes the calculus:

- **REINFORCE with a learned baseline** (Williams, 1992) is correct for one-step episodes, is ~30
  lines, and has no clipping or GAE to get wrong. Recent work (RLOO, Ahmadian et al., 2024) shows
  simple REINFORCE variants are competitive with PPO for LLM alignment.
- **DPO** (Rafailov et al., 2023) skips the reward model entirely and optimizes preferences as a
  classification loss against a frozen reference policy. Simpler and more stable.

**Decision:** REINFORCE-with-baseline + KL-to-reference as the primary path; **DPO implemented as the
comparison arm**. Two paths here is deliberate and defensible — "we compared reward-model RL against
direct preference optimization under identical feedback" is a genuine experimental result, and DPO is
our fallback if the reward model proves too noisy at low feedback volume. PPO-clip stays documented-
but-unbuilt unless we have a measured reason.

**(c) Continual learning — updating without destroying what was learned.**
Catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999) is the central failure mode of any
agent that keeps training after deployment. A handful of user corrections at a normal learning rate
will visibly wreck a pretrained encoder. Three families of defence:

| Family | Method | Fit for us |
|---|---|---|
| Regularisation | **EWC** (Kirkpatrick et al., 2017): penalise movement of parameters with high Fisher information | Good, cheap (diagonal Fisher) |
| Replay | Rehearse stored old examples alongside new; **reservoir sampling** (Vitter, 1985) for a bounded buffer | **Strongest per unit of effort** (Chaudhry et al., 2019) |
| Parameter isolation | Freeze the backbone; train a tiny per-task module | **Essential for multi-user** |

For parameter isolation the two standards are **bottleneck adapters** (Houlsby et al., 2019) and
**LoRA** (Hu et al., 2021), which adds a low-rank `BA` update to a frozen weight matrix, with `B`
initialised to zero so the adapted model starts exactly equal to the base model.

**Decision — this is the core architectural bet:**

```
frozen shared backbone  (pretrained once, identical for every user)
        +
per-user LoRA adapters  (rank 4-8, ~0.1-1% of params)   ← the only thing that trains online
        +
per-user replay buffer + reward model + EWC penalty
```

Why: storing a full fine-tune per user costs `N_users × 6M params` and each user's training corrupts
the shared model. Adapters make a user's learned state a small, swappable, deletable file. It gives
us a clean answer to "different users adapt it to their own data", and LoRA's zero-init means
onboarding a new user is provably a no-op until they teach it something.

**Instruction/interaction framing:** even without a generative decoder, the interaction loop follows
the InstructGPT shape — predict, explain, collect judgement, update — which is the structure worth
citing in the report.

---

## 3. Design decisions, consolidated

| # | Decision | Reason | Rejected alternative |
|---|---|---|---|
| D1 | One `TransformerEncoder` class, two embedding front-ends (WordPiece tokens; OHLCV patches) | Single source of truth for the attention stack | Separate text and time-series models |
| D2 | Pre-LayerNorm, GELU, learned positions | Stability at small scale; debugging budget is finite | Post-LN (original BERT) |
| D3 | MLM with whole-word masking, no NSP | RoBERTa showed NSP is dead weight; WWM helps finance multi-token entities | ELECTRA (kept as documented upgrade) |
| D4 | Patch embedding for OHLCV | PatchTST; shortens sequence, reuses encoder | One token per timestep |
| D5 | Late concat fusion first, cross-attention second, each measured | End-to-end working system before sophistication | Early fusion |
| D6 | Frozen backbone + per-user LoRA | Multi-user isolation, bounded storage, zero-init safety | Full per-user fine-tune |
| D7 | REINFORCE + baseline + KL, with DPO as comparison arm | One-step episodes don't need PPO's machinery; comparison is a real result | PPO-clip |
| D8 | Replay buffer + diagonal EWC | Forgetting is the #1 online-learning failure | Hope |
| D9 | Walk-forward / purged time splits everywhere | Random splits leak the future — invalidates every number | k-fold on shuffled rows |
| D10 | Finite-difference gradient check as a build gate | A hand-written autograd bug is silent and fatal | Trusting the math |

---

## 4. What we will *not* claim

Stating this up front protects the project from its most likely criticism.

1. **We will not claim to beat pretrained FinBERT or any published LLM.** We have a fraction of the
   pretraining compute and no pretrained initialisation. The contribution is the *self-learning
   architecture and the measured adaptation effect*, not absolute NLP accuracy.
2. **We will not claim to predict prices profitably.** Financial returns are close to unpredictable
   from public data at daily frequency; strong reported accuracy in this area is usually leakage,
   survivorship bias, or ignored transaction costs. We frame the numeric task as *directional view
   with calibrated confidence and risk flagging*, we report against a momentum baseline, and we
   report no backtested P&L unless it survives a purged walk-forward evaluation.
3. **We will not report accuracy from a shuffled split** on time-ordered data.
4. **We will not present the RL result without a KL/degradation check** — reward going up while the
   held-out supervised metric collapses is reward hacking, not learning.

The headline result we *are* going for: **a measured learning curve showing accuracy rising with
rounds of human interaction, beating a no-feedback control and a random-query control, without
degrading performance on the original held-out test set.** That is a defensible, falsifiable claim.

---

## 5. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Autograd bug produces plausible-looking but wrong gradients | **Fatal** | ~~Planned~~ **Closed**: 44 finite-difference tests green, `tests/test_gradcheck.py` |
| R2 | Pretraining too slow on CPU to finish | ~~High~~ **Retired** | Measured: 1.0s/step at 5.3M params, ~1h for a 15M-token run. Compute is not the constraint; see R11 |
| R3 | Catastrophic forgetting during online updates | High | Frozen backbone + LoRA + replay + EWC; forgetting is an explicit reported metric, not an assumption |
| R4 | Reward model overfits tiny feedback volume; reward hacking | High | KL-to-reference penalty, small LR, DPO comparison arm, held-out supervised metric as a tripwire |
| R5 | Look-ahead leakage in news→price alignment | High (silently inflates results) | Point-in-time joins only, market-hours/timezone handling, purge+embargo, explicit leakage test (§6.3) |
| R6 | Class imbalance (most news is neutral; most days are flat) | Medium | Report macro-F1 and per-class, not accuracy; majority-class baseline always shown |
| R7 | Active learning asks ten near-identical questions | Medium | Diversity filter over selected batch |
| R8 | GPU expected to help but NumPy cannot use it | Medium | §5.3 — decide explicitly, don't discover it in week 8 |
| R9 | Scope creep into a full LLM chat agent | Medium | Encoder-only is the committed scope; no generative decoder |
| R10 | Small corpus → 8k vocab still yields many `[UNK]`/over-fragmented tickers | Low | Measure subword fertility in Phase 3; add ticker/number normalisation |
| R11 | **Corpus too small for the model** — replaces R2 as the real capacity limit | High | 15M tokens on 5.3M params is ~3 tokens/param vs BERT-base's ~30. Keep the model at `small`, spend effort on collecting text, and report the no-pretraining ablation so the reader can see what pretraining bought |
| R12 | Accelerate BLAS raises spurious FP flags on float32 `sgemm`, masking real overflows | Medium | Flags muted only at the BLAS call; `backend.set_check_numerics(True)` verifies every op output is finite instead, and is on for the whole test suite |

### 5.3 The GPU problem — decide this in Phase 0

NumPy is CPU-only. A Colab GPU does nothing for a NumPy implementation. The honest options:

- **(a) Accept CPU.** NumPy's `matmul` calls into BLAS (OpenBLAS/Accelerate) and is multithreaded, so
  this is far from hopeless at ~6M parameters. Recommended default.
- **(b) Use Colab's CPU** for long pretraining runs so your laptop stays free. Free, no code change.
- **(c) Add a `backend.py` seam** — every module imports `from selfagent.backend import xp` and never
  imports NumPy directly. Later, `xp = cupy` gives GPU execution with a near-identical API. Costs one
  import discipline rule now, and CuPy is arguably still "not a deep-learning library" — but confirm
  with whoever is grading before relying on it.

**Recommendation: (a) + (b), with the (c) seam in place from day one because it is free if done at the
start and expensive to retrofit.**

---

## 6. Methodology requirements

### 6.1 Data contract

The agent must not care where data came from. Two loaders, two fixed schemas:

```
news.csv   : timestamp (ISO-8601, tz-aware), ticker, headline, body?, source?, label?
ohlcv.csv  : timestamp (ISO-8601, tz-aware), ticker, open, high, low, close, volume
```

Anything else is adapted in `data/`, never downstream. Validate at the boundary once: monotonic
timestamps, no duplicate (ticker, timestamp), non-negative volume, `high >= low`, no NaNs in required
columns. **Fail loudly with the offending row** — a silently dropped bad row becomes an unexplained
result three weeks later.

### 6.2 Splitting

Walk-forward only. For each fold: train on `[t0, t1)`, **purge** the window whose label horizon
overlaps `t1`, apply an **embargo** after `t1`, then test on `[t1, t2)`. (Purging/embargo per López de
Prado, *Advances in Financial Machine Learning*, 2018 — the standard fix for leakage through
overlapping labels.) Same fold boundaries for text, numeric, and fused models so comparisons are
valid.

### 6.3 Leakage test

A dedicated test asserting that for every training example, every input timestamp is strictly earlier
than the label's observation time. This test is the single highest-value test in the project — R5
inflates results without ever failing anything else.

### 6.4 Enforcing the library constraint

A test that walks every `.py` file outside `selfagent/eval/` and fails if it imports `sklearn`,
`scipy`, `torch`, `tensorflow`, `keras`, or `transformers`. The constraint is then machine-checked and
provable to a reviewer, rather than a claim in a README.

### 6.5 Reproducibility

Explicit seed in every config; the config file hash recorded in every checkpoint and results file.
Every number in the final report must be regenerable by one command plus a config path.

---

## 7. Open questions for the user

1. **Data specifics** — row counts, date range, tickers, news source, and what the sentiment labels
   are (3-class? who labelled them?). This sets the model size and whether the price task is even
   learnable. Nothing after Phase 0 can be sized without it.
2. **Is CuPy acceptable** if we want GPU later (§5.3)?
3. **Deadline**, and is this graded as a research report, a working system, or both? It changes how
   much goes into the interaction UX versus the experiments.
4. **How much human feedback can you actually produce?** The RLHF phases are budgeted in labels. 200
   labels and 2000 labels are different designs.
5. **Is sklearn genuinely allowed for baselines** (TF-IDF + logistic regression), or must baselines
   also be hand-written? A hand-written logistic regression is ~40 lines, so this is cheap either way
   — but the answer should be on record.

---

## References

Ahmadian et al. 2024, *Back to Basics: REINFORCE for LLM alignment* · Araci 2019, *FinBERT* ·
Chaudhry et al. 2019, *Tiny episodic memories* · Christiano et al. 2017, *Deep RL from human
preferences* · Clark et al. 2020, *ELECTRA* · Devlin et al. 2019, *BERT* · French 1999, *Catastrophic
forgetting* · Gal & Ghahramani 2016, *Dropout as Bayesian approximation* · Hendrycks & Gimpel 2016,
*GELU* · Houlsby et al. 2019, *Parameter-efficient transfer learning* · Hu et al. 2018, *HAN for
stock prediction* · Hu et al. 2021, *LoRA* · Jaegle et al. 2021, *Perceiver* · Joshi et al. 2020,
*SpanBERT* · Kirkpatrick et al. 2017, *EWC* · Liu et al. 2019, *RoBERTa* · López de Prado 2018,
*Advances in Financial Machine Learning* · Lu et al. 2019, *ViLBERT* · McCloskey & Cohen 1989 · Nie
et al. 2023, *PatchTST* · Ouyang et al. 2022, *InstructGPT* · Rafailov et al. 2023, *DPO* · Schulman
et al. 2017, *PPO* · Sener & Savarese 2018, *Core-set active learning* · Settles 2009, *Active
learning survey* · Su et al. 2021, *RoFormer/RoPE* · Vaswani et al. 2017, *Attention Is All You Need*
· Vitter 1985, *Reservoir sampling* · Williams 1992, *REINFORCE* · Xiong et al. 2020, *On Layer
Normalization in the Transformer* · Xu & Cohen 2018, *StockNet* · Yang et al. 2020, *FinBERT* · Zeng
et al. 2023, *Are Transformers Effective for Time Series Forecasting?*
