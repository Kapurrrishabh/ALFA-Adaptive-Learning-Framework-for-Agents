# LEARNME — everything in this project, in plain English

Viva preparation. Read top to bottom once; the last section is a list of likely questions with the
answers. Every number here was measured in this repo, and where something is **not** built or **did
not work**, it says so — examiners reward that far more than a clean story.

---

## 1. What the project is

**Research question:** *can an agent improve its decisions and guidance from interaction, outcomes and
feedback, without continuously retraining its underlying foundation model?*

The idea being tested is that most of what looks like "the model getting smarter" does not need
gradient descent on the big model at all. It can come from things around the model: better retrieval,
better calibration, learned thresholds for when to stay quiet, and ranking of candidate answers. The
foundation model stays frozen and only supplies **language**.

The product we build to test it is a **conversational stock-advisory agent**: you type a question in
your own words about a stock, and it answers using figures it actually computed, or it refuses.

Two packages, deliberately separate:

| Package | Job | Rule |
|---|---|---|
| `dataforge/` | Collect and clean raw text and prices from the internet | Must stay **deletable** — nothing in `selfagent/` imports it |
| `selfagent/` | The models, all in pure NumPy | No TensorFlow or PyTorch in the training path |

### Why pure NumPy

Every layer, the autograd engine, attention, the optimiser — all hand-written on NumPy. No framework.
This is the single strongest thing to defend in a viva, because it proves you understand the
mathematics rather than the API. It costs speed (about 5 training steps per second), which is why the
models are small.

---

## 2. The architecture in one picture

```
                            YOUR QUESTION
                    "does AAPL look overheated to you?"
                                  |
        +-------------------------+-------------------------+
        |                                                   |
   PRICE HISTORY (CSV bars)                          TEXT CORPUS (100.8M tokens)
        |                                                   |
        v                                                   v
  indicators.py  (pure arithmetic, no learning)        WordPiece tokenizer
  RSI, SMA, volatility, drawdown, ATR, returns          (8,000 pieces)
        |                                                   |
        |                                                   v
        |                                          +------------------+
        |                                          |   TextTower      |  <-- pretrained by MLM
        |                                          |  4-layer         |      5.24M params
        |                                          |  transformer     |
        |                                          |  encoder         |
        |                                          +------------------+
        v                                                   |
  +---------------+                                         |
  | PriceTower    |  (transformer)  1.61M                    |
  |     OR        |                                          |
  | RecurrentPrice|  (GRU)          0.20M  <-- GRU won       |
  +---------------+                                          |
        |                                                    |
        v                                                    |
  risk outlook: calm / normal / turbulent + confidence        |
        |                                                    |
        +----------------> EVIDENCE PASSAGE <----------------+
                  "ticker AAPL ; close 182.50 ; rsi_14 62 ;
                   volatility_20d 21.3% ; outlook calm ..."
                                  |
                                  v
                    +---------------------------+
                    |    GroundedGenerator      |  7.47M params
                    |  encoder (shared) +       |
                    |  2-layer decoder with     |
                    |  CROSS-ATTENTION over     |
                    |  question + evidence      |
                    +---------------------------+
                                  |
                                  v
                          guardrails.screen()
              every figure in the answer must appear in the evidence,
                    otherwise the answer is replaced by a refusal
                                  |
                                  v
                            ANSWER TO USER
```

The one sentence version: **the arithmetic knows the facts, the neural network only knows the
wording, and a guardrail checks the wording did not invent a number.**

---

## 3. The models, one at a time

### 3.1 WordPiece tokenizer — `selfagent/tokenizer/`

**What it is.** Text must become integers. WordPiece starts from single characters and repeatedly
merges the most frequent adjacent pair into a new "piece", 8,000 pieces in total. Common words end up
as one piece; rare words break into fragments (`demat` → `dem ##at`).

**Why not just words?** A word-level vocabulary cannot spell a word it never saw, and finance is full
of tickers and rare terms. WordPiece can always fall back to characters, so nothing is unrepresentable.

**Viva point.** The normalisation rules (lowercasing, how to treat `$`, `%`, digits) were each **counted
in the corpus before being kept**. Measurement dropped more rules than it kept and overruled intuition
three times. That is the method the whole project uses: never keep a design choice you did not measure.

### 3.2 TextTower — `selfagent/models/agent.py`, `selfagent/nn/encoder.py`

**What it is.** A 4-layer transformer **encoder**. Each layer is: multi-head self-attention (4 heads),
then a feed-forward network, each wrapped in layer normalisation and a residual connection. Width
(`dim`) 256, FFN width 1024, input up to 128 tokens.

**Pre-LN, not post-LN.** Layer norm goes *before* each sub-layer. This is what lets a deep stack train
without a warmup trick — the residual path stays clean.

**Self-attention in words.** Every token produces a Query, a Key and a Value. The token's new
representation is a weighted average of all Values, where the weight is how well its Query matches each
Key. So "it" can pull meaning from "AAPL" five words earlier. Multi-head means doing this 4 times in
parallel on 64 dimensions each, so different heads can track different relationships.

**Parameters: 5,240,832.** Note that the token embedding alone (8,000 × 256) is 2.05M — **39% of the
model**. Shrinking the vocabulary is the cheapest way to shrink this model.

### 3.3 MaskedLanguageModel — pretraining

**What it is.** The task that teaches the encoder English. Hide about 15% of the tokens and make the
model predict them from both sides of the gap. **Whole-word masking**: if a word is split into three
pieces, all three are hidden, so the model cannot cheat by reading `##ing` to guess `runn`.

The output head's weights are **tied** to the input embedding — the same matrix reads tokens in and
scores them out. Saves 2.05M parameters and is standard practice.

**Parameters: 5,315,136.**

**Measured result (this is a real, passed gate):**

| Metric | Value |
|---|---|
| Validation loss | **3.03** |
| A uniform guess (`ln 8000`) | 8.99 |
| Perplexity | **20.7** |
| Corpus | 100.8M tokens |
| Steps | 48,000 |

Perplexity 20.7 means that on average the model has narrowed 8,000 choices down to about 21. It
genuinely learned finance language.

**The honest weakness.** A probe on *number* pieces scores only **29.8%** — against 10% chance, so it
did learn something about the magnitude and format of numbers, but it is weak. This matters later, and
it is one reason the final design does not let the model produce digits at all.

### 3.4 PriceTower and RecurrentPriceTower — the numeric side

Two competing designs for reading price history, built deliberately so the numbers could decide
between them ("build both, let the numbers decide").

**Input.** A 128-day window, 5 channels (open, high, low, close, volume), cut into overlapping
**patches** of 16 days with stride 8. Patching is the standard trick for time-series transformers: it
shortens the sequence and lets attention work on short trends instead of single days.

- **`PriceTower`** — a transformer over those patches. **1,605,120 params.**
- **`RecurrentPriceTower`** — a **GRU** (Gated Recurrent Unit) instead. A GRU walks the sequence one
  step at a time carrying a hidden state, with two learned gates: an *update* gate deciding how much
  of the old state to keep, and a *reset* gate deciding how much of it to use when forming the new
  candidate. **201,216 params** — 8× smaller.

**`PriceWindowClassifier`** is the tower plus a classification head. **1,605,891 params.**

**Measured result — and this is a negative result, which you must present as a finding, not a failure.**

Tested on 101 tickers, 143,546 training and 28,676 held-out windows, split by date at 2021-01-01,
6,000 steps each:

| Task | Majority baseline | Persistence baseline | Transformer | GRU |
|---|---|---|---|---|
| 5-day **direction** (up/flat/down) | 36.0% | — | **35.7%** | 33.5% |
| 5-day **volatility** bucket | 40.0% | **47.0%** | 45.7% | 46.9% |

Two conclusions:

1. **Direction is dead.** 35.7% against a 36.0% baseline, with the loss flat at `ln 3` for the whole
   run. The model learned nothing. This is not a bug — short-horizon direction is close to
   unpredictable, and the correct response is to **never give directional advice**. That decision is
   now hardcoded into the agent's answers.
2. **Volatility is learnable but not beatable.** The bar is *persistence* — literally
   `np.diff(np.log(closes[-21:])).std()`, one line of arithmetic — which scores 47.0%. The GRU gets
   46.9%. It ties one line of arithmetic. So the neural price head has **not earned its place**.

**The critical viva point about honest baselines.** The naive comparison is against the majority class
(40.0%), which the model "beats" by 7 points. That comparison is misleading, because volatility
persists — tomorrow's volatility looks like today's. Any model that merely rediscovers persistence has
added nothing. **Always report against the strongest cheap baseline, not the weakest.**

**A bug worth telling the examiner about.** Before this was measured properly, the conclusion was "the
model cannot learn volatility". The real cause: `features.standardize` z-scores each window
independently, which **divided the trailing volatility out of the very input meant to predict it**. The
lesson: before believing "the model cannot learn X", check that X is still present in the input.

### 3.5 GroundedGenerator — the model that talks

**What it is.** The full encoder–decoder, and the centre of the project. **7,472,192 params.**

- The **encoder** is the TextTower, warm-started from the pretrained MLM checkpoint.
- The **decoder** is 2 layers. Each layer does three things in order:
  1. **Causal self-attention** over the answer written so far — causal meaning position 5 cannot see
     position 6, enforced by a mask of `-inf`. Without it the model sees the answer while predicting
     it and trains to a beautiful loss curve then produces nonsense at generation time.
  2. **Cross-attention** into the encoder output — this is the only route by which evidence reaches
     the answer.
  3. A feed-forward network.

**Fusion-in-Decoder (FiD).** The question and each evidence passage are encoded, then all their token
states are flattened into one long memory of shape `(batch, passages × length, dim)` that the decoder
cross-attends over. So the decoder sees *every token of every passage*, not one summary vector per
passage. That is what makes copying an exact figure possible.

**Generation.** `generate()` decodes one token at a time. Not greedy — greedy decoding repeated **73%
of its own 4-grams** where human answers repeat 1.7%. Nucleus (top-p) sampling at temperature 0.9 and
top-p 0.9 fixed it, measuring 1.6% invented words.

### 3.6 BM25 retriever — `selfagent/models/retriever.py`

**Not** a neural model, and that is on purpose. BM25 is classical lexical search over an inverted
index: it scores a passage by how many of the query's rare terms it contains, with `k1 = 1.5` bounding
how much repeating a term helps and `b = 0.75` penalising long passages.

**Why it is the right first stage, not a placeholder.** It needs no trained weights, so it works
before the encoder does, and finance questions carry exactly the rare terms BM25 is strongest on — a
ticker, "demat", "1099-B", "repo corridor". Scoring walks posting lists rather than looping over
passages; with 117k passages and 62k queries a per-passage Python loop was the difference between
minutes and hours.

**A neural reranker was explicitly deprioritised** — see §5.2 for why the measurement killed it.

### 3.7 Guardrails — `selfagent/agent/guardrails.py`

Not a model. A regular expression and a search. `figures(text)` lists every number an answer states;
`unsupported_figures(answer, evidence)` returns the ones the evidence does not contain; `screen()`
replaces the whole answer with a refusal if any exist.

This is the **verification layer**, and it is why the system can be trusted. The neural network is
never the last word on a number.

---

## 4. How the pieces connect — the data flow

1. **`dataforge/`** downloads text and prices, respecting robots.txt, one request at a time per host.
2. **`prepare_pretrain.py`** tokenises the corpus into one big integer array.
3. **`train_pretrain.py`** trains the MLM → `artifacts/pretrained.npz`. *This is the frozen foundation.*
4. **`train_price_head.py`** trains the price tower → risk outlook probabilities.
5. **`prepare_advisory.py`** builds the advisory dataset. For each snapshot of each ticker it computes
   the indicators, renders them as an evidence passage, and writes one (question, evidence, answer)
   row per intent.
6. **`train_generator.py`** trains the GroundedGenerator, encoder warm-started from step 3.
7. **Serving:** question → indicators + risk outlook → evidence → generator → `guardrails.screen()`.

### The seven intents

`performance`, `overbought`, `volatility`, `drawdown`, `risk`, `buy`, `unsupported`.

- **`buy` always abstains.** Because direction measured 35.7% against a 36.0% baseline, any confident
  buy/sell wording would be dressing a coin toss as a view. The answer says so and still quotes the
  figures it does have.
- **`unsupported` always refuses.** These rows ask for things the evidence cannot hold — earnings, the
  CEO, the P/E ratio. They exist because a model trained only on answerable rows **assumes the evidence
  always contains the answer and fabricates when it does not**: training without them moved perplexity
  on ungrounded questions from 60 to 310.

### The three splits — the most important design decision in the project

Most projects have train and validation. This has **three**:

| Split | Differs from training by | What it catches |
|---|---|---|
| `train` | — | — |
| `validation` | **the date** | ordinary overfitting |
| `unseen` | **the wording of the question** | whether it learned the task or the templates |

Validation and unseen are built from the *same snapshot and the same facts*, differing only in how the
question is phrased. That isolation is what made the real failure visible — see §5.3.

---

## 5. The measured results, including what failed

### 5.1 First attempt at the generator: fluent and unfaithful

Trained on Stack Exchange question/answer threads. Validation perplexity 63.1 — respectable. Then an
**ablation**: retrain nothing, just corrupt the evidence at evaluation time and see whether the loss
notices.

| Ablation | Perplexity | Cost |
|---|---|---|
| Real passages | 63.1 | — |
| **All passages hidden** | 64.2 | **+0.018** |
| True answer handed over as a passage | 59.9 | −0.051 |
| Question *and* passages hidden | 74.8 | +0.170 |

**Read this table carefully — it is the best single piece of evidence in the project.** Hiding every
passage costs 0.018, which is nothing. Handing the model the verbatim answer buys only 0.051, where a
working copy mechanism would drop the loss to near zero. So the decoder was ignoring its evidence: it
was ~95% an unconditional language model that had learned to write plausible finance prose.

Cross-attention was wired correctly — blinding the whole encoder *does* cost 0.170 — so this was a
learning failure, not a bug.

**Why it happened, and this is the real lesson.** Only **30.7%** of an answer's content words appeared
anywhere in the input. 69% of what the answer had to say was invisible. **The task was unlearnable as
posed**, and generic prose was genuinely the loss-minimising solution. Evidence that is useless in most
rows makes ignoring evidence optimal; having ignored it, the model never builds a copy path, so it
cannot use good evidence even when it arrives.

### 5.2 The fix: make every answer reachable by copying

The advisory dataset was designed so that **every figure in every answer appears in the evidence in the
same characters**. One format per fact, applied once — a number rendered two ways cannot be copied and
cannot be checked.

Then the same ablation on the new, converged model:

| Condition | Loss | Cost vs as-given |
|---|---|---|
| As given | 0.0073 | — |
| **Evidence blanked** | 1.0493 | **+1.0420** |
| **Evidence swapped for another row's** | 1.9537 | **+1.9464** |

Blanking now costs **58× what it cost on the old task** (+1.04 vs +0.018). And note the **ordering**:
swapping costs *more* than blanking. That is the signature of a model genuinely reading its evidence —
a model reciting a memorised shape still scores well with no evidence at all, but cannot score well
while staring at the *wrong* numbers. Early in training the ordering was reversed; the inversion is the
copy circuit forming.

**Grounding works.** Also measured: 100% abstention on questions the evidence cannot answer, with 0%
false refusals.

This also **killed the neural-reranker plan**. Perfect retrieval buys 0.05 while the decoder cannot
copy, so retrieval quality was never the binding constraint. Two of my earlier conclusions were wrong
and should not be revived.

### 5.3 The failure that is still open: reworded questions

With the grounding fixed, the three-way split exposed the next problem:

| Split | Loss |
|---|---|
| Train | 0.0022 |
| Validation (new date) | 0.011 |
| **Unseen (new wording)** | **0.70, and rising** |

A **62× gap**, and the unseen split was *getting worse*. Generate-and-check confirmed it: 45.8% exact
match on trained phrasings, **12.5%** on reworded ones — and the errors were not clumsy wording, they
were **wrong intent**. "Does BAJAJFINSV look extended to you?" (an RSI question) came back with the
*buy refusal*.

Validation alone would have hidden this completely. **This is the argument for the third split.**

### 5.4 Locating the cause with zero training compute

Rather than guess, `scripts/probe_intents.py` measures the encoder directly with **no training at
all**: embed every phrasing of every intent, and ask whether its nearest neighbour among the phrasings
*training saw* shares its intent.

Then hold out two different things, because "paraphrase" hides two very different difficulties:

- a reserved **word** — a synonym appearing in no training question, in a sentence shape training knows
- a reserved **frame** — a sentence shape training never saw, built only from words it knows
- and the **intersection**, which is what a real user actually types

On the **pretrained encoder, before any task training**:

| What is novel | Placed correctly |
|---|---|
| Nothing (trained phrasings) | 108/127 (85%) |
| **Word only** | **15/15 (100%)** |
| **Frame only** | 28/41 (68%) |
| **Word and frame** | 2/7 (29%), against 1/7 by chance |

**So the vocabulary is not the problem — the sentence shape is.** The encoder places a synonym it has
never read perfectly well, as long as the shape around it is familiar.

And task fine-tuning makes it **worse**:

| What is novel | Pretrained | After fine-tuning |
|---|---|---|
| Trained phrasings | 85% | 89% |
| Word only | 100% | 93% |
| **Frame only** | **68%** | **32%** |
| Word and frame | 29% | 14% (chance) |

Five phrasings per intent bought memorisation **at the cost of the general language the pretrained
encoder already had**. This is catastrophic forgetting, measured.

**Why this result is valuable:** it defers a very large piece of work. The obvious next move was to
collect a ~600M-token general-English corpus. This measurement says the corpus is *not* the binding
constraint — sentence-shape diversity in the task data is, and frames are cheap to write.

### 5.5 Two measurement mistakes I made, and what they teach

Both are worth telling an examiner, because the method that caught them is the point.

**(a) Judging a model before the phase transition.** At step 3,000 generate-and-check said the copy
path did not exist: 0% exact match, an invented figure in 100% of answers. At step 4,000 it said 45.8%
exact match and 10.5% unsupported figures. Training loss sat at **0.28 through step 3,000 then
collapsed to 0.014 by 5,400**.

The 0.28 was not a floor — it was the entropy of each field's **marginal prior**. The model was emitting
the *distribution* of `close` rather than *this row's* `close`. The tell: narrow fields look nearly
right (RSI 56 → 62) while wide ones are absurd (close 309.93 → 32.12). The arithmetic predicts it:
boilerplate tokens near zero loss plus ~8 digit tokens at field-prior entropy (~1.2 nats) over ~35
tokens ≈ 0.29. **Copying emerges abruptly.** Never conclude "the architecture cannot do X" from a run
still sitting on its prior-entropy plateau.

**(b) A probe that scored 90% on an untrained model.** After expanding to 190 phrasings, the probe
reported 90% intent separation for an encoder with *no task training* — and 100% on every held-out
cell. That was a broken measurement, not a result. Nothing stopped a reserved phrasing from matching
*another reserved phrasing*: `i am wondering if X has been torrid` found `... has been grim`, one word
apart and both unseen. Restricting neighbours to phrasings training actually saw produced the coherent
gradient in §5.4. **A suspiciously good number is a bug until proven otherwise.**

### 5.6 The slot path — making faithfulness structural

Even working, the copy path slipped a digit: it answered **+3.3% where the evidence said +3.7%**. A
plausible, unfalsifiable, wrong number in front of someone about to trade — the worst possible failure
mode, because it looks right.

The fix removes digits from the model's job entirely. With `--slots`, answers **name the fact** —
"the 20 day move is `return_20d`" — and serving substitutes the value from the evidence already shown.
Faithfulness becomes **structural rather than a rate to monitor**: a slot answer states no figure of its
own, so nothing it writes *can* be fabricated.

Verified lossless: filling a slot answer reproduces the figure answer character-for-character across
every intent and phrasing, and through a full tokenizer encode→decode→fill round trip.

Two bugs caught by design review *before* any training:
- The word "close" in the prose `"close to a coin toss"` collided with the evidence key `close`, and
  would have rendered as `"182.50 to a coin toss"` in **every** served answer. Reworded, and pinned by
  a test.
- `outlook` is a prefix of `outlook_confidence`, so naive replacement corrupted it to
  `turbulent_confidence`. Fixed by substituting longest name first.

Dataset built (321,279 rows, geometry identical to the copy set so the two are directly comparable).
**Not yet trained** — deliberately, because it would hit the same phrasing wall.

### 5.7 Current state

`selfagent/data/advisory.py` now builds questions from **frames × interchangeable words** instead of
one hand-written string each: **190 phrasings**, up from 46, of which 127 are trained and 63 reserved
across the three novelty cells. The dataset rebuilt to **19,183 distinct questions**, up from 4,646,
with the row count unchanged. A training run is in progress to answer whether the held-out gap closes,
with the training procedure held identical so phrasing count is the only variable.

**234 tests pass.**

---

## 6. Why this design, versus the alternatives

### 6.1 Why not just call GPT-4 / an LLM API?

This is the first question an examiner will ask. Answers, strongest first:

1. **It would not answer the research question.** The thesis is about improving *without* retraining the
   foundation model. If the foundation model is a black box behind an API you cannot measure what it
   already knew, cannot freeze it, and cannot attribute an improvement to the adaptation layer rather
   than to the vendor changing the model underneath you.
2. **Measurability.** Because we pretrained the encoder ourselves we can say exactly what it saw. The
   ablations in §5.1 and the probe in §5.4 are only interpretable because of that. You cannot run
   "blank the evidence and see if the loss notices" on a model you did not train.
3. **Cost and constraint.** The project uses no API keys anywhere, by design.

### 6.2 Where this loses honestly

| Against | We lose because |
|---|---|
| A real LLM | 7.5M parameters versus hundreds of billions. Fluency and paraphrase understanding are not comparable, and §5.4 measures exactly where ours breaks |
| A tuned gradient-boosted model on price features | Our price head ties a one-line persistence baseline (§3.4). A GBM with good features would likely beat it |
| A production RAG stack | No trained dense retriever, no reranker, no vector index |

**Say this plainly in the viva.** The value is not "our model is best". It is that **every claim is
measured against an honest baseline, and the negative results are reported**. A project that says "the
neural price head ties one line of arithmetic, so it has not earned its place" is more credible than
one claiming to beat the market.

### 6.3 Why it is genuinely good as research

1. **Falsifiable gates.** Each phase has a numbered success criterion (S3, S4, S13, S14…) fixed *before*
   the measurement, so results cannot be rationalised afterwards.
2. **Ablation as standard practice.** The 63.1-perplexity generator looked fine and was fundamentally
   broken. Only ablation revealed it. That methodology generalises far beyond this project.
3. **The three-way split.** A contribution in its own right — validation split by date would have
   passed a model that fails on any rewording.
4. **Structural faithfulness.** The slot path makes fabrication *impossible* rather than *unlikely*. For
   any system giving financial guidance that is the correct design.
5. **Negative results reported.** Direction is unpredictable; the price head does not beat persistence;
   the reranker was not worth building. Each saved effort and each is defensible.

---

## 7. Every folder and file, and what each function does

### `selfagent/autograd/` — the automatic differentiation engine

The mathematical heart. Build this understanding first; everything else sits on it.

**`tensor.py`**
- `Tensor` — an array plus its gradient plus a record of the operation that produced it.
- `_topological_order` — sorts operations so backprop visits each node only after everything that
  depends on it. This is the chain rule as graph traversal.
- `no_grad` / `grad_enabled` — switch recording off for evaluation, saving memory and time.

**`ops.py`** — each function computes a result *and* registers how to push gradient back:
`add`, `sub`, `mul`, `div`, `neg`, `matmul`, `sum`, `mean`, `exp`, `log`, `sqrt`, `tanh`, `reshape`,
`transpose`, `concat`, `getitem`.
- `_unbroadcast` — the subtle one. If a `(4,1)` tensor was broadcast to `(4,8)` in the forward pass,
  its gradient must be **summed back down** to `(4,1)`. Get this wrong and gradients are silently 8×
  too large.

**`functional.py`** — `sigmoid`, `softmax`, `gelu`, `layer_norm`, `cross_entropy`, `dropout`,
`embedding`. `softmax` subtracts the row max before exponentiating, otherwise `exp` overflows.

### `selfagent/nn/` — the layers

- **`module.py`** — `Module` base class (collects parameters, `train()`/`eval()` modes),
  `ModuleList` for stacks of layers.
- **`layers.py`** — `Linear`, `Embedding`, `LayerNorm`, `Dropout`.
- **`attention.py`** — `MultiHeadAttention`; `padding_mask` hides `[PAD]` positions; `causal_mask`
  stops a position seeing the future.
- **`encoder.py`** — `TransformerBlock` (pre-LN self-attention + FFN), `TransformerEncoder`.
- **`decoder.py`** — `TransformerDecoderBlock` (causal self-attention → **cross-attention** → FFN),
  `TransformerDecoder`.
- **`recurrent.py`** — `GRU`, with `_gate` for the update/reset gates.
- **`init.py`** — `normal`, `zeros`, `ones`. Initial weight scale decides whether a deep stack trains.

### `selfagent/optim/` — training machinery

- **`adamw.py`** — `AdamW`. Adam keeps a running mean and variance of each gradient so every parameter
  gets its own step size. The **W** is decoupled weight decay: shrink weights directly rather than
  through the gradient, which is the mathematically correct form.
- **`clip.py`** — `clip_grad_norm`, rescales the gradient if its norm is too large. Cheap insurance
  against one bad batch destroying a run.
- **`schedule.py`** — `warmup_cosine`. Ramp the learning rate up, then decay it along a cosine. Warmup
  matters because Adam's variance estimates are unreliable in the first few hundred steps.

### `selfagent/tokenizer/`

- **`normalize.py`** — `normalize`, the text-cleaning rules, each counted in the corpus before keeping.
- **`vocab.py`** — the special token IDs: `PAD_ID`, `CLS_ID`, `SEP_ID`, `MASK_ID`.
- **`wordpiece.py`** — `pretokenize` (split on whitespace/punctuation), `count_words`,
  `_frequent_characters`, `_learn_pieces` (the merge loop that builds the vocabulary),
  `_pairs`/`_apply_merge` (merge mechanics), and `WordPiece` with `encode`/`decode`/`save`/`load`.

### `selfagent/data/`

- **`indicators.py`** — pure arithmetic, no learning: `simple_moving_average`, `total_return`,
  `realized_volatility`, `max_drawdown`, `relative_strength_index`, `average_true_range`. `_checked`
  raises when there is not enough history instead of silently returning something wrong. **These are
  the facts the agent states.**
- **`prices.py`** — `load_bars`, `sample_ends`, `window_at`, `label_at`, `forward_volatility`,
  `tertile_edges`, `to_tertile` (bucket boundaries computed on the **training period only**, never the
  future).
- **`features.py`** — `to_features`, `standardize`, `forward_return`. Site of the z-scoring bug in §3.4.
- **`masking.py`** — `word_starts`, `mask_tokens`, `_spread_from_starts`, `_force_one_position` (a batch
  with nothing masked produces no learning signal).
- **`qa_pairs.py`** — `load_threads`, `question_text`, `truncate_to_sentences`, `best_answer`,
  `supervised_pairs`, `preference_pairs` (the last for future preference learning).
- **`advisory.py`** — **the most important data file.**
  - `snapshot(bars, end)` — every fact as of bar `end`. Reads no later bar, so a window cannot see its
    own future.
  - `as_text(facts)` — one format per fact, applied once, so evidence and answer share characters.
  - `render_evidence(ticker, shown)` — the flat `name value ;` passage the encoder reads.
  - `risk_outlook(probabilities)` — the price head's own call as evidence lines, including confidence,
    so the answer quotes it instead of inventing it.
  - `slot_names(shown)` / `fill(text, shown)` — the slot path of §5.6. `fill` substitutes longest name
    first and matches names piece-by-piece, because tokenising turns `return_20d` into `return _ 20 d`.
  - `_FRAMES` / `_WORDS` / `_HELD_OUT_FRAMES` / `_HELD_OUT_WORDS` / `_build` — questions composed from
    frames × words, with both axes reservable.
  - `phrasings`, `is_held_out`, `novelty`, `ask`, `row` — the public surface. `novelty` reports which
    axis is new, which is what makes §5.4's table readable.
  - `_performance`, `_overbought`, `_volatility`, `_drawdown`, `_risk`, `_buy`, `_unsupported` — the
    answer forms, several per intent, rotating independently of the question wording so the model
    cannot learn that one wording implies one answer.

### `selfagent/models/`

- **`embeddings.py`** — `TextEmbedding` (tokens + positions), `AnswerEmbedding` (the decoder side),
  `PricePatchEmbedding` (the 16-day patching).
- **`agent.py`** — `TextTower`, `PriceTower`, `RecurrentPriceTower`, `PriceWindowClassifier`,
  `MaskedLanguageModel`, `FinanceAgentModel` (both towers joined, 6,847,491 params).
- **`heads.py`** — `MaskedLanguageModelHead` (weights tied to the embedding), `ClassificationHead`,
  `pool_cls`.
- **`generator.py`** — `GroundedGenerator`, with `generate()` and the measured decoding constants
  `ANSWER_TEMPERATURE` / `ANSWER_TOP_P` = 0.9 / 0.9.
- **`retriever.py`** — `BM25`.

**A trap worth knowing.** `pool_cls` exists but **do not use it on an MLM-pretrained encoder**. Masked
language modelling never gives `[CLS]` a job, so it comes out near-constant — every pair of *different*
questions scored above 0.97 similarity. Mean-pool over the real tokens instead. The decoder
cross-attends to all token states anyway, so that is the representation it actually works from.

### `selfagent/agent/`

- **`guardrails.py`** — `figures`, `unsupported_figures`, `_states`, `screen`. The verification layer.

### `selfagent/` root

- **`config.py`** — `ModelConfig`, one source of truth for every dimension (§ values in table below).
- **`backend.py`** — `dtype`, `set_dtype`, `default_rng`, `check_numerics`, `set_check_numerics`.
- **`pretrained.py`** — `save` / `load` for checkpoints.

### `scripts/`

| Script | What it does |
|---|---|
| `prepare_pretrain.py` | Corpus → token array for MLM |
| `train_pretrain.py` | Trains the MLM → `pretrained.npz` |
| `prepare_generator.py` | The Stack Exchange generator dataset; `build_sources` packs question + passages |
| `train_generator.py` | Trains the GroundedGenerator; `load_split` is reused by the eval scripts |
| `train_price_head.py` | Trains the price tower against the baselines |
| `prepare_advisory.py` | Builds the advisory dataset; `--slots` for the slot variant |
| `ablate_generator.py` | **Does the decoder read its evidence?** Blanks and swaps it and reports the cost. `evidence_starts`, `blanked`, `swapped`, `answerable` |
| `check_answers.py` | **Generate-and-check** — exact match, unsupported figures, abstention, false refusal, repeated 4-grams |
| `probe_intents.py` | **Does the encoder know two wordings mean the same thing?** No training needed |
| `benchmark_step.py` | Steps per second, for sizing decisions |
| `export_model.py` | Checkpoint → portable form |

**Why `check_answers.py` exists at all.** Perplexity on this data is flattering: the answers are
templated, so most positions are boilerplate a model can predict without reading anything, and the
handful of digit positions that actually need the evidence are lost in the average. Generate-and-check
is the number that decides whether the system works.

### `tests/` — 234 passing

- **`test_gradcheck.py`** — every operation's analytic gradient against a numerical finite-difference
  gradient. **The most important tests in the repo**: a wrong derivative still trains to a plausible
  loss curve, so nothing else would catch it. The decoder gradcheck covers the **memory** input as well
  as the sequence input, because a decoder that passes no gradient back through cross-attention leaves
  the encoder untrained while the loss still falls — it just learns an unconditional prior.
- **`test_model.py`** — shapes, masking, and the causal-leakage test. Verified by *removing* the mask
  and watching the test fail: the leak is only ~1e-2, exactly the size that trains to a plausible curve
  and then generates nonsense.
- **`test_faithfulness.py`** — every phrasing of every intent passes its own guardrail; held-out
  phrasings exist and are a minority; a reserved word appears in **no** trained question of **any**
  intent (the leak that would invalidate the experiment in silence — verified by injecting a leak and
  confirming the test fails); each novelty axis has phrasings to measure; slot filling is lossless;
  prose that merely looks like a field name is left alone; article agreement across frames × words.
- **`test_import_guard.py`** — fails if TensorFlow, Keras or PyTorch is imported in the training path.
- **`conftest.py`** — shared fixtures.

### `dataforge/` — data collection (deletable by design)

**Politeness policy, non-negotiable:** robots.txt honoured per host before any request; one request at
a time per host with a rate floor; a 403 or 429 **aborts the source loudly** rather than retrying; only
public bulk endpoints, official APIs and RSS — no login walls, no paywalls, no CAPTCHA circumvention;
every file's URL and licence recorded in `data/manifest.jsonl`; a real contact email in
`DATAFORGE_CONTACT` or the scripts refuse to run. One documented exception (yfinance) is noted in that
module's docstring.

- **`collect.py`** — `main`, `_run_source`, `log`. The entry point.
- **`config.py`** — `tier`, `ensure_directories`. Source priorities and layout.
- **`common/http.py`** — `PoliteSession` (rate limiting, robots, encoding), `BlockedByHost`,
  `user_agent`.
- **`common/manifest.py`** — `Manifest`, the provenance record.
- **`extract/`** — `html_text.extract`, `pdf_text.extract`, `sec_text.extract` (pulls the primary
  filing document and drops exhibits and hidden taxonomy headers).
- **`process/clean.py`** — `clean`, `normalize`, `_is_prose`, `_letter_ratio`, `_latin_ratio`. Keeps
  prose, drops numeric table rows, keeps the English half of a bilingual circular.
- **`process/dedup.py`** — `ParagraphFilter`, `NearDuplicateIndex`, `_shingle_hashes`. MinHash-style
  near-duplicate removal; boilerplate repeated across filings would otherwise dominate the corpus.
- **`process/pipeline.py`** — `run`, `_extract`, `_cached_text`, `_budget`, `_stride`,
  `_average_cleaned_characters`. Budgets by **cleaned** size, not downloaded size.
- **`process/report.py`** — `summarize`, `render`, `_price_summary`.
- **`sources/`** — `wikipedia`, `sec_edgar`, `gutenberg`, `stackexchange`, `openstax`, `ncert`, `rbi`,
  `sebi`, `fed_press`, `news_rss`, `yahoo_prices`. Each exposes `collect()`.

### `docs/` and `configs/`

- `docs/PLAN_OF_ACTION.md` — phases, the numbered gates, and measured results.
- `docs/AGENT_ARCHITECTURE.md`, `docs/RESEARCH.md` — design and background.
- `configs/tiny.json`, `configs/target.json` — the smoke-test and real model sizes.

---

## 8. Numbers to have memorised

**Configuration** (`selfagent/config.py`):

| Field | Value |
|---|---|
| `vocab_size` | 8,000 |
| `dim` | 256 |
| `num_heads` | 4 (so 64 dims per head) |
| `num_layers` | 4 (encoder) |
| `decoder_layers` | 2 |
| `ffn_dim` | 1,024 |
| `max_text_length` | 128 |
| `max_answer_length` | 192 |
| `dropout` | 0.1 |
| `price_window` / `price_channels` | 128 days / 5 |
| `patch_length` / `patch_stride` | 16 / 8 |
| `num_classes` | 3 |
| `seed` | 0 |

**Parameter counts:**

| Model | Parameters |
|---|---|
| `RecurrentPriceTower` (GRU) | 201,216 |
| `PriceTower` (transformer) | 1,605,120 |
| `PriceWindowClassifier` | 1,605,891 |
| `TextTower` | 5,240,832 |
| `MaskedLanguageModel` | 5,315,136 |
| `FinanceAgentModel` | 6,847,491 |
| **`GroundedGenerator`** | **7,472,192** |

**Headline results:**

| Claim | Number |
|---|---|
| MLM validation loss | 3.03 vs 8.99 uniform (perplexity 20.7) |
| Corpus | 100.8M tokens, 48,000 steps |
| 5-day direction | 35.7% vs 36.0% baseline — **dead** |
| 5-day volatility | GRU 46.9% vs persistence 47.0% — **ties one line of arithmetic** |
| Old generator, evidence blanked | +0.018 — **ignored its evidence** |
| Content words present in input, old task | 30.7% — **unlearnable as posed** |
| New generator, evidence blanked | **+1.0420** |
| New generator, evidence swapped | **+1.9464** |
| Abstention / false refusal | 100% / 0% |
| Exact match, trained vs reworded phrasing | 45.8% vs **12.5%** |
| Encoder: novel word / novel frame / both | 100% / 68% / 29% |
| Same, after fine-tuning | 93% / **32%** / 14% |
| Phrasings, distinct questions | 190, 19,183 |
| Tests | 234 passing |

---

## 9. Likely viva questions, with answers

**"Why not use PyTorch?"**
Writing autograd by hand is what makes the ablations trustworthy — I know exactly what every gradient
does because I derived it and tested it against finite differences. The cost is speed (~5 steps/s),
which is why the models are small. For a project whose contribution is *measurement methodology*, that
is the right trade.

**"Your model is tiny. Isn't it useless?"**
For generating text from scratch, yes. But it is not asked to. The facts come from `indicators.py` —
pure arithmetic, exactly correct. The model only supplies wording, and the guardrail verifies it did
not invent a number. A 7.5M-parameter model is sufficient for phrasing. §5.2 shows it reads its
evidence: blanking the evidence costs +1.04.

**"How do you know it isn't just memorising?"**
Three ways. (1) The date split — validation snapshots are from a period training never saw.
(2) The wording split — `unseen` holds question phrasings training never saw, and it caught a failure
validation missed entirely (0.011 vs 0.70). (3) Ablation — corrupt the evidence and watch the loss.
A memoriser scores fine with blank evidence; ours costs +1.04.

**"What is your biggest weakness?"**
Paraphrase generalisation. Measured, not guessed: the encoder places a novel *word* perfectly (15/15)
but a novel sentence *shape* only 68% of the time, and both-novel at 29%. Worse, fine-tuning *degrades*
shape handling from 68% to 32% — it memorises the task's strings at the cost of the pretrained
language. I am currently testing whether more sentence frames fixes it, and the probe told me the
600M-token corpus I was about to collect is **not** the binding constraint.

**"Can it tell me what to buy?"**
No, and that is a measured decision, not caution. Five-day direction scored 35.7% against a 36.0%
majority baseline with the loss flat for the whole run. Any confident buy/sell wording would be
dressing a coin toss as a view. The `buy` intent abstains explicitly and still quotes the figures it
does have.

**"What stops it inventing numbers?"**
Two layers. The guardrail checks every figure in the answer against the evidence and replaces the whole
answer with a refusal if any is unsupported. And the slot path removes digits from the model's job
entirely — it names facts and serving substitutes values, so a slot answer *states no figure of its
own*. That makes fabrication impossible rather than unlikely. Built because the copy path, though
working, once answered +3.3% where the evidence said +3.7% — a plausible, unfalsifiable wrong number.

**"What is the point of the third split?"**
It is the difference between believing the model works and knowing it does not. Validation fell to
0.011 while `unseen` froze at 0.70 and then regressed — a 62× gap that a conventional train/validation
split would have hidden completely. The two splits share the same snapshot and the same facts and
differ only in how the question is worded, so the gap isolates paraphrase generalisation.

**"What would you do with more time?"**
In order: (1) finish the frames experiment, and lower the encoder's learning rate to stop fine-tuning
destroying its general language; (2) train the slot variant and compare unsupported-figure rates;
(3) build the learning loop, which is the actual thesis — calibration from outcomes, a learned
abstention threshold, retrieval weighting and candidate ranking, **all with frozen weights** — and
report a learning curve against a no-adaptation control curve. That control is essential: without it,
any improvement could be an artifact of the evaluation order.

**"What is not built yet?"**
Stated plainly: no reranker; no `learn/` modules, so the adaptation loop that is the thesis is designed
but not implemented; no LoRA; no trained dense retriever (BM25 only); no FastAPI backend; the slot
dataset is built but untrained. Phases 1–4 are done and measured; the learning loop is the remaining
work.

---

## 10. The one-minute summary

Pure-NumPy transformer stack, 7.5M parameters, trained on a 100.8M-token finance corpus we collected
politely and ourselves. Question comes in; indicators compute the facts; a price head adds a risk
outlook; both become an evidence passage; an encoder–decoder with Fusion-in-Decoder cross-attention
phrases an answer over that evidence; a guardrail verifies every figure and refuses otherwise.

The findings that matter are the negative ones. Five-day direction is unpredictable (35.7% vs 36.0%),
so the agent refuses to give directional advice. The neural price head only ties a one-line persistence
baseline, so it has not earned its place. The first generator was fluent and completely unfaithful —
proven by ablation, caused by a task where 69% of each answer was invisible in the input. Rebuilding
the task so every figure is copyable fixed it: blanking the evidence now costs 58× what it did.

What is still open is paraphrase generalisation, and it is measured precisely enough to act on: the
vocabulary is fine, the sentence shapes are not, and fine-tuning makes them worse.
