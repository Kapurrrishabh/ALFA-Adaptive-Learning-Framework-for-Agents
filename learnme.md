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

### 5.3 The next failure: reworded questions

With the grounding fixed, the three-way split exposed the next problem. These are the **first** advisory
run's numbers, on 46 hand-written phrasings — §5.7 has what they became once the data was reworded:

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
**Not yet trained.** It was held back because it would have hit the same phrasing wall; now that §5.7
has lowered that wall, training it is the next experiment after the frozen-encoder arm.

### 5.7 The phrasing fix worked, and the number is large

`selfagent/data/advisory.py` now builds questions from **frames × interchangeable words** instead of
one hand-written string each: **190 phrasings**, up from 46, of which 127 are trained and 63 reserved
across the three novelty cells. The dataset rebuilt to **19,183 distinct questions**, up from 4,646,
with the row count unchanged so the two runs are directly comparable. The training procedure was held
identical, so phrasing variety is the only variable.

| step | trained phrasings | **unseen phrasings** |
|---|---|---|
| 2,000 | 0.6997 | 0.8461 |
| 4,000 | 0.1404 | 0.4272 |
| 6,000 | 0.0187 | 0.3801 |
| 8,000 | 0.0175 | 0.3709 |
| 10,000 | 0.0130 | **0.2821 ← best** |
| 12,000 | 0.0057 | 0.3061 |
| 14,000 | 0.0041 | 0.3256 |

**The unseen floor fell from 0.70 to 0.2821 — a 60% cut — from rewording the data alone.** No
architecture change, no extra corpus, no longer training. The old run's unseen loss *froze and then
regressed*; this one genuinely descends for 10,000 steps first. That is the probe's prediction confirmed:
sentence shape was the constraint, and more shapes is the cheap lever.

**What the table also shows is where to stop.** After step 10,000 the unseen loss climbs while the
trained loss keeps falling — ordinary overfitting, but only visible because of the third split.
Validation alone says "still improving" the whole way. The script kept a single checkpoint file and
overwrote it every 2,000 steps, so the best model was being destroyed; `{dataset}.best.npz` now keeps
the turn. Note honestly that a number selected this way is a **model-selection** number, not a clean
test number.

**The next experiment is already wired.** `--freeze-encoder` holds the four encoder layers at their
pretrained values (3,159,552 of 7,472,192 parameters) while the shared token matrix stays trainable —
that matrix is *also* the decoder's output layer, so freezing it by name would stop the model learning
to write at all. This tests the probe's other finding: that task training is what *destroys* the
encoder's handling of unseen shapes (28/41 → 13/41).

**312 tests pass** (`tests/` and `dataforge/` together).

### 5.8 Scoring both models on the real human Q&A data

The real dataset is **Stack Exchange**: 76,857 questions and 117,254 answers scraped from the
economics, money and quant sites, prepared into 60,155 training rows and 2,000 validation rows. Both
models were scored on it so the comparison runs in both directions. `--checkpoint` was added to
`check_answers.py` and `ablate_generator.py` so a model can be pointed at a dataset it was not named
after.

| | advisory data | Stack Exchange data |
|---|---|---|
| loss / perplexity | 0.0073 / 1.0 | 4.1227 / **61.7** |
| cost of blanking the evidence | **+1.0420** | **+0.0191** |
| cost of swapping the evidence | **+1.9464** | **+0.0302** |
| exact match | 90.5% | **0%** |
| unsupported figures | **1.7%** | **57.1%** |
| repeated 4-grams (humans) | 0.3% (0.0%) | 2.6% (2.1%) |

Read it plainly. On the synthetic advisory task the model **reads its evidence** — hide the numbers and
it gets much worse. On real human Q&A, hiding the evidence barely matters: **+0.019 versus +1.042 is
roughly 55× weaker**. The model writes fluent, confident finance prose and **57% of the figures in it
are invented**. That is the original failure, reproduced exactly.

**Why the real data is harder, and it is not the model's fault.** Only **30.7%** of the content words
in a Stack Exchange answer appear anywhere in its retrieved evidence. Most rows simply cannot be
answered from what the model is shown, so ignoring the evidence is the *correct* thing for it to learn.
This is a **retrieval-coverage** problem, not an architecture problem.

**Exact match is the wrong score here.** There is no single right wording for "how do I think about
tom/next swap points", so 0% exact match says nothing. Loss and the blank/swap deltas are the honest
measurement on human prose.

**The advisory model on Stack Exchange scores loss 11.96** (perplexity 156,000). Nothing transfers
across answer distributions — expected, and worth stating, because it means one model per answer style,
not one model for everything.

**One more number worth remembering.** The S14 gate is "under 2% unsupported figures". On the advisory
validation split the current model hits **1.7% — the gate closes**. On phrasings it has never seen it
is **4.7% — the gate fails**. So faithfulness itself depends on familiar wording, which is the same
paraphrase weakness from §5.7 arriving at the number the product is actually judged on.

### 5.9 The checkpoint with the better loss is the worse model

This one is worth having ready, because it is counter to what everyone is taught about early stopping.
The chat-register arm was trained for a full epoch and two checkpoints were compared — the one held-out
loss picked, and the last one.

| | loss on unseen phrasings | exact match | unsupported figures | false refusals |
|---|---|---|---|---|
| picked by loss (step 6,000) | **0.2339** | 32.5% | 17.2% of answers | 18.9% |
| the final one (step 28,163) | 0.4162 | **39.0%** | **4.1%** | **4.7%** |

The loss-selected checkpoint is better on loss by a wide margin and **worse on every number a user would
notice**. Early stopping here would have shipped the weaker generator, and nothing in the loss curve says
so.

**Why it happens.** The evidence-swap ablation grows from **+1.69 to +2.60** between those two
checkpoints. The late model is *more* grounded, not less: it commits harder to the specific digits its
evidence states. Cross-entropy on an unfamiliar phrasing punishes exactly that commitment, because a
confidently wrong word costs more than a hedged one. So rising held-out loss is the price of reading the
evidence.

**And it replicated.** A second arm was trained — the same chat-register data with the encoder frozen —
and its loss-picked checkpoint (step 6,000, unseen loss 0.1435) loses to its final one on all four numbers
again: exact match 30.5% against **36.0%**, unsupported figures 17.2% against **2.4%**, false refusals
7.7% against **3.6%**, and on the register's own split 77.0% against **94.5%**. Two arms, two datasets,
same direction and roughly the same size. One such result is an anecdote; this one is a property of the
task.

**What to say if asked.** Loss is a proxy, and it is a proxy for the wrong thing once generation is what
you ship. Every checkpoint decision in this project is now made on generated answers — exact match,
unsupported figures, refusal behaviour — with loss used only to see that training is progressing at all.

### 5.10 Stacking the two fixes: they add on faithfulness, not on exact match

Two separate fixes had each worked alone — freezing the MLM encoder (§5.4), and four times more phrasing
variety in the data (§5.7). The obvious question is whether they add. Both arms were measured on the same
two splits and the same 200 rows, one fix apart:

| | phrasing variety only | both fixes |
|---|---|---|
| loss on unseen phrasings | 0.4162 | **0.2496** |
| exact match, unseen | **39.0%** | 36.0% |
| unsupported figures, unseen | 4.1% of answers | **2.4%** |
| false refusals, unseen | 4.7% | **3.6%** |
| exact match, own split | 94.5% | 94.5% |

Freezing the encoder on top of the better data cuts held-out loss by 40%, cuts unsupported figures and
false refusals by about a quarter each, and leaves the trained-phrasing exact match untouched at the row —
189/200 in both arms.

**Be careful with the 3-point drop on unseen exact match.** It is 6 rows out of 200, and this sample was
separately measured to carry about ±3 points at n=200 (the same accuracy read 31.0% on 200 rows and 28.0%
on 600). So the honest statement is that the faithfulness gain is real and larger than the noise, and the
exact-match cost is inside it. Claiming "freezing the encoder costs 3 points of accuracy" from this
measurement would be reading a number the sample size does not support.

### 5.11 The learning loop — the thesis, and whether it worked

Everything above is about one frozen model. This section is the research question itself: *can the agent
improve from interaction, outcomes and feedback without retraining the foundation model?* Read it as one
argument in six steps. **The short answer is yes, on a narrower claim than the question suggests** — the
agent learns *when to speak*, measurably and from outcomes alone, and it does not learn *what to say*.

**The log — `selfagent/learn/store.py`.** One SQLite table, append-only: question, evidence, answer,
confidence, verdict, **who judged it**, timestamp. Nothing is ever updated or deleted; a label that turns
out wrong is superseded by a later row, because the history of what the agent believed is the data the
thesis is about. Everything else in the loop — calibrator, threshold, ranking — is derived from this file
and can be thrown away and refitted.

**The judge — `teacher.py`.** There is no financial analyst to ask, so there are two judges. The **oracle**
compares against the dataset's gold answer: exact, free, and only available on built rows. The **agent** is
Claude, judging in session, for rows with no gold. Its verdicts are written to a file and replayed, never
re-asked, because a learning curve that saw different feedback on its second run is not a curve. **A cache
miss reports "unjudged" and never guesses**: a fabricated label is indistinguishable from a real one once
it is in the log, and would flatter every number downstream.

The agent is shown the question, the evidence and the answer — and deliberately **not** the gold, because
judging with the gold in hand reduces to string comparison and agreement would be 100% by construction.
Blind, the comparison is a measurement: **agreement 156/198 (78.8%)**. The oracle scored 62/200 right
(31.0%), matching `check_answers.py` exactly — the consistency check that the labeller and the scorer are
one rule — and the agent scored 103/198 (52.0%). **All 42 disagreements run one way**: the oracle said
*differs from the gold answer* and the agent said right. There is **not one row** where the oracle said
right and the agent said wrong. So exact match understates this model by about 21 points, and blind judging
never once rejected an answer the oracle could verify.

What the agent's 95 wrong rows actually are, which is what set the rest of the queue: **intent misrouting
dominates** — a buy-advice template answering an overbought question (14), performance answering an outlook
or volatility question (13), drawdown answering an outlook question (6). Then **30 false refusals**. Then
two kinds the guardrail cannot see: four wrong *comparative* claims ("above both its 20 and 50 day
averages" when the price is below one), where every digit is supported, and four digit-duplication
inventions (`242804.00` for a close of `24280.00`), which it does catch. **A figure guard is not an intent
guard**, and that is what C1's router exists to fix.

**Calibration — `calibrate.py`.** The model's confidence averages 0.990 on answers that are right 31% of
the time, so as a probability it is worthless. Platt scaling, two parameters on the log odds, fitted on half
the log and scored on the other over 25 splits: `p = sigmoid(0.33 × log odds − 3.31)`, and calibration error
falls **0.683 → 0.121** under the oracle and 0.475 → 0.144 under the agent, on every split.

That looks like a triumph and mostly is not, which is why a **third** predictor was measured: a constant
that ignores the confidence and just states the base rate. Against it, calibration error is a coin toss
(12/25) while Brier is a near-sweep (25/25). Read the Brier — calibration error flatters a constant, which
has no spread inside a bin to be wrong about. **The number that mattered downstream is ranking AUC 0.740
± 0.039.** The confidence does sort right above wrong, well clear of a coin toss, and because any monotone
calibration leaves the order untouched, that AUC is a hard **ceiling** on the threshold and the reranker
that follow.

**The learned threshold — `abstain.py`.** The old abstention cut was a hand-picked 0.9980, a mean token
probability with no meaning attached: nothing about it states how often an answer above it turns out right,
which is the only thing a user wants to know. `Abstainer.fit(..., wanted=0.6)` instead takes a stated bar —
"answer only where you are at least 60% likely to be right" — and solves for the confidence that delivers
it. Held out, under the oracle: asking 50% buys **51.5% correct on 43.8% coverage**, asking 60% buys
**58.5% on 32.7%**, asking 70% buys **64.3% on 24.2%**, against 31.0% for answering everything. So +21 to
+34 points of precision for the coverage given up.

Be precise about what this did and did not beat. It **tied** the old number (53.6% on 39.0% against 53.8%
on 40%) and that tie is the result, because 53.8% had been chosen on the very rows it was quoted on — an
upper bound, as `check_answers.py` says in its own docstring. A cut fitted out of sample *reproducing* an
in-sample bound at matched coverage was the thing in doubt. No threshold could have done better: a cut on
a calibrated probability orders rows identically to a cut on the raw confidence, so every threshold lives
on one precision–coverage curve fixed by the model at AUC 0.740. **What changes is the meaning, not the
number** — 0.9980 means nothing on a retrained model, 60% means 60% always — and it reports the bar it
missed instead of claiming it.

**The reranker that failed — `rank.py`.** Sample k answers, rank them with frozen weights, serve the best:
grounded above silent above invented, confidence only deciding among equals. **It does not pay, and the
reason is worth more than a win would have been.** On 600 unseen rows: single sample 28.0%, best of 8 by
confidence 27.8%, by `rank()` 28.5%, by most-repeated 29.2%, ceiling 34.0%. Eight samples produce **1.77
distinct answers**. There is nothing to rank. `_nucleus` keeps a second token only when the top one holds
less than `top_p` of the mass, and this model's per-token mass is about 0.99 — the same fact as the
confidence sitting above 0.999. **A near-deterministic model cannot be improved by sampling it more**, and
loosening the nucleus tenfold moved distinctness by 0.1 and the ceiling not at all. The 1.2-point
consensus lead rescued 14 rows and spoiled 7, which is McNemar **p = 0.189** — a lead, not a result.

Two things fell out of it. **Ranking on confidence is worse than not ranking** (-2.5 points), because
refusals carry the highest confidence the model produces (0.99538 against 0.98797 on answers), so picking
the surest candidate picks the refusal. The AUC 0.740 holds *across* questions and does **not** transfer to
ranking answers to one — a distinction easy to miss and expensive to miss. And the tier order does help
where it fires, beating confidence by 1.5 points; it just fires on 4–6% of rows, because this checkpoint
invents figures on only 0.7% of unseen answers. It is kept for C1, where candidates will differ by
retrieved context rather than by resampling one question.

**The curve — `scripts/learning_curve.py`.** This is the thesis measured. Because the adaptation sets a
threshold, the score is not accuracy but the **miss**: how far the delivered precision lands from the bar
the agent promised, on rows the cut never saw.

| | oracle @ 60% | agent @ 80% |
|---|---|---|
| 10 rows of feedback | 12.9 pts | 13.3 pts |
| 60 rows | **8.5** | 9.0 |
| 118 rows (the whole log) | 9.6 | **6.3** |
| hand-picked 0.9980, sees no feedback | 16.7 | 10.4 |
| no abstention at all | 29.2 | 27.2 |
| fitted on the scored rows (ceiling) | 1.1 | 3.5 |

**Both flat controls are beaten at every log size past ten rows**, and nothing in the transformer changed to
earn it. Feedback also pays in proportion to how hard the promise is: at a 60% bar the curve flattens after
60 rows, while at an 80% bar it falls monotonically and is **still falling at 118 rows**, which is the end
of the log. The flat case is not a null result but an easy bar — the agent's own base rate is 52.8%, so
answering everything nearly clears 60% by accident.

**And the finding to carry:** a miss that will not close is not always a failure to learn. At the oracle's
80% bar the curve bottoms at 15.0 points, but the *ceiling* — a cut fitted on the very rows it is scored on
— also misses by 11.7. This model has no confidence region that is right 80% of the time at 20% coverage or
better, so only 3.3 of those 15.0 points was ever learnable. Splitting every miss into the part the bar
forbids and the part the log has not yet taught is what keeps a threshold from quietly promising something
it cannot deliver.

**The loop itself — `scripts/chat.py`.** Twenty turns end to end: ask, answer, judge, append, refit, show
what moved. The log grew 0 → 20 and the cut moved 0.99800 → 0.99962 **at turn 10**, the warmup boundary —
before it the loop serves the hand-picked value and *says* it is hand-picked, because a threshold solved
from three rows is a coincidence and serving one while calling it learned is the one thing this loop must
not do.

**So: did the loop work?** Yes, with the claim stated exactly. Against a no-adaptation control, a threshold
solved from 60 rows of outcomes keeps its stated promise nearly twice as closely as the hand-picked one and
three times as closely as not abstaining at all, with the foundation model's weights loaded once and never
touched. That is improvement from interaction without retraining. What the loop did **not** do is improve
the wording — and that is not an omission but a measurement: B5 showed there is nothing to choose between
this model's samples. The answers come from frozen weights; the judgement about whether an answer is worth
saying comes from the log. Anyone claiming more than that from these numbers is overselling them.

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

### `selfagent/learn/` — the adaptation loop (§5.11)

Nothing here touches the transformer. Every module reads the feedback log and produces a threshold, a
mapping or an ordering, which is what makes the thesis's "without retraining" claim checkable.

- **`store.py`** — `FeedbackLog`: `append`, `rows`, `agreement`. Append-only SQLite; `labeller` is a
  required argument, because a row whose judge is unknown cannot be excluded from a measurement later.
- **`teacher.py`** — `OracleTeacher` (judges against gold), `AgentTeacher` (replays Claude's cached
  verdicts, `write_requests` / `record`), `verdict_key`. A cache miss returns `None`, never a guess.
- **`calibrate.py`** — `Calibrator` (Platt scaling, `fit` / `__call__`), `expected_calibration_error`,
  `brier`, `ranking_auc`. It refuses to fit on rows that are all right or all wrong.
- **`abstain.py`** — `Abstainer.fit(confidence, is_right, wanted)`, `precision_at`, `COVERAGE_FLOOR`.
  Keeps `expected` so a served threshold can be compared against what actually happens.
- **`rank.py`** — `tier`, `rank`, `best`, `consensus`. Measured not to pay on this model (§5.11); kept
  for C1, where candidates will differ by retrieved context.

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
| `label_feedback.py` | Fills the log with both judges' verdicts. `--ask` generates and labels, `--tell` reads Claude's verdicts back and prints the agreement rate |
| `calibrate_confidence.py` | Fits the calibrator and reports it against **two** baselines, the raw confidence and the base rate |
| `fit_abstention.py` | Solves the abstention cut from a stated bar and quotes it out of sample |
| `rerank.py` | **Does best-of-k buy anything?** Four pickers against taking the first sample. It does not (§5.11) |
| `learning_curve.py` | **The thesis as a curve** — the miss against the stated bar versus how much feedback the cut was solved from, beside two flat controls and a ceiling |
| `chat.py` | The loop, one turn at a time: ask, answer, judge, adapt, show what moved |

**Why `check_answers.py` exists at all.** Perplexity on this data is flattering: the answers are
templated, so most positions are boilerplate a model can predict without reading anything, and the
handful of digit positions that actually need the evidence are lost in the average. Generate-and-check
is the number that decides whether the system works.

### `tests/` — 312 passing

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
| Exact match, trained vs reworded phrasing | 90.5% vs **28.5%** (was 45.8/12.5 pre-fix) |
| Unsupported figures, trained vs reworded | **1.7%** vs 4.7% — S14's 2% gate closes then fails |
| Stack Exchange: loss, blanked, unsupported | 4.1227, **+0.0191**, **57.1%** of figures |
| Advisory model on Stack Exchange | loss **11.96** — nothing transfers |
| Encoder: novel word / novel frame / both | 100% / 68% / 29% |
| Same, after fine-tuning | 93% / **32%** / 14% |
| Phrasings, distinct questions | 190, 19,183 |
| Unseen-phrasing loss, 46 vs 190 phrasings | 0.70 → **0.2821** (60% cut, data reworded only) |
| Encoder frozen by `--freeze-encoder` | 3,159,552 of 7,472,192 parameters |
| Chat register, final checkpoint | **94.5%** exact match on its own split, **39.0%** unseen |
| Same arm, checkpoint picked on loss | 68.0% / 32.5% — **better loss, worse on all four** |
| Evidence-swap penalty, early vs late | +1.69 → **+2.60** — grounding grows as loss rises |
| Both fixes stacked, unseen | loss **0.2496**, unsupported **2.4%**, exact match 36.0% |
| Same arm, checkpoint picked on loss | 77.0% / 30.5% — **the finding replicates** |
| Oracle and agent judges agree | **156/198 (78.8%)**, and all 42 disagreements run one way |
| Calibration error, raw → Platt | 0.683 → **0.121**; but a constant ties it on error and loses on Brier |
| Ranking AUC of the confidence | **0.740 ± 0.039** — the ceiling on the threshold and the reranker |
| Abstention, asked 60% / 70% | **58.5% on 32.7%** / 64.3% on 24.2%, against 31.0% answering everything |
| Best of 8 samples | 28.0% → 29.2%, **p = 0.189**; only **1.77 distinct answers** of 8 |
| Learning curve, miss vs stated bar | **8.5 pts** from 60 rows, against 16.7 hand-picked and 29.2 unabstained |
| Tests | 312 passing |

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
differ only in how the question is worded, so the gap isolates paraphrase generalisation. It then did
the job a second time: it is the only split that shows the reworded run overfitting after step 10,000,
while validation says "still improving" all the way to the end.

**"Did the learning loop actually work?"**
Yes, on a precise claim. The agent is asked to promise a chance of being right, and a threshold solved
from 60 rows of its own outcomes keeps that promise to within **8.5 points**, against 16.7 for the
hand-picked threshold and 29.2 for not abstaining at all — two flat controls, beaten at every log size
past ten rows, with the foundation model's weights loaded once and never touched. At a strict bar the
curve is still improving at 118 rows, the whole log. What the loop does **not** do is improve the
wording, and that is a measurement rather than an omission: eight samples of this model produce 1.77
distinct answers, so there is nothing for a reranker to choose between. **It learns when to speak, not
what to say.** See §5.11.

**"What would you do with more time?"**
In order: (1) a longer feedback log — the curve at a strict bar was still falling when the log ran out at
118 rows, so the cheapest remaining win is more judged turns, not a cleverer fit; (2) fix **intent
misrouting**, which is the largest single error class (33 of the agent's 95 wrong rows) and which the
figure guardrail structurally cannot see — that is C1's router; (3) train the slot variant and compare
unsupported-figure rates.

**"What is not built yet?"**
Stated plainly: no LoRA; no trained dense retriever (BM25 only); no FastAPI backend or frontend; the slot
dataset is built but untrained; and the reranker is built but **measured not to pay** on this model. The
learning loop itself is built and measured (§5.11) — what remains is the served architecture around it.

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

The thesis itself comes out positive but narrow. Asked to promise a chance of being right, the agent
solves for that threshold from its own logged outcomes and keeps the promise to within 8.5 points, where
the hand-picked threshold misses by 16.7 and no abstention by 29.2 — improvement from feedback with the
foundation model frozen. It does not learn better wording, because eight samples of it produce 1.77
distinct answers: there is nothing to choose between. **It learns when to speak, not what to say.**

What is still open is paraphrase generalisation, and it is measured precisely enough to act on: the
vocabulary is fine, the sentence shapes are not, and fine-tuning makes them worse.
