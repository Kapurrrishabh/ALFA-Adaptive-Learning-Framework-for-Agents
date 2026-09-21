# Self-Learning Financial Agent

A Transformer encoder written from scratch in NumPy that reads **finance text** (textbooks, news) and
**OHLCV price windows** through one shared architecture, then keeps learning from the person using it.
Each user gets their own lightweight adapter, so the same base model adapts to different people's data
without forgetting what it already knew.

No TensorFlow, Keras, PyTorch, HuggingFace, or hosted API. The autograd engine, tokenizer, attention,
optimizers and reinforcement-learning loop are all in this repository. `scipy` and `sklearn` are
allowed only under `selfagent/eval/`, for metrics and baselines the model itself never touches —
`tests/test_import_guard.py` enforces this on every source file.

## Setup

```bash
python3 -m pip install numpy pandas matplotlib pytest
```

## Run

```bash
python3 -m pytest tests/ -q          # 114 tests, ~1.5s
python3 scripts/benchmark_step.py    # measures step time for four model sizes
```

## What works today

Phases 0–2 of [docs/PLAN_OF_ACTION.md](docs/PLAN_OF_ACTION.md) are complete.

- **Autograd** — reverse-mode over a graph rebuilt each forward pass, with an iterative topological
  sort so deep graphs cannot exhaust the Python stack. Every derivative is checked against central
  finite differences in float64; that suite is the project's foundation gate.
- **Layers** — Linear, Embedding, LayerNorm, Dropout, multi-head attention (self and cross), pre-LN
  Transformer blocks, `TransformerEncoder`.
- **Two modality front-ends onto one encoder** — `TextEmbedding` for tokens, `PricePatchEmbedding`
  which groups consecutive bars into patches (PatchTST, 2023). This shared stack is what lets one
  agent read prose and price series.
- **Models** — `MaskedLanguageModel` for pretraining (output matrix tied to the input embedding) and
  `FinanceAgentModel` which fuses both towers and classifies.
- **Training** — AdamW with decoupled decay, warmup-then-cosine schedule, global gradient clipping.
- **Price features** — log returns standardised per window, so a $5 stock and a $500 stock look the
  same to the model. Raw price levels are never fed in.

Next: WordPiece tokenizer, the pandas data layer with point-in-time news/price alignment, and MLM
pretraining on the finance corpus.

## Repository

| Path | Contents |
|---|---|
| [docs/RESEARCH.md](docs/RESEARCH.md) | Literature grounding, design decisions and rejected alternatives, risk register |
| [docs/PLAN_OF_ACTION.md](docs/PLAN_OF_ACTION.md) | Module layout, phases with exit gates, measured configs, the headline experiment |
| [selfagent/](selfagent/) | The library |
| [tests/](tests/) | Gates S1, S2, S11 |
| [configs/](configs/) | `tiny.json` for tests, `target.json` for reported results |

## Two things to know before trusting a number

**Compute is not the constraint; corpus size is.** A 5.3M-parameter model pretrains on 15M tokens in
about an hour on a laptop. But that is ~3 tokens per parameter against BERT-base's ~30, so the model
is data-limited, not compute-limited. Collect more text before adding layers.

**Time-ordered data must never be split randomly.** All evaluation uses walk-forward splits with
purging and an embargo. `docs/RESEARCH.md` §4 lists what this project deliberately does *not* claim,
including anything about profitable price prediction.
