# ALFA — a self-learning financial agent and the advisory app built on it

One full-stack project in two halves:

- **ALFA (research):** a Transformer framework written from scratch in NumPy (autograd, layers,
  tokenizer, optimisers), and the models trained with it: a grounded chat agent that answers only when
  its evidence supports it, a return generator that draws possible price paths, and a GRU that projects
  the next 20 days. No deep-learning library is used for these; a test enforces it.
- **StockIntel (the app):** a web app for NSE stocks: should-I-buy verdicts, the NSE momentum signal
  system, a portfolio builder, pattern charts, a chat assistant, and every model's tested record. Its
  forecast layers are our models; open-source ones (Chronos-2 and Kronos fine-tuned on NSE, and a
  small language model that writes the analyst note) are add-ons kept in their own folder.

Every number the app shows comes from a measured record. When a model has no edge, it says so.

## Layout

```
frontend/                      web app (HTML/CSS/JS, installable on a phone)
backend/
  pyproject.toml               one package: `backend`
  tests/models/                ALFA's tests, incl. the no-deep-learning guard
  tests/app/                   the app's tests
  backend/
    api/                       FastAPI app, page data (hub), chat (orchestrator), agent server
    models/
      core/                    from-scratch framework: autograd, nn (incl. GRU), optim, tokenizer
      networks/                ALFA's models: agent, generator, return generator, path GRU, GARCH
      data/ learning/          training datasets; preference, verified and calibration learning
      agent/ guardrails/       the chat agent's runtime and its answer checks
      serving/                 what the app serves: return-model fan, GRU line, registry
      training/                every training and evaluation script
      external/                open-source models (Chronos-2, Kronos, language model, Claude)
      artifacts/               weights and records (not in git)
    advisory/
      statistics/ signals/     indicators, patterns, regime, forecasts; NSE momentum signal system
      fusion/ calculators/     evidence fusion; backtests, screener, performance records
      fundamental/ sentiment/  fundamentals; news and sentiment
      analyst.py               the analyst note: our numbers, written up and checked
    knowledge_base/            lessons, and the retrieval index the agent reads
    user/portfolio/            portfolio, alerts
    database/                  SQLite storage, data sources, corpus collection, data/ (not in git)
docs/                          architecture diagrams, design notes, ALFA research and paper
deploy/                        Hugging Face model Space and publish script
```

## Run it

```bash
cd backend
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev,ml,cloud]"
.venv/bin/stockintel app                                   # opens the app in your browser
.venv/bin/python -m backend.models.agent.run.serve --port 8010   # optional: the chat agent (~4.5 GB RAM)
.venv/bin/pytest -q                                        # both test suites
```

Training (each writes its weights and test record to `backend/backend/models/artifacts/`):

```bash
.venv/bin/python -m backend.models.training.train_return_generator
.venv/bin/python -m backend.models.training.train_path_gru
.venv/bin/stockintel train-chronos && .venv/bin/stockintel train-kronos
.venv/bin/python -m backend.models.external.compare_writers  # picks the analyst's language model
```

## More

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): diagrams and every model with its measured result
- [docs/ALFA.md](docs/ALFA.md), [docs/RESEARCH.md](docs/RESEARCH.md), [docs/paper/](docs/paper/): the research
- [backend/README.md](backend/README.md): the app's commands, signal system and publishing

Decision support, not investment advice.
