# Financial Advisor (StockIntel) — personal stock signal system and research assistant for NSE

Two layers:

1. **Signal system (v2):** tells you what to buy, what to sell, how many shares, where to put
   stops and when to rebalance. It ranks Nifty 200 stocks with NSE's own momentum
   (or low-volatility) method, reads the market's trend and volatility to set exposure, flags
   promoter buying, and adds tax-aware exit notes. It is built on the factor with the strongest
   Indian evidence: the live Nifty200 Momentum 30 ETF beat the Nifty 500 by about 1.5–2.5 points a
   year from Aug 2022 to Sep 2026, and our same-universe backtest showed a similar edge.
2. **Research assistant (v1):** single-stock analysis across technicals, fundamentals, news,
   risk and more, fused into a label with the conflicting evidence kept visible. An optional
   Claude layer explains it and is rejected if it writes a number the evidence doesn't contain.

What the evidence does **not** support, measured on NSE data: candle and chart patterns,
single-stock timing rules, and direction forecasts. The system doesn't use them to trade.
Momentum can fall 35% peak to trough and can trail the market for a year; the plan says so
each time it runs. This is decision support, not a guarantee.

## The app (recommended)

```bash
.venv/bin/stockintel app          # opens http://127.0.0.1:8931 in your browser, logged in
```

A Groww-style web app for personal use:

- **Explore:** Nifty/Sensex/Bank Nifty cards, market-trend banner, top momentum picks,
  top movers, and promoters buying this month.
- **Search:** type a company or symbol (all Nifty 500 names).
- **Stock page:**
  - price chart (1W–5Y) with hover, returns, 52-week range and fundamentals;
  - Research, News and Insider tabs, and an "Ask about this stock" box;
  - a **"Should I buy?" card**: BUY / WAIT / DON'T BUY / HOLD / SELL AT REBALANCE, with
    "why you might buy" and "why you might not", a stop-loss, position size and next
    review date.
- **Technical chart** (stock page → "Technical"): candlesticks with the detected pattern drawn on
  them (labelled swing points such as Top 1 / Neckline / Top 2, trendlines, breakout and textbook
  target), ▲/▼ candle-signal markers, 50/200-day averages, support/resistance, and the next-20-day
  price range. Every drawing carries its tested record ("tested 73 times across 32 stocks — no
  edge"), so a clean-looking shape is never presented as a proven signal.
- **Performance:** charts of how well each part works out of sample: momentum vs the market and
  the live ETF year by year, forecast accuracy and calibration, price-range coverage, promoter
  buying, and every candle and chart pattern with its 95% range. Build it once with
  `stockintel build-performance` (about 30 seconds).
- **Portfolio:** add or delete transactions; see value over time vs money invested, P&L,
  holdings, a sector donut, risk and flags.
- **Signals:** a full buy/sell/trim plan for your money and holdings.
- **Ask AI:** chat with memory ("Should I buy TCS?", "Why?", "Compare it with INFY").
- **Learn:** plain-English lessons with search.
- **Reports:** 18-section research reports you can print to PDF.

Light and dark themes; works on phones (bottom tab bar). The login key is created and passed
in the link automatically; the server listens only on your machine.

## Quick start

```bash
git clone https://github.com/Kapurrrishabh/financial-advisor.git
cd financial-advisor
python3 -m venv .venv                               # Python 3.11 or newer
.venv/bin/pip install -e ".[dev]"

.venv/bin/stockintel --demo analyze RELIANCE         # offline, SYNTHETIC data, no network
.venv/bin/stockintel analyze RELIANCE                # live NSE data (Yahoo Finance + Google News)
.venv/bin/stockintel chat                            # multi-turn research conversation
.venv/bin/pytest                                     # 212 tests, ~5 s, fully offline
```

Dashboard and REST API:

```bash
export STOCKINTEL_API_KEY=$(openssl rand -hex 24)    # required; the server refuses to start without it
.venv/bin/stockintel serve --port 8931               # open http://127.0.0.1:8931 and paste the key
```

## Signal system (v2) — what to buy, sell and when

```bash
.venv/bin/stockintel signals --capital 200000              # this period's plan for ₹2 lakh
.venv/bin/stockintel portfolio import holdings.json         # then plans include your holdings
.venv/bin/stockintel signals --risk defensive               # trend + volatility overlay (lower drawdown)
.venv/bin/stockintel stops                                  # daily stop check
.venv/bin/stockintel insider-fetch                          # update promoter-trade flags
.venv/bin/stockintel evaluate-signals                       # reproduce the backtest + survivorship check
```

In chat: "What should I buy with ₹1 lakh?", "Rebalance my portfolio", "Is the market in a
downtrend?", "Check my stops". Method and measured results: [docs/DESIGN.md §28](docs/DESIGN.md).

## Commands

| Command | What it does |
|---|---|
| `analyze SYMBOL [--horizon short\|medium\|long] [--report] [--json]` | Full analysis; `--report` writes the 18-section research report |
| `chat [--llm off\|narrate\|agent]` | Conversation with memory: "Analyze TCS" → "Why?" → "Compare it with INFY" → "I have ₹1 lakh…" |
| `ask "question"` | One natural-language question |
| `screen --budget 50000 [--risk conservative] [--exclude "Financial Services"]` | Budget-aware research candidates, each with the reasons it qualified |
| `portfolio import holdings.json` / `portfolio show` / `portfolio what-if --amount 20000 --symbols TCS INFY` | Portfolio P/L, exposure, concentration, risk contribution, scenarios |
| `backtest SYMBOL [--fusion]` / `backtest --universe` | Baselines vs the engines, net of costs, next-open fills |
| `validate-patterns --universe [--save]` | Pooled event study of every candle/chart pattern; `--save` makes the results priors for live analysis |
| `forecast-eval --symbols A,B,C [--models logistic,gbm] [--register]` | Walk-forward evaluation; `--register` records a candidate in the model registry |
| `models list\|approve\|deploy\|reject NAME VERSION --by NAME` | Controlled model promotion (no automatic path) |
| `alerts [SYMBOLS...]` | Scan for alerts, or list stored ones |
| `monitor` | Score matured predictions (live Brier) and check return-distribution drift (PSI) |

Global flags: `--db PATH` (default `~/.stockintel/stockintel.db`), `--demo` (synthetic data).

Portfolio file format (holdings are always derived from transactions; enter current holdings
as BUY transactions at your average cost):

```json
{"cash": 20000, "risk_profile": "moderate", "horizon": "long",
 "transactions": [{"symbol": "RELIANCE", "side": "BUY", "quantity": 10, "price": 2400,
                   "trade_date": "2024-01-15", "sector": "Energy"}]}
```

## Generative layer (optional)

`--llm narrate` rewrites computed answers conversationally; `--llm agent` lets Claude choose
the tools itself. Both are verified: every figure in the reply must match a figure in the
evidence, or the reply falls back to the computed answer and says why.

- Claude API: set `ANTHROPIC_API_KEY`. Requests opt into server-side refusal fallbacks
  (`fallbacks="default"`).
- Amazon Bedrock: `STOCKINTEL_LLM_PROVIDER=bedrock`, `AWS_REGION`, and `AWS_PROFILE` for a
  profile with Bedrock model access.
- Model: `STOCKINTEL_LLM_MODEL` (default `claude-opus-5`).

## Measured results

All on real NSE data, reproducible with the commands shown. Details in
[docs/DESIGN.md §17](docs/DESIGN.md#17-evaluation-methodology-and-measured-results).

- **Patterns** (`validate-patterns --universe`): 48 large caps × 5 years, ~12,000 candlestick
  events and ~3,000 chart-pattern completions replayed point-in-time. None of the 43 patterns
  shows a significant forward edge after Bonferroni correction (|z| ≥ 3.25). The engines
  therefore weight well-tested patterns at 0.1 and say so in each claim.
- **Timing** (`backtest --universe`, Dec 2022 – Sep 2026, 17 bps per unit turnover): median
  Sharpe buy-and-hold 0.63; SMA 50/200 0.37; RSI mean reversion 0.21; the replayed technical
  engine 0.23 (beat buy-and-hold on 14/48 stocks); logistic forecast strategy 0.15.
- **Forecasts** (`forecast-eval`, 12 large caps, ~11,000 out-of-sample predictions per horizon):
  neither logistic nor gradient boosting beats the base rate at 1, 5 or 20 days (Brier skill
  −0.016 to −0.218, AUC ≈ 0.50). The skill gate zeroes the forecast's weight and the registry
  refuses to promote such a model.
- **Ranges**: the EWMA 80% range covered a median 79.4% (5-day) and 77.2% (20-day) of realized
  outcomes out of sample.

## Layout

```
stockintel/
  data/          providers (yahoo, rss, local, synthetic), normalisation, quality gate
  analysis/      technical, candlesticks, patterns, fundamentals, news, sentiment,
                 quant, regime, forecast, analogues, validation, indicators, stats
  fusion.py      evidence fusion + decision engine
  service.py     fetch -> validate -> analyze -> fuse (single entry point)
  nlu.py         query understanding     orchestrator.py  intents, tools, sessions, LLM modes
  llm.py         Claude client, faithfulness verifier, narrator, tool agent
  portfolio.py   screener.py  alerts.py  report.py  knowledge.py (+ knowledge_base/)
  backtest/      engine (next-open fills, costs, stops), strategies, comparison
  storage.py     SQLite  registry.py  monitoring.py  api.py  web/index.html  cli.py
tests/           212 offline tests
docs/DESIGN.md   design specification, decisions, research answers, measured results
```

Decision-support only. Not investment advice. Verify figures against primary sources
(NSE/BSE filings) before acting.


## Run it on your own computer

```bash
git clone https://github.com/Kapurrrishabh/financial-advisor.git && cd financial-advisor
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/stockintel app                    # opens the app in your browser, logged in
# optional, once: fill the Performance page and pattern track records (~1 minute)
.venv/bin/stockintel validate-patterns --universe --save && .venv/bin/stockintel build-performance
```

Windows: use `.venv\Scripts\pip` and `.venv\Scripts\stockintel`. Your data (portfolio, reports,
cache) stays in `~/.stockintel/` on your machine.

## Put it on the web (a link others can open)

The app is a single-user program: **one portfolio and one login key per running copy**. If you
put one copy online and share the link and key, everyone sees and edits the same portfolio.
For other people, the right setup is that each person runs their own copy (above) or deploys their own.

**Render (simplest, about 10 minutes):**
1. Sign in at render.com with GitHub, choose *New → Blueprint*, and pick this repository.
   It reads `render.yaml`: a Docker web service with a 1 GB disk for the database, and a random
   `STOCKINTEL_API_KEY`.
2. When it's live, open *Environment*, copy `STOCKINTEL_API_KEY`, and visit
   `https://<your-service>.onrender.com/#key=<that key>`.
3. The Starter plan (paid) keeps the disk and doesn't sleep; the free plan has no disk, so
   the portfolio resets on every restart.

**Any Docker host (Railway, Fly.io, a VPS):**
```bash
docker build -t financial-advisor .
docker run -p 8000:8000 -e STOCKINTEL_API_KEY=$(openssl rand -hex 24) -v advisor-data:/data financial-advisor
```

Things to know before going public:
- **Data sources:** prices and news come from Yahoo Finance and Google News, and insider trades from
  NSE's website. These are free for personal use, but their terms restrict redistribution, and
  cloud servers are sometimes rate-limited or blocked by them. For a public service, switch to a
  licensed data feed (for example a broker API).
- **Keep the key secret:** it is the only login. Share it only with people you trust with your portfolio.
- **Not investment advice:** the app says this on every verdict; keep that if you publish it.

## Tests

`pytest -q` runs 212 offline tests (no network). GitHub Actions runs them on every push.
