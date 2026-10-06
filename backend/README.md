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

- **Home:** what the app does, the measured track record at a glance, and a login box.
- **Portfolio builder:** enter an amount, pick a style (momentum, low volatility, blend), a risk level
  and how many stocks; get share counts, stops, risk, a sector split and last year's backcast, then
  add the whole basket to your portfolio in one tap.
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
  target), ▲/▼ candle-signal markers, 50/200-day averages, support/resistance, and the next-10-day
  price range, plus an optional **AI forecast range** from the open-source Chronos model (median
  points and 50%/80% bands for the next 10 days). Every drawing carries its tested record ("tested 73 times across 32 stocks — no
  edge"), so a clean-looking shape is never presented as a proven signal.
- **Performance:** charts of how well each part works out of sample: momentum vs the market and
  the live ETF year by year, forecast accuracy and calibration, price-range coverage, promoter
  buying, and every candle and chart pattern with its 95% range. Build it once with
  `stockintel build-performance` (about 30 seconds).
- **Portfolio:** add or delete transactions; see value over time vs money invested, P&L,
  holdings, a sector donut, risk and flags.
- **Signals:** a full buy/sell/trim plan for your money and holdings.
- **Ask AI:** chat with memory ("Should I buy TCS?", "Why?", "Compare it with INFY"). If the
  ALFA self-learning agent is running, tick "Also ask my self-learning agent"
  for its answer as a second bubble (it stays quiet when it isn't sure).
- **Learn:** plain-English lessons with search.
- **Reports:** 18-section research reports you can print to PDF.

Light and dark themes; works on phones (bottom tab bar) and can be added to the home screen. The
login key is created once, saved in `~/.stockintel/app_key` and passed in the link; the server
listens only on your machine unless you add `--lan`.

## Quick start

Run from `backend/` (the repository root README has the layout):

```bash
python3 -m venv .venv                               # Python 3.11 or newer
.venv/bin/pip install -e ".[dev]"

.venv/bin/stockintel --demo analyze RELIANCE         # offline, SYNTHETIC data, no network
.venv/bin/stockintel analyze RELIANCE                # live NSE data (Yahoo Finance + Google News)
.venv/bin/stockintel chat                            # multi-turn research conversation
.venv/bin/pytest                                     # both suites, ~10 s, fully offline
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

Global flags: `--db PATH` (default `backend/database/data/stockintel.db`), `--demo` (synthetic data).

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
- **Ranges**: the EWMA 80% range covered a median 79.2% (5-day), 78.6% (10-day) and 77.3% (20-day) of realized
  outcomes out of sample.
- **Open-source models** (`evaluate-models`, 12 large caps, walk-forward): Chronos-Bolt-small got
  5-day direction right 57.4% vs 56.5% for the base rate, and its 80% band held 79.9% vs 79.2% for
  the EWMA band, a tie. Kronos-small got 5-day direction right 49.2% vs 53.6%, with a 5.2% price
  error vs 2.2% for "no change". So Chronos is shown with that record, and Kronos is off by default.

## Layout

See the [root README](../README.md) for the folder map.

Decision-support only. Not investment advice. Verify figures against primary sources
(NSE/BSE filings) before acting.


## Run it on your own computer

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/stockintel app                    # opens the app in your browser, logged in
# optional, once: fill the Performance page and pattern track records (~1 minute)
.venv/bin/stockintel validate-patterns --universe --save && .venv/bin/stockintel build-performance
```

Windows: use `.venv\Scripts\pip` and `.venv\Scripts\stockintel`. Your data (portfolio, reports,
cache) stays in `backend/database/data/` on your machine; the login key and the agent's credentials
stay in `~/.stockintel/`, outside the project folder.

## Use it on your phone

**Same Wi-Fi as your computer (simplest):**
```bash
.venv/bin/stockintel app --lan
```
It prints a second link, `http://<your-computer's-IP>:8931/#key=…`. Open it on the phone, then
use *Share → Add to Home Screen* (iPhone) or *⋮ → Install app* (Android). Use this on your home
network only: the traffic is not encrypted, and anyone on the network who sees the key can log in.

**Anywhere:** deploy it (next section) and open the `https://…onrender.com/#key=…` link on the phone.

## Optional: open-source AI models and the self-learning agent

```bash
.venv/bin/pip install -e ".[ml]"                                    # torch + Chronos (~1 GB)
git clone https://github.com/shiyu-coder/Kronos ~/.stockintel/vendor/kronos   # Kronos code (optional)
.venv/bin/stockintel evaluate-models                                # measure both (~10 min, CPU)
```
With these installed, the chart shows the Chronos layer, and the Kronos toggle appears if the clone exists.

`stockintel train-chronos` and `stockintel train-kronos` fine-tune Chronos-2 and Kronos on NSE data (CPU,
about an hour each) and save the weights with their test records in `backend/models/artifacts/external/`;
the app serves them whenever they are there. ALFA's return generator and the path GRU need no extras.
`STOCKINTEL_ML=0` turns off the torch models and the analyst's language model.

The self-learning chat agent runs as its own process and needs about 4.5 GB of RAM:
```bash
.venv/bin/python -m backend.models.agent.run.serve --port 8010
```
StockIntel finds it at `http://127.0.0.1:8010`; set `SELFAGENT_URL` if it runs elsewhere.

## Put it on the web for free (a link for your phone)

Free hosting as checked in October 2026: Hugging Face no longer lets free accounts create Docker
Spaces, and Render's free plan has 512 MB of RAM and no disk. So the app is split three ways, all free:

| Part | Where | Why there |
|---|---|---|
| Website, signals, verdicts, charts, the path GRU | Render free web service (Docker) | Peaked at 222 MB over the six main pages without torch; sleeps after 15 idle minutes, wakes in about a minute |
| Chronos-2 and Kronos fine-tuned on NSE, ALFA's return generator, the analyst's language model | Your own ZeroGPU Space on Hugging Face (`deploy/hf-models-space`), weights in private model repos | Free accounts may host 2 ZeroGPU Spaces; 5 GPU-minutes a day, and each forecast takes seconds |
| Portfolio, reports, caches | A private Hugging Face dataset | Render's free disk is wiped on restart; the app restores on start and saves every 10 minutes and on shutdown |

**1. Hugging Face (once).** Your account must have a verified email and be at least 30 days old to host
a ZeroGPU Space. Push this repository's `main` to GitHub first (the Space installs the backend from there).
Create a second, read-only token for the Space, then:
```bash
.venv/bin/hf auth login                                         # paste a write token
HF_SPACE_TOKEN=<read-only token> .venv/bin/python ../deploy/publish_to_hf.py
```
It creates private repos for the fine-tuned Chronos-2 and Kronos weights and for our NumPy models (`alfa-weights`:
the return generator and the path GRU), uploads the model Space, sets its variables and read token, and asks for
ZeroGPU hardware.

**2. Save your current state from this computer** (portfolio, track record, price cache):
```bash
export STOCKINTEL_STATE_REPO=<you>/stockintel-state HF_TOKEN=<write token>
.venv/bin/stockintel state push                                 # creates the private dataset on first use
```

**3. Render.** Sign in at render.com with GitHub → *New → Blueprint* → pick this repository. It reads
`render.yaml`. Fill in `STOCKINTEL_STATE_REPO`, `STOCKINTEL_MODEL_SPACE` (`<you>/stockintel-models`),
`ALFA_MODELS_REPO` (`<you>/alfa-weights`) and `HF_TOKEN`. When it is live, copy `STOCKINTEL_API_KEY` from *Environment* and open
`https://<service>.onrender.com/#key=<that key>` on your phone, then *Add to Home Screen*.

Things to know:
- **Data sources:** Yahoo Finance sometimes rate-limits cloud servers (the app uses `curl_cffi`, which
  helps), and nseindia.com often blocks them, so run `stockintel insider-fetch` on your computer and then
  `stockintel state push`. Their terms restrict redistribution; keep the site for your own use.
- **The self-learning agent** needs about 5 GB of RAM, which no free host offers. It stays on your
  computer; the Ask AI toggle appears only when the app can reach it.
- **Keep the key secret:** it is the only login. One portfolio per copy of the app.
- **Not investment advice:** the app says this on every verdict; keep that if you publish it.

Any Docker host works too:
```bash
docker build -t financial-advisor .
docker run -p 8000:8000 -e STOCKINTEL_API_KEY=$(openssl rand -hex 24) -v advisor-data:/data financial-advisor
```

## Tests

`pytest -q` runs 227 offline tests (no network). GitHub Actions runs them on every push.
