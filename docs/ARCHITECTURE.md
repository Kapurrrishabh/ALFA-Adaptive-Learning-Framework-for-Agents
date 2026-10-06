# StockIntel architecture

Four pictures: what the parts are, where they run, what happens when you open a stock, and how
the models are trained and tested. A table of every model follows. Images are in
`docs/architecture/`; each one's source is the Mermaid block under it (the first one's is an HTML file).

## 1. The parts

![overview](architecture/overview.png)

Source: `architecture/overview.html` (a layered block diagram reads better than an auto-laid-out graph here).


## 2. Where it runs

![deployment](architecture/deployment.png)

```mermaid
flowchart LR
  subgraph L["On your own computer: everything"]
    L1["stockintel app<br/>web + API + engines + torch models"]
    L2["backend.models.agent.run.serve :8010<br/>chat agent"]
    L1 -- "HTTP + WebSocket" --> L2
  end
  subgraph R["Free cloud setup"]
    PH["Phone browser"] -- HTTPS --> RD["Render free web service<br/>StockIntel and the path GRU, no torch (222 MB peak)"]
    RD -- "gradio_client" --> SP["HF ZeroGPU Space: stockintel-models<br/>Chronos-2-NSE, Kronos-NSE, analyst writer on GPU<br/>ALFA return model on CPU"]
    SP -- "loads weights" --> MR["HF private model repos<br/>stockintel-chronos2-nse · stockintel-kronos-nse<br/>alfa-weights"]
    RD -- "GRU weights on start" --> MR
    RD <-- "restore on start, save every 10 min" --> DS[("HF private dataset<br/>stockintel-state")]
    RD -. "only if reachable" .-> AG["ALFA chat agent<br/>your computer, or a paid HF PRO Space"]
  end
```

## 3. Opening a stock: "Should I buy TCS?"

![request flow](architecture/request-flow.png)

```mermaid
sequenceDiagram
  participant U as You
  participant A as API
  participant H as Hub
  participant S as Service
  participant D as Yahoo · News · NSE
  participant E as Engines + fusion
  participant G as Signal system
  participant F as Forecast models
  U->>A: open TCS
  A->>H: overview, chart, verdict
  H->>S: analyse TCS
  S->>D: prices, fundamentals, news
  S->>E: run the engines, fuse their evidence
  E-->>H: research label + evidence for and against
  H->>G: momentum rank of TCS in the Nifty 200
  G-->>H: BUY / WAIT / DON'T BUY, stop-loss, next review
  H-->>U: verdict with "why buy" and "why not"
  U->>A: chart layers
  A->>H: technical (patterns, candles, range) and AI fans
  H->>F: ALFA fan, GRU line, Chronos / Kronos (in process, or the Space)
  F-->>U: fan, GRU median and paths, and candles, each with its tested record
  U->>A: analyst note
  A->>H: our numbers for TCS, judged in words by our code
  H->>F: language model rewrites the computed note
  F-->>H: draft
  H-->>U: the draft if every check passes, else the computed note and the reason
```

## 4. Training and testing

![training](architecture/training.png)

```mermaid
flowchart LR
  P[("Price panel<br/>Nifty 500, 10 years, cached")]
  P --> TC["train-chronos<br/>train ≤ 2021 · choose on 2022–23<br/>retrain ≤ 2023 · test once on 2024–26"]
  P --> TK["train-kronos<br/>same splits, tokenizer frozen"]
  P --> TG["train_path_gru (ours, NumPy)<br/>same splits, early stopping on 2022–23"]
  TG --> GW["path_gru.npz<br/>weights + its test record"]
  GW --> HF
  W["compare_writers<br/>4 open language models, same 24 notes"] --> LMC["chosen writer<br/>by faithfulness checks"]
  P --> EV["validate-patterns · evaluate-signals<br/>build-performance · evaluate-models"]
  ALFA["ALFA project training<br/>return generator vs GARCH"] --> AW["returns.npz<br/>weights + its own test record"]
  TC --> CW["chronos-2-nse<br/>weights + stockintel_record.json"]
  TK --> KW["kronos-nse<br/>weights + stockintel_record.json"]
  EV --> J["performance.json · model_evals.json"]
  CW --> HF["HF model repos"]
  KW --> HF
  AW --> HF
  J --> UI["Performance page · chart notes · verdicts"]
  HF --> UI
```

Every figure the app shows is read from these records; when a record is missing, the app
says so instead of showing a number.

## 5. How ALFA (the other project) is trained

![ALFA training](architecture/alfa-training.png)

```mermaid
flowchart TB
  subgraph D["1 · Data (dataforge)"]
    CORP["Text corpus<br/>SEC filings · Fed · RBI · SEBI · news<br/>Wikipedia · StackExchange · textbooks"]
    PRICES["Daily prices<br/>101 instruments, about 50 on NSE"]
  end
  subgraph P["2 · Pretraining"]
    TOK["prepare_pretrain<br/>train tokenizer · encode corpus"]
    PRE["train_pretrain<br/>NumPy transformer encoder"]
  end
  subgraph R["3 · Retrieval"]
    IDX["build_index / refresh_index<br/>chunk dated documents · embed with the frozen encoder"]
  end
  subgraph G["4 · Answer generator"]
    DS["prepare_generator · prepare_advisory<br/>question + retrieved evidence + answer<br/>(source thread excluded; 1 in 7 must refuse)"]
    SFT["train_generator<br/>supervised, teacher forcing"]
    PREF["train_preference<br/>which of two answers readers preferred"]
    RAFT["train_verified<br/>write k answers, keep those the evidence supports"]
  end
  subgraph J["5 · Judges and gates"]
    JUDGE["label_feedback · judge_preference<br/>oracle vs Claude judge, and a human check"]
    GATES["route · fit_abstention · calibrate_confidence<br/>intent guard · answer cut · confidence map"]
  end
  subgraph M["Price models"]
    HEAD["train_price_head<br/>GRU vs transformer tower<br/>direction 35.7% vs 36.0% baseline"]
    RET["train_return_generator<br/>train · select · test by date<br/>vs GARCH(1,1)-t and a volatility Gaussian"]
  end
  PROMO["promote → registry<br/>ships only if its own numbers clear the bar"]
  SERVE["serve.py · chat agent<br/>answers only above the cut, says why otherwise"]
  LOG[("Served turns<br/>feedback log")]

  CORP --> TOK --> PRE --> IDX
  PRE --> SFT
  IDX --> DS --> SFT --> PREF --> RAFT --> PROMO
  JUDGE --> GATES --> SERVE
  PROMO --> SERVE
  PRICES --> HEAD --> SERVE
  PRICES --> RET --> SERVE
  RET -. "returns.npz" .-> SI["StockIntel<br/>'our model' fan"]
  SERVE --> LOG --> JUDGE
```

Steps 1–4 teach the agent to answer from evidence. Step 5 decides when it may answer. The loop at
the bottom (served turns → judges → gates) is the self-learning part: judged answers refit the answer
cut, and unjudged ones are stored but do not move it.

## 6. Every model

| Model | What it is | Trained by | Runs | Used for | Measured out of sample |
|---|---|---|---|---|---|
| NSE momentum rank | Nifty200 Momentum 30 formula: z of 12- and 6-month return ÷ 1-year volatility | Formula, not trained | Server | The buy/sell signal | Live ETF beat the Nifty 500 by +1.9 pts/yr (2022–26), unevenly |
| Candle and chart-pattern rules | 15 candle rules, 19 chart-pattern shapes (28 with breakout direction) | Rules | Server | Drawn on charts as context | 0 of 43 show an edge |
| Logistic regression, gradient boosting | scikit-learn direction forecasts | Walk-forward at run time | Server | Research engine | No skill; weight set to 0 |
| EWMA volatility band | Volatility-scaled range | Formula | Server | Grey range on charts | Its 80% band held 77–83% across our tests |
| Chronos-Bolt-small | Amazon time-series model | Amazon (zero-shot) | Server or Space | Optional range layer | Range worse than the band; direction no better than the base rate |
| **Chronos-2-NSE** | AutoGluon Chronos-2-small (28M parameters), fine-tuned on NSE | **Us** | Server or Space | Range layer | Range −4.5% / −0.6% vs band (5 / 20 days); direction 48.9% / 52.0% vs 51.7% / 53.4%. At 10 days: range +0.9%, direction 53.0% vs 50.5% (+0.5 to +4.7 pts grouped by date), tentative because several horizons and models were tried |
| Kronos-small | Candlestick model + tokenizer | NeoQuasar (zero-shot) | Server or Space | Experimental candles | Worse than "no change" (5.2% vs 2.2% error) |
| **Kronos-NSE** | Kronos-small predictor fine-tuned on NSE | **Us** | Server or Space | Experimental candles | Error 4.0% vs 3.3% for "no change" (untuned: 9.7%); direction 48.8% vs 53.2% |
| **Path GRU** | GRU on ALFA's NumPy framework: 60 days in, a mean and spread for each of the next 10 days out | **Us** | Server (NumPy) | Dashed median and thin possible paths beside the median dots | Daily move sizes beat the EWMA band (likelihood +0.019 to +0.058 per day, grouped by date); 10-day error 4.71% vs 4.68% for "no change"; direction 49.5% vs 50.6%, within noise |
| **ALFA return generator** | NumPy transformer, 875,776 parameters | **Us (ALFA)** | Server (local) or Space (CPU) | Gold "our model" fan | Beat GARCH(1,1)-t by 0.032 ± 0.003 nats per return over 80,576 returns; no direction edge |
| GARCH(1,1)-t | Classic volatility model | ALFA, same run | Inside ALFA's loader | Draws the fan if the generator stops beating it | Baseline |
| **ALFA chat agent** | NumPy transformer generator + retrieval index + price head + answer gate | **Us (ALFA)** | Its own server, about 5 GB RAM | "Also ask my self-learning agent" | Answers only above its fitted confidence cut |
| Claude (optional) | Narrator / tool-using agent | Anthropic | Anthropic API or Bedrock | Rewrites answers; every number checked against the evidence | Off by default |
