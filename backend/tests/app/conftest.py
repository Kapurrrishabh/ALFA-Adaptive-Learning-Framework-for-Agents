from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backend.database.sources import synthetic
from backend.database.sources.local import LocalProvider
from backend.database.sources.provider import NewsItem
from backend.advisory.fusion.evidence import Provenance
from backend.database.storage import Store

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
END = "2026-09-25"


def bars(closes, spread=0.01, volume=1e6, end=END):
    """OHLCV frame from a close path; open = previous close."""
    closes = np.asarray(closes, float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    hi = np.maximum(opens, closes) * (1 + spread)
    lo = np.minimum(opens, closes) * (1 - spread)
    idx = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame({"open": opens, "high": hi, "low": lo, "close": closes,
                         "volume": np.full(len(closes), volume)}, index=idx)


def statements(revenue, net_income, eps, equity, debt, assets, ebit, interest,
               ca=None, cl=None, fcf=None, periods=None):
    periods = periods or [pd.Timestamp(f"{2026 - i}-03-31") for i in range(len(revenue))]
    income = pd.DataFrame({p: {"Total Revenue": r, "Net Income": n, "Diluted EPS": e, "EBIT": b,
                               "Operating Income": b, "Interest Expense": i_, "Pretax Income": b - i_,
                               "Tax Provision": (b - i_) * 0.25}
                           for p, r, n, e, b, i_ in zip(periods, revenue, net_income, eps, ebit, interest)})
    balance = pd.DataFrame({p: {"Stockholders Equity": q, "Total Debt": d, "Total Assets": a,
                                "Current Assets": c1, "Current Liabilities": c2,
                                "Cash And Cash Equivalents": d * 0.2}
                            for p, q, d, a, c1, c2 in zip(periods, equity, debt, assets,
                                                          ca or [None] * len(periods),
                                                          cl or [None] * len(periods))})
    out = {"income": income, "balance": balance}
    if fcf:
        out["cashflow"] = pd.DataFrame({p: {"Free Cash Flow": f} for p, f in zip(periods, fcf)})
    return out


def news_item(headline, days_ago, source="Reuters", summary="", symbol="TEST.NS"):
    ts = (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")
    return NewsItem(headline=headline, summary=summary, published_at=ts, source=source,
                    symbol=symbol, provenance=Provenance(source=f"test:{source}", as_of=ts))


@pytest.fixture
def random_walk():
    return synthetic.ohlcv(900, seed=11, end=END)


@pytest.fixture
def bench():
    return synthetic.ohlcv(900, seed=99, vol=0.009, end=END)


@pytest.fixture
def store():
    return Store(":memory:")


@pytest.fixture
def local_provider(random_walk, bench):
    p = LocalProvider("/nonexistent", frames={"TEST.NS": random_walk, "^NSEI": bench,
                                               "PEER.NS": synthetic.ohlcv(900, seed=5, end=END)})
    p.infos["TEST.NS"] = {"longName": "Test Industries Limited", "sector": "Industrials",
                          "currency": "INR", "currentPrice": float(random_walk["close"].iloc[-1]),
                          "marketCap": 5e11, "trailingPE": 20.0}
    p.statements["TEST.NS"] = statements(
        revenue=[1200, 1000, 900, 800], net_income=[150, 120, 100, 90], eps=[15, 12, 10, 9],
        equity=[1000, 900, 800, 700], debt=[300, 300, 300, 300], assets=[2000, 1800, 1600, 1500],
        ebit=[220, 180, 150, 130], interest=[20, 20, 20, 20], ca=[600] * 4, cl=[400] * 4,
        fcf=[140, 110, 90, 80])
    p.news_items["TEST.NS"] = [
        news_item("Test Industries Q2 results: net profit jumps 25%, beats estimates", 1),
        news_item("Test Industries wins large order from railways", 3, source="Economic Times"),
        news_item("Test Industries wins big order from railways", 3, source="Moneycontrol"),
        news_item("Test Industries faces probe over accounting lapses", 20, source="Mint"),
        news_item("Test Industries shares fall after weak guidance", 15, source="Business Standard"),
    ]
    return p
