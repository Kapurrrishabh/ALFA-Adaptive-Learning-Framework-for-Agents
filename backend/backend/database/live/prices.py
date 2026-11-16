"""Daily OHLCV from Yahoo, in the one file format the collector and the agent both write.

Here rather than in `dataforge/`, which is deleted before submission: the agent fetches a share it holds no
file for while the question waits. yfinance manages its own HTTP session, so robots.txt is not consulted.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import yfinance

from backend.config import market_of
from backend.database.sources.provider import closed_bars

PERIOD = "10y"
COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


def download(tickers, period, timeout):
    return yfinance.download(tickers, period=period, interval="1d", auto_adjust=False, actions=False,
                             progress=False, threads=False, group_by="ticker", timeout=timeout)


def table_for(frame, ticker):
    """`ticker`'s closed bars in `frame`, without the bar of a session still open, whose close is the moving price."""
    table = frame[ticker] if isinstance(frame.columns, pd.MultiIndex) else frame
    table = table.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    table = table.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    table = table[list(COLUMNS)]
    table.index.name = "date"
    return closed_bars(table, market_of(ticker), datetime.now(timezone.utc))


def write(table, path):
    """Written whole or not at all: the agent's routes read these files while a fetch runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    table.to_csv(part, date_format="%Y-%m-%d")
    os.replace(part, path)


def fetch(ticker, path, timeout):
    frame = download([ticker], PERIOD, timeout)
    table = frame if frame.empty else table_for(frame, ticker)
    if table.empty:
        # yfinance answers an unreachable host and an unknown symbol alike, with an empty frame
        raise RuntimeError(f"yahoo returned no daily prices for {ticker}; it may be unreachable or not carry it")
    write(table, path)
