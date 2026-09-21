"""Daily OHLCV via yfinance. No key needed.

yfinance manages its own HTTP session, so the shared PoliteSession is unused here and robots.txt
is not consulted for this source. Tickers are fetched in batches to keep the request count low.
"""

import pandas as pd
import yfinance

from .. import config

BATCH_SIZE = 10
COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
LICENCE = "Yahoo Finance terms; personal and research use"
SOURCE = "yahoo_prices"


def _table_for(frame, ticker):
    table = frame[ticker] if isinstance(frame.columns, pd.MultiIndex) else frame
    table = table.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    table = table.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    table = table[list(COLUMNS)]
    table.index.name = "date"
    return table


def collect(session, manifest, settings, log):
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    price_range = settings["price_range"]
    fetched = 0

    tickers = config.TICKERS + config.INDIA_TICKERS
    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start : start + BATCH_SIZE]
        frame = yfinance.download(
            batch,
            period=price_range,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        if frame.empty:
            log(f"  {' '.join(batch)}: nothing returned")
            continue

        for ticker in batch:
            try:
                table = _table_for(frame, ticker)
            except KeyError:
                log(f"  {ticker}: not in the response")
                continue
            if table.empty:
                log(f"  {ticker}: no usable rows")
                continue

            # Prices are rewritten rather than skipped, so each run picks up days that have
            # closed since the last one.
            destination = config.PRICES_DIR / f"{ticker}.csv"
            table.to_csv(destination, date_format="%Y-%m-%d")
            manifest.record(
                f"yfinance://{ticker}?period={price_range}", destination, SOURCE, LICENCE
            )
            fetched += 1
            log(f"  {ticker}: {len(table)} days {table.index[0].date()}..{table.index[-1].date()}")

    return fetched
