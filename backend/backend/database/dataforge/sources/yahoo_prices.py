"""Daily OHLCV via yfinance. No key needed.

yfinance manages its own HTTP session, so the shared PoliteSession is unused here and robots.txt
is not consulted for this source. Tickers are fetched in batches to keep the request count low.
"""

from backend.database.dataforge import config
from backend.database.live import prices

BATCH_SIZE = 10
# yfinance's own default, which this collector has always run with
TIMEOUT_S = 10
LICENCE = "Yahoo Finance terms; personal and research use"
SOURCE = "yahoo_prices"


def collect(session, manifest, settings, log):
    config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
    price_range = settings["price_range"]
    fetched = 0

    tickers = config.TICKERS + config.INDIA_TICKERS
    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start : start + BATCH_SIZE]
        frame = prices.download(batch, price_range, TIMEOUT_S)
        if frame.empty:
            log(f"  {' '.join(batch)}: nothing returned")
            continue

        for ticker in batch:
            try:
                table = prices.table_for(frame, ticker)
            except KeyError:
                log(f"  {ticker}: not in the response")
                continue
            if table.empty:
                log(f"  {ticker}: no usable rows")
                continue

            # Prices are rewritten rather than skipped, so each run picks up days that have
            # closed since the last one.
            destination = config.PRICES_DIR / f"{ticker}.csv"
            prices.write(table, destination)
            manifest.record(
                f"yfinance://{ticker}?period={price_range}", destination, SOURCE, LICENCE
            )
            fetched += 1
            log(f"  {ticker}: {len(table)} days {table.index[0].date()}..{table.index[-1].date()}")

    return fetched
