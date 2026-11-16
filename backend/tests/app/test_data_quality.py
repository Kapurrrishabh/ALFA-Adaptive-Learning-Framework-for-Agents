from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.database.sources.provider import DataUnavailable, normalize_ohlcv
from backend.database.sources.quality import DataQualityError, check_ohlcv
from backend.database.sources.local import LocalProvider

from .conftest import bars

NOW = datetime(2026, 9, 26)


def test_empty_frame_is_rejected():
    with pytest.raises(DataQualityError, match="empty"):
        check_ohlcv(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))


def test_too_little_history_is_rejected():
    with pytest.raises(DataQualityError, match="rows of history"):
        check_ohlcv(bars(np.linspace(10, 11, 30)), now=NOW)


def test_duplicate_dates_are_rejected():
    df = bars(np.linspace(10, 11, 80))
    df = pd.concat([df, df.iloc[[-1]]])
    with pytest.raises(DataQualityError):
        check_ohlcv(df, now=NOW)


def test_non_chronological_index_is_rejected():
    df = bars(np.linspace(10, 11, 80)).iloc[::-1]
    with pytest.raises(DataQualityError, match="chronological"):
        check_ohlcv(df, now=NOW)


def test_non_positive_prices_are_rejected():
    df = bars(np.linspace(10, 11, 80))
    df.iloc[5, df.columns.get_loc("low")] = 0.0
    with pytest.raises(DataQualityError, match="non-positive"):
        check_ohlcv(df, now=NOW)


def test_high_below_low_is_rejected():
    df = bars(np.linspace(10, 11, 80))
    df.iloc[5, df.columns.get_loc("high")] = df["low"].iloc[5] * 0.5
    with pytest.raises(DataQualityError, match="high < low"):
        check_ohlcv(df, now=NOW)


def test_unadjusted_split_is_flagged_not_silently_accepted():
    closes = np.concatenate([np.full(60, 1000.0), np.full(40, 500.0)])
    rep = check_ohlcv(bars(closes), now=NOW)
    assert any("corporate-action" in w for w in rep.warnings)


def test_stale_data_is_flagged():
    df = bars(np.linspace(10, 11, 80), end="2026-08-01")
    rep = check_ohlcv(df, now=NOW)
    assert rep.is_stale and any("days old" in w for w in rep.warnings)


def test_fresh_clean_data_passes_without_warnings():
    rep = check_ohlcv(bars(np.linspace(10, 11, 80)), now=NOW)
    assert rep.warnings == [] and not rep.is_stale


def test_normalize_prefers_adjusted_close_and_drops_missing_closes():
    idx = pd.bdate_range(end="2026-09-25", periods=3)
    raw = pd.DataFrame({"Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0.5, 1, 2], "Close": [1.5, 2.5, None],
                        "Adj Close": [1.4, 2.4, None], "Volume": [10, 20, 30]}, index=idx)
    out = normalize_ohlcv(raw)
    assert out["close"].tolist() == [1.4, 2.4]


def test_normalize_rejects_frames_missing_columns():
    with pytest.raises(DataUnavailable, match="missing columns"):
        normalize_ohlcv(pd.DataFrame({"close": [1.0]}))


def test_local_provider_raises_instead_of_inventing_missing_data(tmp_path):
    p = LocalProvider(str(tmp_path))
    for call in (lambda: p.ohlcv("NOPE"), lambda: p.info("NOPE"), lambda: p.financials("NOPE"),
                 lambda: p.news("NOPE")):
        with pytest.raises(DataUnavailable):
            call()


@pytest.mark.parametrize("fetch", ["info", "financials", "upcoming_events", "news"])
def test_yahoos_own_error_text_goes_to_the_log_not_to_the_user(monkeypatch, caplog, fetch):
    """yfinance raised this for NYKAA's news, and the analysis showed it to the user as its main doubt."""
    from yfinance.exceptions import YFDataException
    from backend.database.sources import yahoo

    class Down:
        def __init__(self, symbol):
            pass

        def __getattr__(self, name):
            raise YFDataException("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***")
    monkeypatch.setattr(yahoo, "_yf", lambda: SimpleNamespace(Ticker=Down))
    with pytest.raises(DataUnavailable) as raised:
        getattr(yahoo.YahooProvider(), fetch)("NYKAA.NS")
    assert ("NYKAA.NS" in str(raised.value), "CURRENTLY DOWN" in str(raised.value), "CURRENTLY DOWN" in caplog.text) \
        == (True, False, True)


def test_the_price_panel_drops_a_day_yahoo_sent_with_no_closes(monkeypatch, tmp_path):
    """Yahoo sent 16 Nov 2026 with each share's open, high and low but no close, and ranked on it no share had a rank."""
    import yfinance
    from backend.database.sources import panel
    days = pd.bdate_range(end="2026-11-16", periods=3)
    fields = {f: pd.DataFrame({"TCS.NS": [1.0, 2.0, 3.0], "INFY.NS": [4.0, 5.0, 6.0]}, index=days)
              for f in ("Open", "High", "Low", "Close", "Volume")}
    fields["Close"].iloc[-1] = np.nan
    monkeypatch.setattr(panel, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.concat(fields, axis=1))
    p = panel.load_panel(["TCS", "INFY"])
    assert [list(frame.index) for frame in (p.close, p.open, p.high, p.low, p.volume)] == [list(days[:-1])] * 5


def test_adjusted_close_adjusts_the_whole_bar_through_a_split():
    idx = pd.bdate_range(end="2026-09-25", periods=4)
    raw = pd.DataFrame({"Open": [200, 200, 100, 100], "High": [202, 202, 101, 101],
                        "Low": [198, 198, 99, 99], "Close": [200, 200, 100, 100],
                        "Adj Close": [100, 100, 100, 100], "Volume": [1, 1, 2, 2]}, index=idx, dtype=float)
    out = normalize_ohlcv(raw)
    assert ((out["close"] <= out["high"]) & (out["close"] >= out["low"])).all()
    assert out["open"].tolist() == [100, 100, 100, 100]


@pytest.mark.parametrize("writer", ["the collector", "ALFA's fetch"])
@pytest.mark.parametrize("now, tcs_ends, aapl_ends", [
    ("2026-11-02T14:26+05:30", "2026-10-30", "2026-10-30"),     # NSE's session open, and the US's not begun
    ("2026-11-02T15:00-05:00", "2026-11-02", "2026-10-30"),     # the US's open, and NSE's closed
    ("2026-11-02T16:30-05:00", "2026-11-02", "2026-11-02"),     # both closed
])
def test_no_price_writer_writes_a_bar_for_a_session_still_open(tmp_path, monkeypatch, writer, now, tcs_ends, aapl_ends):
    """The held TCS file ended on 2,505.00 for 2 Nov, written at 14:26 IST; that day closed at 2,501.70. ALFA's own
    fetch of a share it holds no file for wrote the same moving price."""
    from backend.database.dataforge import config
    from backend.database.dataforge.sources import yahoo_prices

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(now).astimezone(tz)
    bar = {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Adj Close": 1.0, "Volume": 1.0}
    days = pd.bdate_range(end="2026-11-02", periods=3)
    frame = pd.concat({ticker: pd.DataFrame(bar, index=days) for ticker in ("TCS.NS", "AAPL")}, axis=1)
    monkeypatch.setattr(yahoo_prices.prices, "datetime", Clock)
    monkeypatch.setattr(yahoo_prices.prices, "download", lambda *_: frame)
    monkeypatch.setattr(config, "PRICES_DIR", tmp_path)
    monkeypatch.setattr(config, "TICKERS", ["AAPL"])
    monkeypatch.setattr(config, "INDIA_TICKERS", ["TCS.NS"])
    if writer == "the collector":
        yahoo_prices.collect(None, SimpleNamespace(record=lambda *_: None), {"price_range": "5d"}, lambda _: None)
    else:
        for ticker in ("TCS.NS", "AAPL"):
            yahoo_prices.prices.fetch(ticker, tmp_path / f"{ticker}.csv", 1)
    assert [pd.read_csv(tmp_path / f"{ticker}.csv")["date"].iloc[-1] for ticker in ("TCS.NS", "AAPL")] == \
        [tcs_ends, aapl_ends]
