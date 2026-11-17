import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.database.sources.provider import DataUnavailable, normalize_ohlcv
from backend.database.sources.quality import DataQualityError, check_ohlcv
from backend.database.sources.local import LocalProvider

from .conftest import PANEL_SHARES, FakeYahoo, bars, cached_panel, refreshes, settled, until

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


@pytest.mark.parametrize("cached", [False, True])
def test_the_price_panel_drops_a_day_yahoo_sent_with_no_closes(monkeypatch, tmp_path, cached):
    """Yahoo sent 16 Nov 2026 with each share's open, high and low but no close, and ranked on it no share had a rank.
    A panel the older code had cached kept that row, and stopped every verdict until the cache expired."""
    import yfinance
    from backend.database.sources import panel
    days = pd.bdate_range(end="2026-11-16", periods=3)
    fields = {f: pd.DataFrame({"TCS.NS": [1.0, 2.0, 3.0], "INFY.NS": [4.0, 5.0, 6.0]}, index=days)
              for f in ("Open", "High", "Low", "Close", "Volume")}
    fields["Close"].iloc[-1] = np.nan
    monkeypatch.setattr(panel, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.concat(fields, axis=1))
    if cached:
        panel.load_panel(["TCS", "INFY"])
        cache, = tmp_path.glob("panel_*.pkl")
        old = {f: frame.rename(columns=lambda c: c[:-3]) for f, frame in fields.items()}
        pd.to_pickle(panel.Panel(old["Close"], old["Open"], old["High"], old["Low"], old["Volume"],
                                 {"TCS": "IT", "INFY": "IT"}, "yahoo"), cache)
        monkeypatch.setattr(yfinance, "download", None)     # so only the cache can answer
    p = panel.load_panel(["TCS", "INFY"])
    assert [list(frame.index) for frame in (p.close, p.open, p.high, p.low, p.volume)] == [list(days[:-1])] * 5


def last_close(p):
    return float(p.close.iloc[-1, 0])


@pytest.mark.parametrize("yahoo_down", [False, True])
def test_a_stale_panel_is_served_at_once_while_one_background_download_replaces_it(monkeypatch, tmp_path, yahoo_down):
    """Each request after the 12 hours waited 1.5 to 3 minutes for the download. Had each refresh waited its turn
    instead, with Yahoo down each would try the failing download again."""
    from backend.database.sources import panel
    yahoo = cached_panel(monkeypatch, tmp_path, panel.CACHE_TTL_S + 60)
    yahoo.gate.clear()
    yahoo.down = yahoo_down
    with ThreadPoolExecutor(8) as pool:
        served = [last_close(p) for p in pool.map(lambda _: panel.load_panel(PANEL_SHARES, serve_stale=True), range(8))]
    until(lambda: len(refreshes()) == 1)
    yahoo.gate.set()
    settled()
    assert (served, yahoo.calls) == ([1.0] * 8, 1)


def test_a_one_off_job_waits_for_the_download_rather_than_take_a_stale_panel(monkeypatch, tmp_path):
    """Calibration fitted on a panel up to 3 days old without a word, then exited and killed the refresh it started."""
    from backend.database.sources import panel
    cached_panel(monkeypatch, tmp_path, panel.CACHE_TTL_S + 60)
    assert (last_close(panel.load_panel(PANEL_SHARES)), refreshes()) == (2.0, [])


def test_a_refresh_asked_for_downloads_even_when_the_cached_panel_is_fresh(monkeypatch, tmp_path):
    """The hub's error log tells the operator to recover a bad panel with universe_panel(refresh=True)."""
    from backend.database.sources import panel
    yahoo = cached_panel(monkeypatch, tmp_path, 60)
    assert (last_close(panel.load_panel(PANEL_SHARES, refresh=True)), yahoo.calls) == (2.0, 1)


def test_with_no_cached_panel_two_readers_wait_for_one_download(monkeypatch, tmp_path):
    """On a first boot the hub's preload and a buy plan both downloaded: 6 minutes instead of 3."""
    import yfinance
    from backend.database.sources import panel
    yahoo = FakeYahoo(1.0)
    yahoo.gate.clear()
    monkeypatch.setattr(panel, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(yfinance, "download", yahoo)
    with ThreadPoolExecutor(2) as pool:
        readers = [pool.submit(panel.load_panel, PANEL_SHARES) for _ in range(2)]
        until(lambda: yahoo.calls == 1)
        yahoo.gate.set()
        served = [last_close(r.result()) for r in readers]
    assert (served, yahoo.calls) == ([1.0, 1.0], 1)


def write_the_panel_with_another_process(cache_dir, both_written):
    """Writes the panel, as the prices job does beside the app, and replaces the cache only once both have written."""
    import yfinance
    from backend.database.sources import panel
    write = pd.to_pickle

    def write_then_wait(obj, path):
        write(obj, path)
        both_written.wait()
    panel.CACHE_DIR, yfinance.download, pd.to_pickle = cache_dir, FakeYahoo(2.0), write_then_wait
    panel.load_panel(PANEL_SHARES, refresh=True)


def test_two_processes_writing_the_panel_at_once_both_finish_and_leave_it_whole(tmp_path):
    """Each process has its own lock, and they shared one temp file: the second replace found none, and a reader
    between the two writes opened half a file."""
    from backend.database.sources import panel
    spawn = multiprocessing.get_context("spawn")
    both_written = spawn.Barrier(2, timeout=60)
    writers = [spawn.Process(target=write_the_panel_with_another_process, args=(tmp_path, both_written))
               for _ in range(2)]
    for w in writers:
        w.start()
    for w in writers:
        w.join(120)
    assert ([w.exitcode for w in writers], last_close(panel.load_panel(PANEL_SHARES))) == ([0, 0], 2.0)


def test_a_failed_refresh_is_logged_and_the_stale_panel_keeps_serving(monkeypatch, tmp_path, caplog):
    from backend.database.sources import panel
    yahoo = cached_panel(monkeypatch, tmp_path, panel.CACHE_TTL_S + 60)
    yahoo.down = True
    panel.load_panel(PANEL_SHARES, serve_stale=True)
    settled()
    served = last_close(panel.load_panel(PANEL_SHARES, serve_stale=True))
    settled()
    failed = [r for r in caplog.records if r.levelname == "ERROR" and "is still served" in r.getMessage()]
    assert (served, len(failed), str(failed[0].exc_info[1])) == (1.0, 2, "batch download returned no data")


@pytest.mark.parametrize("yahoo_down", [False, True])
def test_a_panel_past_the_stale_limit_is_refused_and_the_reader_waits_for_a_download(monkeypatch, tmp_path,
                                                                                     yahoo_down):
    from backend.database.sources import panel
    yahoo = cached_panel(monkeypatch, tmp_path, panel.STALE_LIMIT_S + 60)
    yahoo.down = yahoo_down
    if yahoo_down:
        with pytest.raises(DataUnavailable, match="batch download returned no data"):
            panel.load_panel(PANEL_SHARES, serve_stale=True)
    else:
        assert last_close(panel.load_panel(PANEL_SHARES, serve_stale=True)) == 2.0


def test_a_refresh_that_fails_while_writing_leaves_the_cached_panel_whole(monkeypatch, tmp_path):
    """Readers open the cache while a refresh writes it, so a half-written file would stop every one of them."""
    from backend.database.sources import panel
    cached_panel(monkeypatch, tmp_path, panel.CACHE_TTL_S + 60)

    def disk_full(obj, path, *a, **kw):
        with open(path, "wb") as half:
            half.write(b"\x80\x05half a pickle")
        raise OSError("No space left on device")
    monkeypatch.setattr(pd, "to_pickle", disk_full)
    panel.load_panel(PANEL_SHARES, serve_stale=True)
    settled()
    assert (list(tmp_path.glob("*.part")), last_close(panel.load_panel(PANEL_SHARES, serve_stale=True))) == ([], 1.0)
    settled()


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
