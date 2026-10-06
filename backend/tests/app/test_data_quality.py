from datetime import datetime

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


def test_adjusted_close_adjusts_the_whole_bar_through_a_split():
    idx = pd.bdate_range(end="2026-09-25", periods=4)
    raw = pd.DataFrame({"Open": [200, 200, 100, 100], "High": [202, 202, 101, 101],
                        "Low": [198, 198, 99, 99], "Close": [200, 200, 100, 100],
                        "Adj Close": [100, 100, 100, 100], "Volume": [1, 1, 2, 2]}, index=idx, dtype=float)
    out = normalize_ohlcv(raw)
    assert ((out["close"] <= out["high"]) & (out["close"] >= out["low"])).all()
    assert out["open"].tolist() == [100, 100, 100, 100]
