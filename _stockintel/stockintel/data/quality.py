"""Data-quality gate: every OHLCV frame passes through here before analysis.

Fatal issues raise DataQualityError; recoverable ones come back as warnings
that engines attach to their results, so staleness and gaps stay visible in
the final answer instead of silently shaping it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from ..config import THRESHOLDS


class DataQualityError(Exception):
    pass


@dataclass
class QualityReport:
    warnings: List[str] = field(default_factory=list)
    rows: int = 0
    first_date: str = ""
    last_date: str = ""
    is_stale: bool = False


def check_ohlcv(df: pd.DataFrame, min_rows: int = 60,
                now: datetime = None) -> QualityReport:
    if df is None or df.empty:
        raise DataQualityError("OHLCV frame is empty — refusing to analyze nothing")
    if len(df) < min_rows:
        raise DataQualityError(
            f"only {len(df)} rows of history; need at least {min_rows} for stable indicators")
    if not df.index.is_monotonic_increasing:
        raise DataQualityError("OHLCV index is not chronological — possible corrupt feed")
    if df.index.has_duplicates:
        raise DataQualityError(f"{int(df.index.duplicated().sum())} duplicate dates in OHLCV index")

    report = QualityReport(rows=len(df),
                           first_date=str(df.index[0].date()),
                           last_date=str(df.index[-1].date()))

    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad_price.any():
        raise DataQualityError(f"{int(bad_price.sum())} rows with non-positive prices "
                               f"(first: {df.index[bad_price][0].date()})")
    inverted = df["high"] < df["low"]
    if inverted.any():
        raise DataQualityError(f"{int(inverted.sum())} rows where high < low")

    if (df["volume"] < 0).any():
        report.warnings.append("negative volume values present; volume analysis skipped for those rows")
    zero_vol = float((df["volume"] == 0).mean())
    if zero_vol > 0.10:
        report.warnings.append(f"{zero_vol:.0%} of sessions have zero volume — illiquid or bad feed")

    # A >40% single-day move usually means an unadjusted split, not a market event.
    jumps = df["close"].pct_change().abs()
    big = jumps[jumps > 0.40]
    if not big.empty:
        report.warnings.append(
            f"{len(big)} day(s) with >40% close-to-close moves (e.g. {big.index[0].date()}); "
            "verify corporate-action adjustment")

    gaps = df.index.to_series().diff().dt.days.dropna()
    long_gaps = gaps[gaps > 7]
    if not long_gaps.empty:
        report.warnings.append(f"{len(long_gaps)} calendar gap(s) longer than a week in the history")

    now = now or datetime.now()
    age = now - df.index[-1].to_pydatetime().replace(tzinfo=None)
    if age > timedelta(days=THRESHOLDS.stale_days):
        report.is_stale = True
        report.warnings.append(
            f"last bar is {age.days} days old — analysis reflects {report.last_date}, not today")
    return report
