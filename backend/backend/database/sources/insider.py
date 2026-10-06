"""NSE insider-trading (SEBI PIT) disclosures: promoter and director market trades.

Source: www.nseindia.com/api/corporates-pit (full JSON records, 2020 → Apr 2026;
from May 2026 NSE serves metadata + XBRL via corporates-pit-gg, parsed in
`fetch_xbrl_range`). The information time is the exchange broadcast timestamp
(`date`), not the trade date — trades are disclosed days later, and using the
trade date would leak future information into any backtest.
"""
from __future__ import annotations

import csv
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

from backend.database.sources.panel import CACHE_DIR
from backend.database.sources.provider import DataUnavailable

log = logging.getLogger("stockintel.insider")

API = "https://www.nseindia.com/api/corporates-pit"
API_GG = "https://www.nseindia.com/api/corporates-pit-gg"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/128.0 Safari/537.36",
           "Referer": "https://www.nseindia.com/", "Accept": "application/json"}
INSIDER_CATEGORIES = {"Promoters", "Promoter Group", "Director", "Promoter", "Key Managerial Personnel"}
MODES = {"Market Purchase": 1, "Market Sale": -1}
CACHE = CACHE_DIR / "insider_trades.csv"
FIELDS = ["broadcast", "symbol", "company", "category", "direction", "shares", "value", "trade_from",
          "trade_to", "pct_before", "pct_after", "pid"]
LEGACY_END = date(2026, 4, 30)


def _num(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def _parse_legacy(rows: List[Dict]) -> Iterator[Dict]:
    for r in rows:
        cat, mode = (r.get("personCategory") or "").strip(), (r.get("acqMode") or "").strip()
        if cat not in INSIDER_CATEGORIES or mode not in MODES or r.get("secType") != "Equity Shares":
            continue
        try:
            ts = datetime.strptime(r["date"], "%d-%b-%Y %H:%M")
        except (KeyError, ValueError):
            continue
        yield {"broadcast": ts.isoformat(), "symbol": r.get("symbol", "").strip(), "company": r.get("company", ""),
               "category": cat, "direction": MODES[mode], "shares": _num(r.get("secAcq")),
               "value": _num(r.get("secVal")), "trade_from": r.get("acqfromDt"), "trade_to": r.get("acqtoDt"),
               "pct_before": _num(r.get("befAcqSharesPer")), "pct_after": _num(r.get("afterAcqSharesPer")),
               "pid": r.get("pid")}


def _get(client, url: str, params: Dict[str, str]):
    for attempt in range(4):
        resp = client.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2 * (attempt + 1))
    raise DataUnavailable(f"{url} returned HTTP {resp.status_code} for {params}")


def fetch_range(start: date, end: date, path: Path = CACHE, step_days: int = 7,
                pause_s: float = 0.6) -> int:
    """Download legacy PIT JSON week by week and append filtered rows to `path`.
    Resumable: weeks already covered (by max broadcast date in the file) are skipped."""
    import httpx
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        existing = pd.read_csv(path, usecols=["pid"])
        seen = set(existing["pid"].astype(str))
    new_file = not path.exists()
    added = 0
    end = min(end, LEGACY_END)
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client, \
            open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        d = start
        while d <= end:
            to = min(d + timedelta(days=step_days - 1), end)
            data = _get(client, API, {"index": "equities", "from_date": d.strftime("%d-%m-%Y"),
                                      "to_date": to.strftime("%d-%m-%Y")})
            rows = data.get("data", []) if isinstance(data, dict) else data
            for row in _parse_legacy(rows):
                if str(row["pid"]) in seen:
                    continue
                seen.add(str(row["pid"]))
                writer.writerow(row)
                added += 1
            fh.flush()
            log.info("%s → %s: %d raw rows", d, to, len(rows))
            d = to + timedelta(days=1)
            time.sleep(pause_s)
    return added


def load(path: Path = CACHE) -> pd.DataFrame:
    if not path.exists():
        raise DataUnavailable(f"no insider data cached at {path}; run `stockintel insider-fetch`")
    df = pd.read_csv(path, parse_dates=["broadcast"])
    return df.drop_duplicates("pid")


def events(df: pd.DataFrame, direction: int = 1, min_value: float = 0.0,
           categories: Optional[set] = None) -> pd.DataFrame:
    """One event per (symbol, broadcast day): the net insider value that day."""
    sel = df[(df["direction"] == direction)]
    if categories:
        sel = sel[sel["category"].isin(categories)]
    sel = sel.assign(day=sel["broadcast"].dt.normalize(),
                     after_close=sel["broadcast"].dt.hour * 60 + sel["broadcast"].dt.minute >= 15 * 60 + 30)
    g = sel.groupby(["symbol", "day"]).agg(value=("value", "sum"), filings=("pid", "count"),
                                            after_close=("after_close", "max"),
                                            first_broadcast=("broadcast", "min")).reset_index()
    return g[g["value"] >= min_value]


def recent_buys(df: pd.DataFrame, days: int = 30, min_value: float = 1e7,
                today: Optional[date] = None) -> pd.DataFrame:
    today = today or date.today()
    ev = events(df, 1, min_value, {"Promoters", "Promoter Group", "Promoter"})
    return ev[ev["day"] >= pd.Timestamp(today - timedelta(days=days))].sort_values("value", ascending=False)


TAG = re.compile(r'<in-bse-co:([A-Za-z]+)[^>]*contextRef="(Disclosure\d+)"[^>]*>([^<]*)<')


def parse_xbrl(xml: str, symbol: str, broadcast: datetime, file_id: str) -> List[Dict]:
    """Rows in the same shape as the legacy JSON, one per disclosure context."""
    ctx: Dict[str, Dict[str, str]] = {}
    for tag, c, val in TAG.findall(xml):
        ctx.setdefault(c, {})[tag] = val.strip()
    out = []
    for c, f in ctx.items():
        cat, mode = f.get("CategoryOfPerson", ""), f.get("ModeOfAcquisitionOrDisposal", "")
        if cat not in INSIDER_CATEGORIES or mode not in MODES or f.get("TypeOfInstrument") != "Equity":
            continue
        pct = lambda k: _num(f.get(k)) * 100  # noqa: E731  (XBRL stores fractions)
        out.append({"broadcast": broadcast.isoformat(), "symbol": symbol, "company": "", "category": cat,
                    "direction": MODES[mode], "shares": _num(f.get("SecuritiesAcquiredOrDisposedNumberOfSecurity")),
                    "value": _num(f.get("SecuritiesAcquiredOrDisposedValueOfSecurity")),
                    "trade_from": f.get("DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate"),
                    "trade_to": f.get("DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate"),
                    "pct_before": pct("SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding"),
                    "pct_after": pct("SecuritiesHeldPostAcquistionOrDisposalPercentageOfShareholding"),
                    "pid": f"{file_id}:{c}"})
    return out


def fetch_recent(days: int = 45, path: Path = CACHE, pause_s: float = 0.3,
                 today: Optional[date] = None) -> int:
    """Post-April-2026 filings: metadata from corporates-pit-gg, details from each XBRL file.
    One request at a time; files already stored are skipped."""
    import httpx
    today = today or date.today()
    start = max(today - timedelta(days=days), LEGACY_END + timedelta(days=1))
    seen = set(pd.read_csv(path, usecols=["pid"])["pid"].astype(str)) if path.exists() else set()
    fetched_log = path.with_name("insider_xbrl_fetched.txt")
    fetched = set(fetched_log.read_text().split()) if fetched_log.exists() else set()
    new_file = not path.exists()
    added = 0
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client, \
            open(path, "a", newline="") as fh, open(fetched_log, "a") as flog:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        d = start
        while d <= today:
            to = min(d + timedelta(days=6), today)
            meta = _get(client, API_GG, {"index": "equities", "from_date": d.strftime("%d-%m-%Y"),
                                         "to_date": to.strftime("%d-%m-%Y")})
            rows = meta.get("data", []) if isinstance(meta, dict) else meta
            for m in rows:
                url, sym = m.get("xmlFileName"), (m.get("symbol") or "").strip()
                if not url or not sym:
                    continue
                file_id = url.rsplit("/", 1)[-1]
                if file_id in fetched:
                    continue
                try:
                    ts = datetime.strptime(m["broadcastDateTime"], "%d-%b-%Y %H:%M:%S")
                    resp = client.get(url)
                except (KeyError, ValueError, httpx.HTTPError) as exc:
                    log.warning("skipping %s: %s", url, exc)
                    continue
                if resp.status_code != 200:
                    continue
                fetched.add(file_id)
                flog.write(file_id + "\n")
                for row in parse_xbrl(resp.text, sym, ts, file_id):
                    if row["pid"] not in seen:
                        seen.add(row["pid"])
                        writer.writerow(row)
                        added += 1
                time.sleep(pause_s)
            fh.flush()
            d = to + timedelta(days=1)
    return added
