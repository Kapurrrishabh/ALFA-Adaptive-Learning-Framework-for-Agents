"""Indian mutual funds: AMFI's daily NAV file for the scheme list, and mfapi.in for NAV history.

AMFI publishes every scheme's latest NAV at NAVAll.txt and rewrites it during the day, so it is
downloaded at most once a calendar day. mfapi.in is unofficial and publishes no terms of use; it is
used only for NAV history, and every response that carries its figures says so.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from backend.database.sources.panel import CACHE_DIR
from backend.database.sources.provider import DataUnavailable

NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
HISTORY_URL = "https://api.mfapi.in/mf/{code}"
NAV_SOURCE = "AMFI (amfiindia.com NAVAll.txt)"
HISTORY_SOURCE = "mfapi.in (unofficial: no published terms of use)"
# in folders because hfstate saves and restores the cache's top-level files, and a restored copy
# would look downloaded today
NAV_FILE = CACHE_DIR / "amfi" / "NAVAll.txt"
HISTORY_DIR = CACHE_DIR / "mfapi"
# provisional: closed schemes stay in the file at their last NAV, some from 2018. On 2026-10-30 a week
# kept 8,807 of 14,441 priced rows (8,772 priced that day), so a scheme can miss a few NAVs and stay
LIVE_DAYS = 7
TIMEOUT_S, ATTEMPTS = 20, 2       # provisional: the 1.5 MB file took under a second on 2026-10-30
# provisional: a failed download takes up to about 40 s, and in an outage each request would wait for its own
RETRY_AFTER_S = 300
MIN_QUERY_CHARS = 2               # two-letter fund houses exist: JM, NJ

COLUMNS = {"code": "Scheme Code", "isin_growth": "ISIN Div Payout/ ISIN Growth",
           "isin_reinvestment": "ISIN Div Reinvestment", "fund": "Scheme Name", "plan": "Plan",
           "option": "Option", "nav": "Net Asset Value", "date": "Date"}
CATEGORY_LINE = re.compile(r"^([A-Za-z ]+Schemes)\s*\((.*)\)$")
NO_NAV = "N.A."
DIRECT = "Direct Plan"
SYNONYMS = {"absl": "aditya birla sun life"}
ISIN = re.compile(r"^INF[A-Z0-9]{9}$")
NOTE = re.compile(r"\([^)]*\)")

_LOCKS: Dict[Path, threading.Lock] = {}
_READ: Dict[Path, Tuple[int, List["Scheme"]]] = {}
_FAILED: Dict[Path, Tuple[float, str]] = {}


@dataclass(frozen=True)
class Scheme:
    """One AMFI scheme code: a fund in one plan (Direct or Regular) and one option (Growth, IDCW...)."""
    code: int
    isin_growth: Optional[str]
    isin_reinvestment: Optional[str]
    fund: str
    plan: Optional[str]
    option: Optional[str]
    nav: float
    nav_date: date
    scheme_type: str
    category: str
    amc: str

    @property
    def is_growth(self) -> bool:
        return any(word in (self.option or "").lower() for word in ("growth", "cumulative"))  # ICICI calls it Cumulative


def parse(text: str) -> List[Scheme]:
    """Every priced row of NAVAll.txt, read by AMFI's column names. Category lines ("Open Ended
    Schemes(Equity Scheme - Mid Cap Fund)") and fund-house lines head the rows below them."""
    lines = text.splitlines()
    header = [c.strip() for c in (lines[0] if lines else "").split(";")]
    missing = [c for c in COLUMNS.values() if c not in header]
    if missing:
        raise DataUnavailable(f"AMFI's NAV file has no {missing} column; its header is {header}. "
                              "The format changed: update amfi.COLUMNS.")
    at = {key: header.index(name) for key, name in COLUMNS.items()}
    # where a scheme has no ISIN AMFI writes "-", "Redeemed" or other text (152713: "HDFCNIVODG")
    isin = lambda f, key: f[at[key]] if ISIN.match(f[at[key]]) else None  # noqa: E731
    out: List[Scheme] = []
    scheme_type = category = amc = None
    for number, raw in enumerate(lines[1:], start=2):
        # Bandhan writes "Bandhan Gilt Fund" with 1 to 4 spaces between words, which would make one fund four
        line = " ".join(raw.split())
        if not line:
            continue
        if ";" not in line:
            heading = CATEGORY_LINE.match(line)
            if heading:     # the fund house is kept: Franklin's rows follow a category line with no house line
                scheme_type, category = heading.group(1), heading.group(2).strip()
            else:
                amc = line
            continue
        f = [x.strip() for x in line.split(";")]
        if len(f) != len(header) or amc is None or category is None:
            raise DataUnavailable(f"line {number} of AMFI's NAV file is not a scheme under a category and "
                                  f"fund house: {line[:120]!r}")
        if f[at["nav"]] == NO_NAV:
            continue
        try:
            out.append(Scheme(code=int(f[at["code"]]), isin_growth=isin(f, "isin_growth"),
                              isin_reinvestment=isin(f, "isin_reinvestment"), fund=f[at["fund"]],
                              plan=f[at["plan"]] or None, option=f[at["option"]] or None, nav=float(f[at["nav"]]),
                              nav_date=datetime.strptime(f[at["date"]], "%d-%b-%Y").date(),
                              scheme_type=scheme_type, category=category, amc=amc))
        except ValueError as exc:
            raise DataUnavailable(f"line {number} of AMFI's NAV file could not be read ({exc}): {line[:120]!r}") from exc
    if not out:
        raise DataUnavailable("AMFI's NAV file lists no priced scheme")
    return out


def current(schemes: List[Scheme], days: int = LIVE_DAYS) -> List[Scheme]:
    """The schemes priced within `days` of the newest NAV in the file."""
    newest = max(s.nav_date for s in schemes)
    return [s for s in schemes if s.nav_date >= newest - timedelta(days=days)]


def name_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("&", " and "))


def _index(name: str) -> Set[str]:
    """A name's words plus each neighbouring pair joined, so "Mid Cap" also reads "midcap"."""
    w = name_words(name)
    return set(w + [a + b for a, b in zip(w, w[1:])])


def _plan_words(s: Scheme) -> List[str]:
    return name_words(f"{s.plan or ''} {s.option or ''}")


def _plan_order(s: Scheme, asked: List[str]) -> tuple:
    named = all(any(w.startswith(t) for w in _plan_words(s)) for t in asked)
    return not named, s.plan != DIRECT, not s.is_growth, s.plan or "", s.option or "", s.code


def _matched(tokens: List[str], words: Set[str], searchable: Set[str]) -> Tuple[bool, int]:
    """(whether every query word starts a searchable word, how many are whole words of the fund's name). Two
    neighbouring query words may also be found as one: "mid cap" finds "DSP Midcap Fund"."""
    found = [any(w.startswith(t) for w in searchable) for t in tokens]
    whole = [t in words for t in tokens]
    for i, pair in enumerate(a + b for a, b in zip(tokens, tokens[1:])):
        if any(w.startswith(pair) for w in searchable):
            found[i] = found[i + 1] = True
        if pair in words:
            whole[i] = whole[i + 1] = True
    return all(found), sum(whole)


def query_words(query: str) -> List[str]:
    """The words `search` looks for, with abbreviations spelt out. ValueError if there is too little to search."""
    words = [t for w in name_words(query) for t in SYNONYMS.get(w, w).split()]
    if len(query.strip()) < MIN_QUERY_CHARS or not words:
        raise ValueError(f"type at least {MIN_QUERY_CHARS} letters of the fund's name; got {query!r}")
    return words


def search(schemes: List[Scheme], query: str, limit: int = 10) -> List[Dict[str, object]]:
    """Funds whose name or fund house has a word starting with each word of `query`, or the scheme with that ISIN.
    Best first: more words matched whole in the fund's name, then shorter names. Each fund lists its plans,
    the ones the query names first ("direct growth"), then Direct and Growth; a plan is ambiguous when AMFI
    labels another code of the fund the same way."""
    q = query.strip()
    tokens = query_words(q)
    if ISIN.match(q.upper()):
        schemes, tokens = [s for s in schemes if q.upper() in (s.isin_growth, s.isin_reinvestment)], []
    funds: Dict[Tuple[str, str], List[Scheme]] = {}
    for s in schemes:
        funds.setdefault((s.amc, s.fund), []).append(s)
    ranked = []
    for (amc, fund), plans in funds.items():
        words = _index(fund)
        searchable = words | _index(amc)
        # a query may name the plan too: "HDFC Mid Cap Fund - Direct Plan - Growth Option"
        plan_words = {w for s in plans for w in _plan_words(s)}
        found, whole = _matched(tokens, words, searchable | plan_words)
        if found:
            asked = [t for t in tokens if not any(w.startswith(t) for w in searchable)]
            # a note in brackets, like "(erstwhile Bluechip Fund)", does not make the name longer
            ranked.append(((-whole, len(name_words(NOTE.sub("", fund))), fund), amc, fund, sorted(plans, key=lambda s: _plan_order(s, asked))))
    ranked.sort(key=lambda r: r[0])
    return [{"fund": fund, "amc": amc, "category": plans[0].category, "scheme_type": plans[0].scheme_type,
             "plans": [{"code": s.code, "plan": s.plan, "option": s.option, "nav": s.nav, "nav_date": s.nav_date,
                        "is_ambiguous": sum((o.plan, o.option) == (s.plan, s.option) for o in plans) > 1}
                       for s in plans]}
            for _, amc, fund, plans in ranked[:limit]]


def parse_history(text: str, code: int) -> pd.Series:
    """mfapi.in's {"meta": ..., "data": [{"date": "DD-MM-YYYY", "nav": "884.65"}, ...]} as NAV by date, oldest first."""
    try:
        body = json.loads(text)
        found, rows = body["meta"]["scheme_code"], body["data"]
        nav = pd.Series([float(r["nav"]) for r in rows], name="nav",
                        index=pd.to_datetime([r["date"] for r in rows], format="%d-%m-%Y"))
    except (ValueError, KeyError, TypeError) as exc:
        raise DataUnavailable(f"mfapi.in's history for scheme {code} could not be read: {exc!r}") from exc
    if found != code or nav.empty:
        raise DataUnavailable(f"mfapi.in has no NAV history for scheme {code}")
    if nav.index.duplicated().any():
        raise DataUnavailable(f"mfapi.in lists these dates twice in scheme {code}'s history: "
                              f"{sorted({str(d.date()) for d in nav.index[nav.index.duplicated()]})[:5]}")
    nav = nav.sort_index()
    unpriced = nav.index[~(nav > 0)]      # segregated portfolios are carried at 0
    if len(unpriced):
        raise DataUnavailable(f"mfapi.in's history for scheme {code} has {len(unpriced)} NAVs that are not above zero, "
                              f"first on {[str(d.date()) for d in unpriced[:3]]}, so no return can be measured from it")
    return nav


# NAVs are published to 4 decimals and mfapi.in writes 5, so anything past rounding is a different number
AGREES_WITHIN = 1e-4


def history_check(nav: pd.Series, scheme: Scheme) -> str:
    """How mfapi.in's history stands against AMFI's own NAV; DataUnavailable when they disagree on the same day,
    because an unofficial source that is wrong there may be wrong anywhere."""
    on_day = nav.get(pd.Timestamp(scheme.nav_date))
    if on_day is None:
        return (f"not checked against AMFI: mfapi.in's latest NAV is from {nav.index[-1].date()} and AMFI's "
                f"is from {scheme.nav_date}")
    if abs(on_day - scheme.nav) > AGREES_WITHIN * scheme.nav:
        raise DataUnavailable(f"mfapi.in gives scheme {scheme.code} a NAV of {on_day} on {scheme.nav_date} while "
                              f"AMFI gives {scheme.nav}, so its history is not shown")
    return f"matches AMFI's NAV on {scheme.nav_date}"


def _download(url: str) -> str:
    from backend.database.live.session import PoliteSession
    try:
        return PoliteSession(timeout=TIMEOUT_S, attempts=ATTEMPTS).get_text(url)
    except RuntimeError as exc:          # refused, gave up after retries, or no contact address set
        raise DataUnavailable(f"could not download {url}: {exc}") from exc


def _todays_copy(url: str, path: Path, check: Callable[[str], object]) -> Path:
    """`path`, downloaded again unless it was written today. A download `check` rejects is never kept, and a
    failed download is not tried again for RETRY_AFTER_S: its error is raised again, saying how long ago it was."""
    with _LOCKS.setdefault(path, threading.Lock()):      # per file, so a slow history download holds up no search
        if not path.exists() or date.fromtimestamp(path.stat().st_mtime) != date.today():
            failed_at, error = _FAILED.get(path, (float("-inf"), ""))
            age = time.monotonic() - failed_at
            if age < RETRY_AFTER_S:
                raise DataUnavailable(f"{error} (that was {age:.0f} s ago; it is tried again in {RETRY_AFTER_S - age:.0f} s)")
            try:
                text = _download(url)
                check(text)
            except DataUnavailable as exc:
                _FAILED[path] = (time.monotonic(), str(exc))
                raise
            _FAILED.pop(path, None)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            partial.write_text(text, encoding="utf-8")
            os.replace(partial, path)          # so a reader never sees a half-written file
    return path


def schemes() -> List[Scheme]:
    """The current schemes in today's copy of AMFI's NAV file, parsed once per copy."""
    path = _todays_copy(NAV_URL, NAV_FILE, parse)
    stamp = path.stat().st_mtime_ns
    if _READ.get(path, (None,))[0] != stamp:
        _READ[path] = (stamp, current(parse(path.read_text(encoding="utf-8"))))
    return _READ[path][1]


def history(code: int) -> pd.Series:
    """One scheme's NAV history from mfapi.in, downloaded at most once a day."""
    path = _todays_copy(HISTORY_URL.format(code=code), HISTORY_DIR / f"{code}.json",
                        lambda text: parse_history(text, code))
    return parse_history(path.read_text(encoding="utf-8"), code)
