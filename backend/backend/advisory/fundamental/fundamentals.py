"""Fundamental analysis engine. Every ratio is computed here from statement
line items (or read verbatim from the provider snapshot, labelled as such).
A metric that cannot be computed is listed under `unavailable` — it is never
estimated, defaulted, or filled from memory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, clamp, unavailable

DOMAIN = "fundamental"

FINANCIAL_SECTORS = {"Financial Services", "Financial", "Banks"}

# Provider multiples outside these ranges are data errors, not valuations. PROVISIONAL:
# wide enough for any listed large cap; found live when INFY's EV/EBITDA read 903.
PLAUSIBLE = {"forward_pe": (0, 500), "price_to_book": (0, 100),
             "ev_to_ebitda": (-100, 200), "ev_to_sales": (0, 100)}
STATEMENT_BASED = {"ev_to_ebitda", "ev_to_sales"}
TRAILING_PE = "info.trailingPE"      # yahoo's P/E over the last twelve months, not a financial year

# Yahoo line-item aliases; first match wins.
LINES = {
    "revenue": ("Total Revenue", "Operating Revenue", "Revenue"),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income", "EBIT"),
    "net_income": ("Net Income", "Net Income Common Stockholders",
                   "Net Income From Continuing Operation Net Minority Interest"),
    "eps": ("Diluted EPS", "Basic EPS"),
    "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
    "pretax_income": ("Pretax Income",),
    "tax": ("Tax Provision",),
    "total_debt": ("Total Debt",),
    "equity": ("Stockholders Equity", "Common Stock Equity",
               "Total Equity Gross Minority Interest"),
    "total_assets": ("Total Assets",),
    "current_assets": ("Current Assets",),
    "current_liabilities": ("Current Liabilities",),
    "cash": ("Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments"),
    "invested_capital": ("Invested Capital",),
    "fcf": ("Free Cash Flow",),
    "ocf": ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
    "capex": ("Capital Expenditure",),
}


def line(statements: Dict[str, pd.DataFrame], kind: str, item: str) -> pd.Series:
    """A statement line as a Series indexed by period end, newest first."""
    df = statements.get(kind)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in LINES[item]:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if not s.empty:
                s.index = pd.to_datetime(s.index)
                return s.sort_index(ascending=False)
    return pd.Series(dtype=float)


def _period(s: pd.Series, i: int = 0) -> str:
    ts = s.index[i]
    return f"FY ending {ts.date()}"


def _one_year_apart(a: pd.Timestamp, b: pd.Timestamp) -> bool:
    # Providers drop empty periods, so the "previous" column can be two years back.
    return abs((a - b).days - 365) <= 45


def _growth(s: pd.Series) -> Optional[Tuple[float, str]]:
    if len(s) < 2 or s.iloc[1] == 0 or not _one_year_apart(s.index[0], s.index[1]):
        return None
    # Growth off a negative base is sign-ambiguous; refuse rather than mislead.
    if s.iloc[1] < 0:
        return None
    return float(s.iloc[0] / s.iloc[1] - 1.0), f"{_period(s, 0)} vs {_period(s, 1)}"


def _band(value: float, bands: Sequence[Tuple[float, float]], below: float) -> float:
    """Score = first band whose threshold the value meets (bands high->low)."""
    for threshold, score in bands:
        if value >= threshold:
            return score
    return below


class _Collector:
    def __init__(self, symbol: str, source: str, currency: Optional[str] = None):
        self.symbol, self.source, self.currency = symbol, source, currency
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.missing: List[str] = []
        self.evidence: List[Evidence] = []
        self.sub: Dict[str, List[float]] = {"growth": [], "profitability": [],
                                            "health": [], "valuation": []}

    def add(self, key: str, value: Optional[float], period: str, origin: str,
            group: str, score: Optional[float], claim: str, unit: str = "%"):
        if value is None or not np.isfinite(value):
            self.missing.append(key)
            return
        shown = value * 100 if unit == "%" else value
        self.metrics[key] = {"value": round(float(shown), 4), "unit": unit, "period": period,
                             "source": f"{self.source}:{origin}"}
        if unit == "abs":
            self.metrics[key]["currency"] = self.currency or "Data unavailable"
        if score is not None:
            self.sub[group].append(score)
            self.evidence.append(Evidence(
                domain=DOMAIN, claim=claim.format(v=shown), direction=int(np.sign(score)),
                strength=min(1.0, abs(score)), value=round(float(shown), 4),
                provenance=Provenance(source=f"{self.source}:{origin}", as_of=period,
                                      period=period)))

    def skip(self, key: str, reason: str):
        self.metrics[key] = {"value": None, "note": reason}


def analyze(symbol: str, info: Dict[str, Any], statements: Dict[str, pd.DataFrame],
            source: str = "provider") -> DomainResult:
    if not statements and not info:
        return unavailable(DOMAIN, f"no fundamentals available for {symbol}")
    price_ccy = info.get("currency")
    reported = info.get("financialCurrency") or price_ccy       # the currency of yahoo's own totals, EBITDA among them
    stmt_ccy = statement_currency(info, statements)
    # Some issuers (e.g. Infosys) report statements in USD while trading in INR;
    # price-based ratios would mix currencies, so they come from the provider.
    mixed_ccy = bool(price_ccy and stmt_ccy and price_ccy != stmt_ccy)
    c = _Collector(symbol, source, stmt_ccy)
    sector = info.get("sector") or "Data unavailable"
    is_financial = sector in FINANCIAL_SECTORS

    rev = line(statements, "income", "revenue")
    ni = line(statements, "income", "net_income")
    eps = line(statements, "income", "eps")
    gp = line(statements, "income", "gross_profit")
    op = line(statements, "income", "operating_income")
    interest = line(statements, "income", "interest_expense")
    pretax = line(statements, "income", "pretax_income")
    tax = line(statements, "income", "tax")
    debt = line(statements, "balance", "total_debt")
    equity = line(statements, "balance", "equity")
    assets = line(statements, "balance", "total_assets")
    ca = line(statements, "balance", "current_assets")
    cl = line(statements, "balance", "current_liabilities")
    cash = line(statements, "balance", "cash")
    ic = line(statements, "balance", "invested_capital")
    fcf = line(statements, "cashflow", "fcf")
    if fcf.empty:
        ocf, capex = line(statements, "cashflow", "ocf"), line(statements, "cashflow", "capex")
        if not ocf.empty and not capex.empty:
            fcf = (ocf - capex.abs()).dropna().sort_index(ascending=False)

    # Growth
    for key, s, label in (("revenue_growth", rev, "Revenue"), ("net_income_growth", ni, "Net income"),
                          ("eps_growth", eps, "EPS"), ("fcf_growth", fcf, "Free cash flow")):
        g = _growth(s)
        if g is None:
            c.add(key, None, "", "", "growth", None, "")
            continue
        value, period = g
        score = _band(value, [(0.20, 1.0), (0.10, 0.6), (0.03, 0.2), (0.0, 0.0), (-0.10, -0.5)], -1.0)
        c.add(key, value, period, "income_stmt" if key != "fcf_growth" else "cashflow",
              "growth", score if key != "fcf_growth" else score * 0.5,
              f"{label} changed {{v:+.1f}}% year on year.")
    if len(rev) >= 4 and rev.iloc[3] > 0 and rev.iloc[0] > 0:
        years = (rev.index[0] - rev.index[3]).days / 365.25
        cagr = (rev.iloc[0] / rev.iloc[3]) ** (1 / years) - 1
        c.add("revenue_cagr", cagr, f"{_period(rev, 3)} to {_period(rev, 0)} ({years:.1f} years)",
              "income_stmt", "growth", None, "")

    q = line(statements, "income_quarterly", "revenue")
    qni = line(statements, "income_quarterly", "net_income")
    if len(q) >= 5 and q.iloc[4] > 0 and _one_year_apart(q.index[0], q.index[4]):
        c.add("latest_quarter_revenue_yoy", q.iloc[0] / q.iloc[4] - 1,
              f"quarter ending {q.index[0].date()}", "quarterly_income_stmt", "growth", None, "")
    if len(qni) >= 5 and qni.iloc[4] > 0 and _one_year_apart(qni.index[0], qni.index[4]):
        c.add("latest_quarter_net_income_yoy", qni.iloc[0] / qni.iloc[4] - 1,
              f"quarter ending {qni.index[0].date()}", "quarterly_income_stmt", "growth", None, "")

    # Profitability
    def ratio(num: pd.Series, den: pd.Series, i: int = 0) -> Tuple[Optional[float], str]:
        """num/den in the i-th newest year the two share, and that year, which is the one the figure is dated by."""
        if len(num) <= i or len(den) <= i:
            return None, ""
        both = pd.concat([num, den], axis=1, join="inner").sort_index(ascending=False)
        if len(both) <= i or both.iloc[i, 1] == 0:
            return None, ""
        return float(both.iloc[i, 0] / both.iloc[i, 1]), _period(both, i)

    if not rev.empty:
        if not is_financial:
            gm, when = ratio(gp, rev)
            c.add("gross_margin", gm, when, "income_stmt", "profitability", None, "")
        om, when = ratio(op, rev)
        c.add("operating_margin", om, when, "income_stmt", "profitability",
              None if om is None else _band(om, [(0.20, 0.6), (0.10, 0.3), (0.0, -0.1)], -0.6),
              "Operating margin is {v:.1f}%.")
        nm, when = ratio(ni, rev)
        pair = pd.concat([ni, rev], axis=1, join="inner").sort_index(ascending=False)
        nm_prev, before = (ratio(ni, rev, 1) if len(pair) >= 2 and _one_year_apart(pair.index[0], pair.index[1])
                           else (None, ""))
        c.add("net_margin", nm, when, "income_stmt", "profitability",
              None if nm is None else _band(nm, [(0.15, 0.6), (0.08, 0.3), (0.0, -0.1)], -0.8),
              "Net margin is {v:.1f}%.")
        if nm is not None and nm_prev is not None:
            delta = nm - nm_prev
            c.add("net_margin_change", delta, f"{when} vs {before}",
                  "income_stmt", "profitability", clamp(delta * 20, -0.6, 0.6),
                  "Net margin moved {v:+.1f} percentage points year on year.")
    else:
        c.missing.extend(["gross_margin", "operating_margin", "net_margin"])

    roe, when = ratio(ni, equity)
    c.add("roe", roe, when, "income_stmt+balance_sheet",
          "profitability",
          None if roe is None else _band(roe, [(0.18, 1.0), (0.12, 0.5), (0.08, 0.0), (0.0, -0.4)], -1.0),
          "Return on equity is {v:.1f}%.")
    roa, when = ratio(ni, assets)
    c.add("roa", roa, when, "income_stmt+balance_sheet",
          "profitability", None, "")
    if not op.empty:
        # As interest cover: the operating margin's profit line, not yahoo's "EBIT" (176 of 186 snapshot shares with both
        # differ), in that line's latest year with that year's tax rate and capital: NESTLEIND's FY 2025 ROIC read as 2026's.
        latest = op.iloc[:1]
        rate, _ = ratio(tax[tax.index.isin(latest.index)], pretax)
        rate = min(max(rate, 0.0), 0.5) if rate is not None else None
        capital = ic
        if capital.empty and not debt.empty and not equity.empty:
            capital = (debt + equity - (cash if not cash.empty else 0)).dropna()
        r, when = ratio(latest, capital)
        roic = None if r is None or rate is None else r * (1 - rate)
        if not is_financial:
            c.add("roic", roic, when, "income_stmt+balance_sheet", "profitability",
                  None if roic is None else _band(roic, [(0.15, 0.6), (0.10, 0.3), (0.06, 0.0)], -0.5),
                  "Return on invested capital is {v:.1f}%.")

    # Financial health (leverage ratios are not comparable for lenders).
    if is_financial:
        for key in ("debt_to_equity", "interest_coverage", "current_ratio"):
            c.skip(key, "not meaningful for financial-sector balance sheets")
    else:
        if not equity.empty and equity.iloc[0] <= 0:
            c.skip("debt_to_equity", "not meaningful: shareholders' equity is negative")
            c.add("shareholders_equity", float(equity.iloc[0]), _period(equity), "balance_sheet",
                  "health", -1.0, "Shareholders' equity is negative ({v:,.0f}).", unit="abs")
        else:
            de, when = ratio(debt, equity)
            c.add("debt_to_equity", de, when,
                  "balance_sheet", "health",
                  None if de is None else -_band(de, [(2.0, 0.8), (1.0, 0.3), (0.5, 0.0)], -0.5),
                  "Debt/equity is {v:.2f}.", unit="x")
        # From the operating margin's profit line, in its latest year: yahoo's "EBIT" holds one-off charges (RHIM read -7.7
        # beside a +6.8% margin), and an older year's interest gave SHOP 2022's -597 under FY 2025.
        if not op.empty and op.index[0] in interest.index:
            icov, when = ratio(op, interest.abs())
            c.add("interest_coverage", icov, when, "income_stmt", "health",
                  None if icov is None else _band(icov, [(8, 0.6), (4, 0.2), (2, -0.3)], -0.9),
                  "EBIT covers interest {v:.1f} times.", unit="x")
        else:
            c.missing.append("interest_coverage")
        cr, when = ratio(ca, cl)
        c.add("current_ratio", cr, when, "balance_sheet", "health",
              None if cr is None else _band(cr, [(1.5, 0.3), (1.0, 0.0)], -0.4),
              "Current ratio is {v:.2f}.", unit="x")
    if not cash.empty:
        c.add("cash", float(cash.iloc[0]), _period(cash), "balance_sheet", "health", None, "",
              unit="abs")
    if not fcf.empty:
        c.add("free_cash_flow", float(fcf.iloc[0]), _period(fcf), "cashflow", "health",
              0.3 if fcf.iloc[0] > 0 else -0.4,
              "Latest free cash flow is {v:,.0f} " + (stmt_ccy or "(currency unknown)") + ".",
              unit="abs")

    # Valuation: snapshot fields are provider-reported. P/E is yahoo's over the last twelve months, the one the stock header
    # and quote pages show; price over the last FY's EPS, as 189 of 200 snapshot shares had it, put NYKAA at 515.5 to 322.2.
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    snap = f"snapshot retrieved {pd.Timestamp.now(tz='UTC').date()}"
    pe = info.get("trailingPE")
    if isinstance(pe, (int, float)) and pe > 0:
        c.add("pe", float(pe), snap, TRAILING_PE, "valuation",
              _band(-pe, [(-15, 0.5), (-30, 0.0), (-50, -0.3)], -0.6),
              "P/E is {v:.1f} (absolute bands; not sector-adjusted).", unit="x")
    elif not eps.empty and eps.iloc[0] <= 0:
        c.skip("pe", "negative earnings — P/E not meaningful")
        c.add("eps_latest", float(eps.iloc[0]), _period(eps), "income_stmt", "valuation", -0.5,
              "Latest annual EPS is {v:.2f} (loss-making).", unit="abs")
    else:
        c.missing.append("pe")
    for key, field_ in (("forward_pe", "forwardPE"), ("price_to_book", "priceToBook"),
                        ("ev_to_ebitda", "enterpriseToEbitda"), ("ev_to_sales", "enterpriseToRevenue")):
        v = info.get(field_)
        lo, hi = PLAUSIBLE[key]
        if isinstance(v, (int, float)) and price_ccy and reported != price_ccy and key in STATEMENT_BASED:
            c.skip(key, f"provider value mixes {reported} figures with {price_ccy} market value; not used")
        elif isinstance(v, (int, float)) and not lo < v < hi:
            c.skip(key, f"implausible provider value {v:,.1f}; not used")
        else:
            c.add(key, float(v) if isinstance(v, (int, float)) else None, snap, f"info.{field_}",
                  "valuation", None, "", unit="x")
    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    c.add("peg", float(peg) if isinstance(peg, (int, float)) else None, snap, "info.pegRatio",
          "valuation",
          None if not isinstance(peg, (int, float)) or peg <= 0 else
          _band(-float(peg), [(-1.0, 0.4), (-2.5, 0.0)], -0.4),
          "PEG ratio is {v:.2f}.", unit="x")
    mcap = info.get("marketCap")
    if mixed_ccy:
        c.skip("fcf_yield", f"statements in {stmt_ccy}, price in {price_ccy}; not computed")
    elif mcap and not fcf.empty:
        fy = float(fcf.iloc[0]) / float(mcap)
        c.add("fcf_yield", fy, snap, "cashflow/info.marketCap", "valuation",
              _band(fy, [(0.05, 0.4), (0.02, 0.1), (0.0, -0.1)], -0.3),
              "Free cash flow yield is {v:.1f}%.")
    else:
        c.missing.append("fcf_yield")
    rate = info.get("dividendRate")
    if rate and price:
        c.add("dividend_yield", float(rate) / float(price), snap, "info.dividendRate/price",
              "valuation", None, "")
    elif isinstance(info.get("trailingAnnualDividendYield"), (int, float)):
        c.add("dividend_yield", float(info["trailingAnnualDividendYield"]), snap,
              "info.trailingAnnualDividendYield", "valuation", None, "")

    weights = {"growth": 0.3, "profitability": 0.3, "health": 0.2, "valuation": 0.2}
    sub_scores = {k: float(np.mean(v)) for k, v in c.sub.items() if v}
    if not sub_scores:
        return unavailable(DOMAIN, f"statements for {symbol} lacked the line items needed "
                                   f"for any scored metric ({len(set(c.missing))} metrics missing)")
    wsum = sum(weights[k] for k in sub_scores)
    score = clamp(sum(weights[k] * v for k, v in sub_scores.items()) / wsum)
    computed = [k for k, m in c.metrics.items() if m.get("value") is not None]
    coverage = len(computed) / max(1, len(computed) + len(set(c.missing)))
    trend = _trend(c.metrics)
    state = ("strong" if score > 0.4 else "sound" if score > 0.1 else
             "mixed" if score > -0.2 else "weak")
    details = {"sector": sector, "industry": info.get("industry") or "Data unavailable",
               "metrics": c.metrics, "unavailable": sorted(set(c.missing)),
               "sub_scores": {k: round(v, 3) for k, v in sub_scores.items()},
               "coverage": round(coverage, 2), "trend": trend,
               "statement_currency": stmt_ccy or "Data unavailable", "price_currency": price_ccy or "Data unavailable",
               "profile": company_profile(info)}
    latest = rev.index[0].date() if not rev.empty else None
    return DomainResult(domain=DOMAIN, state=state, score=score, evidence=c.evidence,
                        details=details, as_of=str(latest) if latest else None,
                        confidence=round(0.4 + 0.6 * coverage, 2))


def statement_currency(info: Dict[str, Any], statements: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Yahoo's `financialCurrency`, unless the last four quarters' EPS add up, to within 1%, to its trailing EPS, which is
    in the price currency: HCLTECH.NS's own totals are in USD and its statements in INR. 90% of 295 one-currency shares
    agree to 1%, and the nearest pair in use, ASML's euro statements and dollar price, is 16% apart."""
    price_ccy, labelled = info.get("currency"), info.get("financialCurrency") or info.get("currency")
    quarters, trailing = line(statements, "income_quarterly", "eps"), info.get("trailingEps")
    if labelled != price_ccy and len(quarters) >= 4 and isinstance(trailing, (int, float)) and trailing \
            and abs(quarters.iloc[:4].sum() / trailing - 1) <= 0.01:
        return price_ccy
    return labelled


def _trend(metrics: Dict[str, Dict[str, Any]]) -> str:
    signals = []
    for key in ("revenue_growth", "eps_growth", "net_margin_change"):
        v = metrics.get(key, {}).get("value")
        if v is not None:
            signals.append(np.sign(v))
    if not signals:
        return "Data unavailable"
    s = float(np.mean(signals))
    return "improving" if s > 0.3 else "deteriorating" if s < -0.3 else "stable/mixed"


# The provider fields that move with the quote: as the quote (1), or against it (-1)
QUOTED = {"marketCap": 1, "trailingPE": 1, "forwardPE": 1, "priceToBook": 1, "trailingPegRatio": 1, "pegRatio": 1,
          "trailingAnnualDividendYield": -1}
EV_MULTIPLES = ("enterpriseToEbitda", "enterpriseToRevenue")


def priced_at(info: Dict[str, Any], close: float) -> Dict[str, Any]:
    """`info` moved from its live quote to `close`, with every price-based field `analyze` reads.

    With no quote to move them from they are dropped, not given a price they were not taken at. EV is market value plus
    debt less cash and the like, so it moves by its market value alone, and without both its multiples are dropped.
    """
    live = info.get("currentPrice") or info.get("regularMarketPrice")
    moved = {k: v for k, v in info.items() if k not in QUOTED and k not in EV_MULTIPLES and k != "enterpriseValue"}
    moved.update(currentPrice=close, regularMarketPrice=close)
    if not live:
        return moved
    for key, power in QUOTED.items():
        if isinstance(info.get(key), (int, float)):
            moved[key] = info[key] * (close / live) ** power
    ev, cap = info.get("enterpriseValue"), info.get("marketCap")
    if isinstance(ev, (int, float)) and ev and isinstance(cap, (int, float)):
        moved["enterpriseValue"] = ev + cap * (close / live - 1)
        for key in EV_MULTIPLES:
            if isinstance(info.get(key), (int, float)):
                moved[key] = info[key] * moved["enterpriseValue"] / ev
    return moved


def company_profile(info: Dict[str, Any]) -> Dict[str, Any]:
    """Verbatim provider profile fields; absent fields say so explicitly."""
    fields = {"name": "longName", "sector": "sector", "industry": "industry",
              "country": "country", "employees": "fullTimeEmployees", "website": "website",
              "market_cap": "marketCap", "currency": "currency", "exchange": "exchange",
              "summary": "longBusinessSummary",
              "insider_holding_pct": "heldPercentInsiders",
              "institutional_holding_pct": "heldPercentInstitutions"}
    return {k: info.get(v, "Data unavailable") for k, v in fields.items()}
