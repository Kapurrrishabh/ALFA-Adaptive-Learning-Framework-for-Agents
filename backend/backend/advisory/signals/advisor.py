"""Live signal advisor: turns the validated cross-sectional signals into
concrete actions for the user's capital and holdings.

What is (and isn't) evidenced — see docs/research/ and docs/DESIGN.md §28:
- Momentum selection follows NSE's Nifty200 Momentum 30 score. The live ETF
  tracking it beat the Nifty 500 by +1.5 to +2.5 pts/yr (Sep 2022–Sep 2026, −10 pts in 2025); our
  same-universe backtest edge was +2.9 pts/yr. Absolute backtest returns are
  inflated by survivorship bias and are never shown as expectations.
- Low-volatility selection gives roughly market returns with less risk.
- The trend + volatility overlay reduced max drawdown (−35% → −22%) at the
  cost of return; it is offered as a risk setting, not a return enhancer.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.database.sources.panel import Panel
from backend.advisory.fusion.evidence import utcnow_iso
from backend.advisory.signals import factors as F
from backend.advisory.signals import strategies as S

N_DEFAULT = 20
DP_CHARGE = 15.34            # ₹ per scrip per sell day (CDSL/NSDL + broker), Zerodha 2026
MIN_POSITION = 20_000.0      # below this the DP charge alone costs > 0.08% per exit
STOP_PCT = 0.10              # Han–Zhou–Zhu stop inside momentum
LTCG_DAYS = 365
TAX_WINDOW_DAYS = 60         # PROVISIONAL: defer profitable exits this close to long-term status
REBALANCE_MONTHS = (6, 12)   # NSE semi-annual, executed in June and December


@dataclass
class Holding:
    symbol: str
    quantity: float
    avg_cost: float
    buy_date: Optional[str] = None


@dataclass
class Action:
    symbol: str
    action: str              # BUY | SELL | HOLD | TRIM | ADD
    shares: int
    price: float
    value: float
    rank: Optional[int]
    reasons: List[str]
    stop_level: Optional[float] = None
    tax_note: Optional[str] = None


@dataclass
class Plan:
    as_of: str
    generated_at: str
    strategy: str
    capital: float
    invest_fraction: float
    positions: int
    market: Dict[str, object]
    actions: List[Action]
    ranking: List[Dict[str, object]]
    cash_left: float
    next_rebalance: str
    evidence: List[str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def market_state(bench: pd.Series, asof: pd.Timestamp) -> Dict[str, object]:
    me = bench.loc[F.month_ends(bench.index)].loc[:asof]
    sma10 = float(me.iloc[-10:].mean())
    last_me = float(me.iloc[-1])
    trend_up = last_me > sma10
    r = np.log(bench / bench.shift(1)).loc[:asof]
    ewma_vol = float(np.sqrt((r ** 2).ewm(halflife=20).mean().iloc[-1] * F.YEAR))
    vol_expo = min(1.0, 0.15 / ewma_vol) if ewma_vol > 0 else 1.0
    b = bench.loc[:asof]
    r24 = float(b.iloc[-1] / b.iloc[-2 * F.YEAR] - 1) if len(b) > 2 * F.YEAR else None
    roll = (r.rolling(126).std() * np.sqrt(F.YEAR)).dropna()
    vol_q = float((roll < roll.iloc[-1]).mean()) if len(roll) else None
    crash_risk = bool(r24 is not None and r24 < 0 and vol_q is not None and vol_q >= 0.8)
    return {"benchmark_last": round(float(b.iloc[-1]), 2), "month_end_close": round(last_me, 2),
            "sma_10m": round(sma10, 2), "trend": "up" if trend_up else "down",
            "ewma_vol_pct": round(ewma_vol * 100, 1), "vol_target_exposure": round(vol_expo, 2),
            "return_24m_pct": None if r24 is None else round(r24 * 100, 1),
            "vol_percentile": None if vol_q is None else round(vol_q * 100),
            "momentum_crash_risk": crash_risk}


CRASH_EXPOSURE = 0.5        # 'balanced' exposure in momentum crash-risk regimes


def exposure_for(risk_mode: str, m: Dict[str, object]) -> float:
    """full: always invested. balanced: cut momentum in crash-risk regimes.
    defensive: trend filter × volatility target (lowest drawdown in the backtest)."""
    if risk_mode == "full":
        return 1.0
    if risk_mode == "balanced":
        return CRASH_EXPOSURE if m["momentum_crash_risk"] else 1.0
    if risk_mode == "defensive":
        return (1.0 if m["trend"] == "up" else 0.0) * float(m["vol_target_exposure"])
    raise ValueError(f"risk_mode must be full, balanced or defensive; got {risk_mode!r}")


def next_rebalance(today: date) -> str:
    for y in (today.year, today.year + 1):
        for mth in REBALANCE_MONTHS:
            d = date(y, mth, 1)
            if d > today:
                return f"first trading day of {d.strftime('%B %Y')} (signal from end-{date(y, mth - 1, 1).strftime('%b')} prices)"
    return "unknown"


def plan(panel: Panel, bench: pd.Series, capital: float, holdings: List[Holding],
         strategy: str = "momentum", risk_mode: str = "balanced", n: Optional[int] = None,
         universe: Optional[List[str]] = None, today: Optional[date] = None,
         insider_trades: Optional[pd.DataFrame] = None, *, measured: List[str]) -> Plan:
    """`measured` is the track record to show with the plan (performance.momentum_evidence)."""
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")
    if strategy not in ("momentum", "lowvol", "blend"):
        raise ValueError(f"strategy must be momentum, lowvol or blend; got {strategy!r}")
    today = today or date.today()
    close = panel.close
    asof = close.index[-1]
    uni = pd.Index([s for s in (universe or []) if s in close.columns]) if universe else \
        F.liquid_universe(close, panel.volume, asof)
    warnings: List[str] = []
    if universe:
        missing = sorted(set(universe) - set(close.columns))
        if missing:
            warnings.append(f"{len(missing)} universe stocks have no price data and were skipped: "
                            f"{', '.join(missing[:8])}")
    held_value = sum(h.quantity * float(close[h.symbol].iloc[-1]) for h in holdings
                     if h.symbol in close.columns)
    total = capital + held_value
    n = n or max(3, min(N_DEFAULT, int(total // MIN_POSITION)))
    if total / n < MIN_POSITION:
        warnings.append(f"₹{total:,.0f} across {n} stocks is under ₹{MIN_POSITION:,.0f} each; "
                        "DP charges and rounding will eat into returns")
    mom = F.momentum_score(close, asof, uni)
    low = F.lowvol_score(close, asof, uni)
    held_syms = [h.symbol for h in holdings if h.quantity > 0]
    m = market_state(bench, asof)
    invest = exposure_for(risk_mode, m) if strategy != "lowvol" or risk_mode == "defensive" else 1.0
    budget = total * invest
    slot = budget / n if n else 0.0
    prices = close.iloc[-1]

    def affordable(idx: pd.Index) -> pd.Index:
        # A share that costs more than its slot can't be bought; the next rank takes the slot.
        keep = [s for s in idx if s in held_syms or prices[s] <= slot]
        dropped = [s for s in idx[:2 * n] if s not in keep]
        if dropped:
            warnings.append(f"skipped {', '.join(dropped)}: one share costs more than a ₹{slot:,.0f} slot")
        return pd.Index(keep)

    if strategy == "momentum":
        ranked = mom
        picks = S.buffered_pick(affordable(ranked.index), held_syms, n, 2 * n)
        weights = S.equal_weight(picks)
    elif strategy == "lowvol":
        ranked = low
        picks = S.buffered_pick(affordable(ranked.index), held_syms, n, 2 * n)
        weights = S.inverse_vol_weight(picks, F.volatility(close, asof))
    else:
        half = max(1, n // 2)
        pm = S.buffered_pick(affordable(mom.index), held_syms, half, 2 * half)
        rest = affordable(low.index.difference(pd.Index(pm), sort=False))
        pl = S.buffered_pick(rest, held_syms, n - half, 2 * (n - half))
        picks = pm + pl
        weights = S.equal_weight(picks)
        ranked = mom
    if m["trend"] == "down":
        warnings.append(f"Market trend is DOWN: Nifty month-end {m['month_end_close']:,.0f} is below its "
                        f"10-month average {m['sma_10m']:,.0f}. 'defensive' mode would hold cash now; "
                        "historically this filter cut drawdowns but also cut returns.")
    rank_of = {s: i + 1 for i, s in enumerate(ranked.index)}
    by_sym = {h.symbol: h for h in holdings}
    actions: List[Action] = []
    for sym in picks:
        px = float(prices[sym])
        target_sh = int(math.floor(budget * weights[sym] / px))
        have = int(by_sym[sym].quantity) if sym in by_sym else 0
        reasons = _reasons(sym, mom, low, rank_of, strategy)
        stop = round(px * (1 - STOP_PCT), 2) if strategy != "lowvol" else None
        if have == 0:
            if target_sh == 0:
                warnings.append(f"{sym}: one share (₹{px:,.0f}) exceeds its ₹{budget * weights[sym]:,.0f} slot; skipped")
                continue
            actions.append(Action(sym, "BUY", target_sh, px, round(target_sh * px, 2), rank_of.get(sym),
                                  reasons, stop))
        else:
            delta = target_sh - have
            kind = "HOLD" if abs(delta) * px < 0.25 * budget * weights[sym] else ("ADD" if delta > 0 else "TRIM")
            if kind == "HOLD":
                delta = 0
            ref = by_sym[sym].avg_cost
            actions.append(Action(sym, kind, abs(delta), px, round(abs(delta) * px, 2), rank_of.get(sym),
                                  reasons + [f"still ranks within the keep band (top {2 * n})"],
                                  round(max(ref, px) * (1 - STOP_PCT), 2) if strategy != "lowvol" else None,
                                  _tax_note(by_sym[sym], px, today) if kind == "TRIM" else None))
    for h in holdings:
        if h.symbol in picks or h.quantity <= 0:
            continue
        px = float(prices[h.symbol]) if h.symbol in prices.index and not np.isnan(prices[h.symbol]) else float("nan")
        rk = rank_of.get(h.symbol)
        why = [f"ranked {rk} of {len(ranked)}, outside the keep band (top {2 * n})" if rk
               else "no longer in the ranked universe"]
        note = _tax_note(h, px, today)
        action = "HOLD" if note and note.startswith("DEFER") else "SELL"
        if action == "HOLD":
            why.append("sell signal deferred for tax (see note)")
        actions.append(Action(h.symbol, action, int(h.quantity) if action == "SELL" else 0, px,
                              round(h.quantity * px, 2) if action == "SELL" else 0.0, rk, why, None, note))
    if insider_trades is not None and len(insider_trades):
        _attach_insider_flags(actions, insider_trades, asof)
    ranking = [{"rank": i + 1, "symbol": s, "score": round(float(ranked.loc[s, "score"]), 3),
                **({"r6_pct": round(float(mom.loc[s, "r6"]) * 100, 1), "r12_pct": round(float(mom.loc[s, "r12"]) * 100, 1)}
                   if s in mom.index else {}),
                "vol_pct": round(float(ranked.loc[s, "sigma"]) * 100, 1)}
               for i, s in enumerate(ranked.index[:max(30, 2 * n)])]
    evidence = [
        "Momentum score = NSE Nifty200 Momentum 30 method (z of 12m and 6m return ÷ 1y volatility).",
        *measured,
        "Momentum crashes happen in sharp rebounds after bear markets; "
        f"'balanced' mode cuts exposure to {CRASH_EXPOSURE:.0%} in that regime.",
        f"{STOP_PCT:.0%} stop per stock follows Han–Zhou–Zhu (momentum crash protection).",
        "Expect to trail the market in some years. Stick to the schedule.",
    ]
    return Plan(as_of=str(asof.date()), generated_at=utcnow_iso(), strategy=strategy, capital=total,
                invest_fraction=round(invest, 2), positions=len(picks), market=m, actions=actions,
                ranking=ranking, cash_left=round(capital + _net_cash(actions), 2),
                next_rebalance=next_rebalance(today), evidence=evidence, warnings=warnings)


INSIDER_LOOKBACK_DAYS = 90
PROMOTER = {"Promoters", "Promoter Group", "Promoter"}


def _attach_insider_flags(actions: List[Action], trades: pd.DataFrame, asof: pd.Timestamp) -> None:
    """Informational only: promoter buying showed +0.7% 20-day excess return in liquid
    stocks (t 2.4, not significant after multiple-testing correction), so it never
    changes a selection; promoter selling showed no effect."""
    recent = trades[(trades["broadcast"] >= asof - pd.Timedelta(days=INSIDER_LOOKBACK_DAYS))
                    & trades["category"].isin(PROMOTER)]
    for a in actions:
        mine = recent[recent["symbol"] == a.symbol]
        if mine.empty:
            continue
        buys, sells = mine[mine["direction"] > 0], mine[mine["direction"] < 0]
        parts = []
        if len(buys):
            parts.append(f"promoters bought ₹{buys['value'].sum() / 1e7:,.2f} cr in {len(buys)} filing(s), "
                         f"latest {buys['broadcast'].max().date()}")
        if len(sells):
            parts.append(f"promoters sold ₹{sells['value'].sum() / 1e7:,.2f} cr in {len(sells)} filing(s)")
        a.reasons.append("insider (last 90 days, weak evidence): " + "; ".join(parts))


def _net_cash(actions: List[Action]) -> float:
    sign = {"BUY": -1, "ADD": -1, "SELL": 1, "TRIM": 1, "HOLD": 0}
    return sum(sign[a.action] * a.value for a in actions if not np.isnan(a.value))


def _reasons(sym: str, mom: pd.DataFrame, low: pd.DataFrame, rank_of: Dict[str, int], strategy: str) -> List[str]:
    out = []
    if sym in mom.index:
        r = mom.loc[sym]
        out.append(f"momentum rank {list(mom.index).index(sym) + 1}: 6m {r.r6 * 100:+.1f}%, 12m {r.r12 * 100:+.1f}%, "
                   f"volatility {r.sigma * 100:.0f}%")
    if strategy in ("lowvol", "blend") and sym in low.index:
        out.append(f"low-volatility rank {list(low.index).index(sym) + 1} (1y vol {low.loc[sym, 'sigma'] * 100:.0f}%)")
    return out


def _tax_note(h: Holding, px: float, today: date) -> Optional[str]:
    if not h.buy_date or np.isnan(px):
        return None
    held = (today - datetime.fromisoformat(h.buy_date).date()).days
    gain = (px - h.avg_cost) * h.quantity
    if held > LTCG_DAYS:
        if gain < 0:
            return f"long-term loss of ₹{-gain:,.0f} (held {held} days): offsets only long-term gains"
        return f"long-term gain of ₹{gain:,.0f} (held {held} days): 12.5% tax on gains above ₹1.25 lakh per year"
    to_lt = LTCG_DAYS + 1 - held
    if gain > 0 and to_lt <= TAX_WINDOW_DAYS:
        saved = gain * (0.20 - 0.125)
        return (f"DEFER: becomes long-term in {to_lt} days; waiting saves about ₹{saved:,.0f} tax "
                f"(20% → 12.5%) — exit early only if it falls below its stop")
    if gain < 0:
        return f"short-term loss of ₹{-gain:,.0f}: selling now books it to offset other gains"
    return f"short-term gain of ₹{gain:,.0f}: 20% tax applies"


def render(p: Plan) -> str:
    m = p.market
    lines = [f"StockIntel signal plan — {p.strategy} strategy, prices as of {p.as_of} (generated {p.generated_at})",
             f"Portfolio value (holdings + new money) ₹{p.capital:,.0f} | invest {p.invest_fraction:.0%} | {p.positions} positions | "
             f"next scheduled rebalance: {p.next_rebalance}",
             "",
             f"MARKET: Nifty {m['benchmark_last']:,.0f}, trend {m['trend'].upper()} (month-end {m['month_end_close']:,.0f} "
             f"vs 10-month avg {m['sma_10m']:,.0f}); volatility {m['ewma_vol_pct']}%; 24-month return "
             f"{m['return_24m_pct']}%; momentum crash risk: {'YES' if m['momentum_crash_risk'] else 'no'}",
             ""]
    order = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4}
    for a in sorted(p.actions, key=lambda a: (order[a.action], a.rank or 999)):
        head = f"{a.action:<4} {a.symbol:<12}"
        if a.action in ("BUY", "ADD", "SELL", "TRIM"):
            head += f" {a.shares:>6} sh @ ₹{a.price:>10,.2f} = ₹{a.value:>10,.0f}"
        else:
            head += f" {'':>6}    @ ₹{a.price:>10,.2f}"
        if a.stop_level:
            head += f" | stop ₹{a.stop_level:,.2f}"
        lines.append(head)
        lines += [f"       · {r}" for r in a.reasons]
        if a.tax_note:
            lines.append(f"       · tax: {a.tax_note}")
    lines.append(f"\nCash left after these orders: ₹{p.cash_left:,.0f}")
    if p.warnings:
        lines.append("\nWarnings:")
        lines += [f"  ! {w}" for w in p.warnings]
    lines.append("\nHow to use: place orders on the next trading day; check stops daily "
                 "(`stockintel stops`); re-run at the scheduled rebalance.")
    lines.append("Evidence and limits:")
    lines += [f"  - {e}" for e in p.evidence]
    return "\n".join(lines)
