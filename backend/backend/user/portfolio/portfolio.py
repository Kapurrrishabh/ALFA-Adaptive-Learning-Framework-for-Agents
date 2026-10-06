"""Portfolio intelligence. Every number here is arithmetic on transactions
and prices — the generative layer only ever reads these results.

Positions are rebuilt from transactions with FIFO lot matching, so realized
P/L is exact and a sell larger than the holding is rejected, not netted.
Historical risk uses today's weights held constant over the lookback
(a standard "current portfolio backcast"); that assumption is reported.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.advisory.statistics.quant import beta_alpha, max_drawdown, var_cvar
from backend.config import ANNUALIZATION_DAYS as A, RISK_PROFILES, canonical_symbol


@dataclass
class Transaction:
    symbol: str
    side: str            # BUY | SELL
    quantity: float
    price: float
    trade_date: str
    fees: float = 0.0
    sector: Optional[str] = None

    def __post_init__(self):
        self.symbol = canonical_symbol(self.symbol)
        self.side = self.side.upper()
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"transaction side must be BUY or SELL, got {self.side!r}")
        if self.quantity <= 0 or self.price <= 0:
            raise ValueError(f"quantity and price must be positive: {self}")


@dataclass
class Position:
    symbol: str
    quantity: float
    cost_basis: float
    realized_pl: float
    sector: Optional[str]

    @property
    def avg_cost(self) -> float:
        return self.cost_basis / self.quantity if self.quantity else 0.0


def build_positions(transactions: Sequence[Transaction]) -> Dict[str, Position]:
    lots: Dict[str, Deque[List[float]]] = {}
    realized: Dict[str, float] = {}
    sectors: Dict[str, Optional[str]] = {}
    for tx in sorted(transactions, key=lambda t: t.trade_date):
        q = lots.setdefault(tx.symbol, deque())
        realized.setdefault(tx.symbol, 0.0)
        if tx.sector:
            sectors[tx.symbol] = tx.sector
        if tx.side == "BUY":
            q.append([tx.quantity, tx.price + tx.fees / tx.quantity])
            continue
        held = sum(l[0] for l in q)
        if tx.quantity > held + 1e-9:
            raise ValueError(f"SELL of {tx.quantity} {tx.symbol} on {tx.trade_date} exceeds the "
                             f"{held} shares held; fix the transaction history")
        remaining = tx.quantity
        proceeds_per_share = tx.price - tx.fees / tx.quantity
        while remaining > 1e-12:
            lot = q[0]
            take = min(lot[0], remaining)
            realized[tx.symbol] += take * (proceeds_per_share - lot[1])
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-12:
                q.popleft()
    out = {}
    for sym, q in lots.items():
        qty = sum(l[0] for l in q)
        out[sym] = Position(symbol=sym, quantity=qty, cost_basis=sum(l[0] * l[1] for l in q),
                            realized_pl=realized[sym], sector=sectors.get(sym))
    return out


def _returns_matrix(prices: Dict[str, pd.Series], lookback: int = A) -> pd.DataFrame:
    frame = pd.concat({s: p for s, p in prices.items()}, axis=1).sort_index().ffill()
    return frame.pct_change().dropna(how="any").iloc[-lookback:]


def risk_decomposition(weights: pd.Series, rets: pd.DataFrame) -> Dict[str, object]:
    w = weights.reindex(rets.columns).fillna(0.0).to_numpy()
    cov = rets.cov().to_numpy() * A
    var_p = float(w @ cov @ w)
    vol_p = math.sqrt(var_p) if var_p > 0 else 0.0
    marginal = cov @ w
    rc = w * marginal / vol_p if vol_p > 0 else np.zeros_like(w)
    vols = np.sqrt(np.diag(cov))
    corr = rets.corr()
    n = len(w)
    pairwise = corr.to_numpy()[np.triu_indices(n, 1)] if n > 1 else np.array([])
    return {"portfolio_vol_pct": round(vol_p * 100, 2),
            "risk_contribution_pct": {s: round(float(c / vol_p * 100), 2) if vol_p else 0.0
                                      for s, c in zip(rets.columns, rc)},
            "holding_vol_pct": {s: round(float(v * 100), 2) for s, v in zip(rets.columns, vols)},
            "diversification_ratio": round(float((w * vols).sum() / vol_p), 3) if vol_p else None,
            "avg_pairwise_correlation": round(float(pairwise.mean()), 3) if len(pairwise) else None,
            "correlation": corr.round(3).to_dict()}


@dataclass
class PortfolioReport:
    as_of: str
    total_value: float
    invested_value: float
    cash: float
    cash_utilization_pct: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_pct: float
    realized_pl: float
    holdings: List[Dict[str, object]]
    sector_exposure_pct: Dict[str, float]
    concentration: Dict[str, object]
    risk: Dict[str, object]
    week_change: Dict[str, object]
    flags: List[str]
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def analyze(positions: Dict[str, Position], cash: float, prices: Dict[str, pd.Series],
            risk_profile: str = "moderate", bench: Optional[pd.Series] = None,
            rf: float = 0.0, sectors: Optional[Dict[str, str]] = None) -> PortfolioReport:
    if risk_profile not in RISK_PROFILES:
        raise ValueError(f"risk_profile must be one of {list(RISK_PROFILES)}, got {risk_profile!r}")
    profile = RISK_PROFILES[risk_profile]
    open_pos = {s: p for s, p in positions.items() if p.quantity > 1e-12}
    missing = [s for s in open_pos if s not in prices or prices[s].dropna().empty]
    if missing:
        raise ValueError(f"no price history for holdings {missing}; cannot value the portfolio")
    sectors = {**{s: p.sector for s, p in open_pos.items() if p.sector}, **(sectors or {})}
    last = {s: float(prices[s].dropna().iloc[-1]) for s in open_pos}
    as_of = str(max(prices[s].dropna().index[-1] for s in open_pos).date()) if open_pos else "n/a"
    values = {s: open_pos[s].quantity * last[s] for s in open_pos}
    invested = sum(values.values())
    total = invested + cash
    cost = sum(p.cost_basis for p in open_pos.values())
    realized = sum(p.realized_pl for p in positions.values())
    holdings = []
    for s, p in sorted(open_pos.items(), key=lambda kv: -values[kv[0]]):
        holdings.append({"symbol": s, "quantity": p.quantity, "avg_cost": round(p.avg_cost, 2),
                         "price": round(last[s], 2), "value": round(values[s], 2),
                         "weight_pct": round(values[s] / total * 100, 2) if total else 0.0,
                         "unrealized_pl": round(values[s] - p.cost_basis, 2),
                         "unrealized_pl_pct": round((values[s] / p.cost_basis - 1) * 100, 2)
                         if p.cost_basis else None,
                         "realized_pl": round(p.realized_pl, 2),
                         "sector": sectors.get(s, "Data unavailable")})
    sector_exp: Dict[str, float] = {}
    for h in holdings:
        sector_exp[h["sector"]] = round(sector_exp.get(h["sector"], 0.0) + h["weight_pct"], 2)
    w_inv = pd.Series({s: v / invested for s, v in values.items()}) if invested else pd.Series(dtype=float)
    hhi = float((w_inv ** 2).sum()) if invested else 0.0
    concentration = {"hhi": round(hhi, 4), "effective_holdings": round(1 / hhi, 2) if hhi else 0,
                     "top_holding_pct_of_invested": round(float(w_inv.max()) * 100, 2) if invested else 0}

    risk: Dict[str, object] = {}
    assumptions = ["Historical risk uses current weights held constant over the last year."]
    if len(open_pos) >= 1 and invested > 0:
        rets = _returns_matrix({s: prices[s] for s in open_pos})
        if len(rets) >= 60:
            risk = risk_decomposition(w_inv, rets)
            port_r = (rets * w_inv.reindex(rets.columns)).sum(axis=1)
            curve = (1 + port_r).cumprod()
            risk["max_drawdown_pct"] = max_drawdown(curve)["max_drawdown_pct"]
            v, cv = var_cvar(port_r)
            risk["var_95_1d_pct"], risk["cvar_95_1d_pct"] = round(v * 100, 2), round(cv * 100, 2)
            risk["var_95_1d_value"] = round(-v * invested, 2)
            if bench is not None:
                ba = beta_alpha(port_r, bench.pct_change().dropna(), rf)
                risk["beta"] = ba.get("beta", "Data unavailable")
            risk["observations"] = len(rets)
        else:
            risk["note"] = f"only {len(rets)} overlapping return days; risk metrics need 60"

    week: Dict[str, object] = {"by_holding": {}}
    total_change = 0.0
    for s in open_pos:
        p = prices[s].dropna()
        if len(p) > 5:
            chg = open_pos[s].quantity * (p.iloc[-1] - p.iloc[-6])
            total_change += chg
            week["by_holding"][s] = {"change_value": round(float(chg), 2),
                                     "change_pct": round(float(p.iloc[-1] / p.iloc[-6] - 1) * 100, 2)}
    week["total_change_value"] = round(total_change, 2)
    week["total_change_pct"] = round(total_change / (invested - total_change) * 100, 2) \
        if invested - total_change else None
    if week["by_holding"]:
        ranked = sorted(week["by_holding"].items(), key=lambda kv: kv[1]["change_value"])
        week["biggest_detractor"], week["biggest_contributor"] = ranked[0][0], ranked[-1][0]

    flags = []
    for h in holdings:
        if h["weight_pct"] / 100 > profile.max_position_weight:
            flags.append(f"{h['symbol']} is {h['weight_pct']:.1f}% of the portfolio, above the "
                         f"{profile.max_position_weight:.0%} {profile.name} position limit.")
    for sec, pct in sector_exp.items():
        if pct / 100 > profile.max_sector_weight and sec != "Data unavailable":
            flags.append(f"{sec} exposure is {pct:.1f}%, above the {profile.max_sector_weight:.0%} "
                         f"{profile.name} sector limit.")
    for s, v in (risk.get("holding_vol_pct") or {}).items():
        if v / 100 > profile.max_holding_vol:
            flags.append(f"{s} volatility {v:.0f}% exceeds the {profile.max_holding_vol:.0%} "
                         f"{profile.name} ceiling.")
    if "risk_contribution_pct" in risk:
        top, pct = max(risk["risk_contribution_pct"].items(), key=lambda kv: kv[1])
        weight = next(h["weight_pct"] for h in holdings if h["symbol"] == top)
        if pct > 1.5 * weight * total / invested:
            flags.append(f"{top} contributes {pct:.0f}% of portfolio risk from "
                         f"{weight * total / invested:.0f}% of invested capital.")
    return PortfolioReport(
        as_of=as_of, total_value=round(total, 2), invested_value=round(invested, 2),
        cash=round(cash, 2), cash_utilization_pct=round(invested / total * 100, 2) if total else 0.0,
        cost_basis=round(cost, 2), unrealized_pl=round(invested - cost, 2),
        unrealized_pl_pct=round((invested / cost - 1) * 100, 2) if cost else 0.0,
        realized_pl=round(realized, 2), holdings=holdings, sector_exposure_pct=sector_exp,
        concentration=concentration, risk=risk, week_change=week, flags=flags,
        assumptions=assumptions)


def what_if(positions: Dict[str, Position], cash: float, prices: Dict[str, pd.Series],
            scenarios: Dict[str, Dict[str, float]], risk_profile: str = "moderate",
            sectors: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Compare risk for alternative ways to deploy new money.

    scenarios: {"name": {symbol: amount_to_add}}. Purchases are valued at the
    last price; fractional shares are rounded down, leftover stays as cash.
    """
    base = analyze(positions, cash, prices, risk_profile, sectors=sectors)
    out = {"current": _risk_summary(base), "scenarios": {}}
    for name, adds in scenarios.items():
        pos = {s: Position(p.symbol, p.quantity, p.cost_basis, p.realized_pl, p.sector)
               for s, p in positions.items()}
        spent = 0.0
        bought = {}
        for sym, amount in adds.items():
            sym = canonical_symbol(sym)
            if sym not in prices:
                raise ValueError(f"no price history for scenario symbol {sym}")
            px = float(prices[sym].dropna().iloc[-1])
            qty = math.floor(amount / px)
            if qty <= 0:
                raise ValueError(f"{amount} buys no whole shares of {sym} at {px:.2f}")
            spent += qty * px
            bought[sym] = qty
            if sym in pos:
                pos[sym].quantity += qty
                pos[sym].cost_basis += qty * px
            else:
                pos[sym] = Position(sym, qty, qty * px, 0.0, (sectors or {}).get(sym))
        rep = analyze(pos, cash + sum(adds.values()) - spent, prices, risk_profile, sectors=sectors)
        out["scenarios"][name] = {"shares_bought": bought, "spent": round(spent, 2),
                                  **_risk_summary(rep)}
    return out


def _risk_summary(rep: PortfolioReport) -> Dict[str, object]:
    return {"total_value": rep.total_value, "portfolio_vol_pct": rep.risk.get("portfolio_vol_pct"),
            "max_drawdown_pct": rep.risk.get("max_drawdown_pct"),
            "cvar_95_1d_pct": rep.risk.get("cvar_95_1d_pct"),
            "effective_holdings": rep.concentration["effective_holdings"],
            "sector_exposure_pct": rep.sector_exposure_pct, "flags": rep.flags}
