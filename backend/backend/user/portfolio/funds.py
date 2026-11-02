"""Mutual fund holdings, worked out from fund transactions as stock holdings are from theirs (FIFO, no sale of
units not held), and valued at AMFI's latest NAV. They stay out of the stock report, which prices every holding
from the exchange."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from backend.database.sources.amfi import LIVE_DAYS, Scheme
from backend.database.sources.provider import DataUnavailable
from backend.user.portfolio.portfolio import Position, Transaction, build_positions


def positions(transactions: List[Dict[str, Any]]) -> Dict[int, Position]:
    """Open lots per AMFI scheme code; ValueError when a sale is larger than the units held then."""
    txs = [Transaction(symbol=str(t["scheme_code"]), side=t["side"], quantity=t["units"], price=t["nav"],
                       trade_date=t["trade_date"], fees=t.get("fees", 0.0)) for t in transactions]
    return {int(code): p for code, p in build_positions(txs).items()}


def holdings(transactions: List[Dict[str, Any]], schemes: Mapping[int, Scheme]) -> Dict[str, Any]:
    """Each fund still held at its latest NAV, and the totals. DataUnavailable when a held scheme has no recent
    NAV, because a total without it would look complete and be too low."""
    held = {code: p for code, p in positions(transactions).items() if p.quantity > 1e-9}
    missing = sorted(set(held) - set(schemes))
    if missing:
        raise DataUnavailable(f"AMFI's NAV file has no NAV in the last {LIVE_DAYS} days for the schemes {missing} "
                              "you hold, so your funds cannot be valued")
    rows = []
    for code, p in sorted(held.items()):
        s, value = schemes[code], p.quantity * schemes[code].nav
        rows.append({"code": code, "fund": s.fund, "plan": s.plan, "option": s.option, "units": round(p.quantity, 4),
                     "avg_nav": round(p.avg_cost, 4), "invested": round(p.cost_basis, 2), "nav": s.nav,
                     "nav_date": s.nav_date, "value": round(value, 2), "gain": round(value - p.cost_basis, 2),
                     "gain_pct": round((value / p.cost_basis - 1) * 100, 2)})
    invested = sum(p.cost_basis for p in held.values())
    value = sum(p.quantity * schemes[code].nav for code, p in held.items())
    return {"invested": round(invested, 2), "value": round(value, 2), "gain": round(value - invested, 2),
            "gain_pct": round((value / invested - 1) * 100, 2) if invested else None, "holdings": rows}
