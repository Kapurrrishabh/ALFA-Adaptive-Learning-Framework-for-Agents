"""CAS statements (CAMS, KFintech, NSDL, CDSL), read by casparser (the optional `cas` extra) in a process of its own,
as it runs native PDF code on files from users. The route sends a PDF and its password here, not to importers.parse,
so importers never needs to import this module.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from backend.user.portfolio.importers import PDF_SWITCHED_OFF, SHORTER_RANGE, ImportUnsupported, Row, Statement

CAS_TIMEOUT_S = 30          # provisional: no real statement has been timed; confirm with a long CAMS one on the server
WORKER = Path(__file__).with_name("cas_worker.py")
FUND_HOUSE_FILES = ("CAMS", "KFINTECH")      # the rest, NSDL and CDSL, list demat holdings without transactions
SIDES = {"PURCHASE": "BUY", "PURCHASE_SIP": "BUY", "SWITCH_IN": "BUY", "SWITCH_IN_MERGER": "BUY",
         "DIVIDEND_REINVEST": "BUY", "REDEMPTION": "SELL", "SWITCH_OUT": "SELL", "SWITCH_OUT_MERGER": "SELL"}
STAMP_DUTY = "STAMP_DUTY_TAX"
# casparser copies the statement's period as printed: "01-Apr-2024", and from CDSL also "01-09-2024" or "01/09/2024"
PERIOD_DATES = ("%d-%b-%Y", "%d-%m-%Y")
PARTLY_READ = "Part of this statement didn't add up when it was read, so check this row against it."


def read(data: bytes, password: str) -> Dict[str, object]:
    """Holdings and transactions as casparser writes them (its keys, decimals as text), plus parse_warning_count. A
    file casparser cannot read raises ValueError, which the route answers with 400."""
    request = json.dumps({"pdf": base64.b64encode(data).decode(), "password": password}).encode()
    try:
        # env={} only keeps the server's secrets out of its environment; it runs as the same user, so it is no sandbox
        done = subprocess.run([sys.executable, "-I", str(WORKER)], input=request, capture_output=True,
                              timeout=CAS_TIMEOUT_S, env={})
    except subprocess.TimeoutExpired:
        raise ValueError(f"This statement took more than {CAS_TIMEOUT_S} seconds to read, so we stopped. "
                         f"{SHORTER_RANGE}") from None
    if done.returncode:     # stderr is left out: the PDF libraries write there, and nothing checks it for the statement
        raise RuntimeError(f"The CAS reader stopped with exit code {done.returncode}.")
    try:
        reply = json.loads(done.stdout)
    except ValueError as exc:   # our fault, not the file's, so not a 400
        raise RuntimeError(f"The CAS reader's reply is not JSON ({exc}). "
                           "Check that nothing in the reader prints to stdout.") from None
    error = reply.get("error")
    if error == "missing":
        raise ImportUnsupported(PDF_SWITCHED_OFF)
    if error == "password":
        raise ValueError("That password didn't open the file. For CAMS or KFintech it is the one you chose; "
                         "for NSDL or CDSL it is usually your PAN in capitals.")
    if error == "unreadable":
        raise ValueError(f"This PDF isn't a CAS statement we can read: {reply['detail']}")
    if error == "failed":
        raise RuntimeError(f"The CAS reader failed with {reply['detail']}.")
    return reply["statement"]


def statement(parsed: Dict[str, object]) -> Statement:
    """What `read` returns, as rows for importers.match and match_funds. Purchases and sales become rows, stamp duty
    becomes their fees, and every other kind of row is left out and named in a warning. A summary CAS, which lists
    no purchases, raises ValueError."""
    as_of = _period_end(parsed["statement_period"]["to"])
    doubt = PARTLY_READ if parsed["parse_warning_count"] else None
    source = f"{parsed['file_type']} CAS"
    if parsed["file_type"] not in FUND_HOUSE_FILES:
        return _demat(parsed, source, as_of, doubt)
    if parsed["cas_type"] == "SUMMARY":
        raise ValueError("This is a summary CAS, which lists no purchases or sales. Ask CAMS or KFintech for the "
                         "detailed one and upload that.")
    if parsed["cas_type"] != "DETAILED":    # casparser refuses any other kind itself, so this is a change in casparser
        raise RuntimeError(f"The CAS reader gave a CAS of kind {parsed['cas_type']!r}, neither detailed nor summary.")
    left_out: Counter = Counter()
    funds = [row for folio in parsed["folios"] for scheme in folio["schemes"]
             for row in _scheme_rows(scheme, parsed["statement_period"]["from"], doubt, left_out)]
    kinds = ", ".join(f"{kind.lower().replace('_', ' ')} ({n})" for kind, n in sorted(left_out.items()))
    warnings = [f"This statement's rows that are not purchases or sales are left out: {kinds}."] if left_out else []
    return Statement(source, as_of, [], funds, warnings)


def _period_end(text: str) -> Optional[date]:
    """The statement's last day, or None where casparser found no period and wrote ''."""
    if not text:
        return None
    for form in PERIOD_DATES:
        try:
            return datetime.strptime(text.replace("/", "-"), form).date()
        except ValueError:
            pass
    raise ValueError(f"This statement's period ends on {text!r}, which we can't read as a day, month and year.")


def _scheme_rows(scheme: Dict, began: str, doubt: Optional[str], left_out: Counter) -> List[Row]:
    """The units a scheme held when the statement began, each purchase and sale, then any units the statement ends
    with that these don't give. A stamp duty row is added to the purchase it follows on the same day."""
    trades: List[Dict] = []
    for t in scheme["transactions"]:
        if t["type"] in SIDES:
            trades.append({**t, "units": abs(Decimal(t["units"])), "side": SIDES[t["type"]], "fees": 0.0})
        elif t["type"] == STAMP_DUTY and trades and (trades[-1]["side"], trades[-1]["date"]) == ("BUY", t["date"]):
            trades[-1]["fees"] += float(t["amount"])
        else:
            left_out[t["type"]] += 1
    opened = Decimal(scheme["open"])
    held = opened + sum((t["units"] if t["side"] == "BUY" else -t["units"] for t in trades), Decimal(0))
    unexplained = Decimal(scheme["close"]) - held
    if unexplained:
        doubt = (f"This fund's purchases and sales add up to {held} units, but the statement ends with "
                 f"{scheme['close']}. Check them against it.")
    code = int(scheme["amfi"]) if scheme["amfi"] else None
    rows = [Row(None, name=scheme["scheme"], isin=scheme["isin"], symbol=None, quantity=float(opened), price=None,
                trade_date=None, side="BUY", note=f"Held when the statement began{' on ' + began if began else ''}, "
                "so it gives no date or cost.", fees=None, scheme_code=code, doubt=doubt)] if opened else []
    # without the last row, a fund whose units all came as a gift or a segregation would leave no row at all
    return rows + [Row(None, name=scheme["scheme"], isin=scheme["isin"], symbol=None,
                       quantity=float(t["units"]), price=float(t["nav"] or 0) or None,
                       trade_date=date.fromisoformat(t["date"]), side=t["side"], note=None, fees=t["fees"],
                       scheme_code=code, doubt=doubt) for t in trades] + (
        [Row(None, name=scheme["scheme"], isin=scheme["isin"], symbol=None, quantity=float(unexplained), price=None,
             trade_date=None, side="BUY", note="The statement ends with these units but shows no purchase of them, "
             "so it gives no date or cost.", fees=None, scheme_code=code, doubt=doubt)] if unexplained > 0 else [])


def _demat(parsed: Dict[str, object], source: str, as_of: Optional[date], doubt: Optional[str]) -> Statement:
    """NSDL and CDSL list what each demat account holds, with no dates, and no cost for shares. A fund's ISIN is the
    depository's, and casparser finds its AMFI code from that ISIN, so the code is not passed on to check it by."""
    stocks, funds, bonds = [], [], 0
    for account in parsed["accounts"]:
        stocks += [Row(None, name=e["name"], isin=e["isin"], symbol=None, quantity=float(e["num_shares"]),
                       price=None, trade_date=None, side="BUY", note=None, fees=None, scheme_code=None, doubt=doubt)
                   for e in account["equities"]]
        funds += [Row(None, name=m["name"], isin=m["isin"], symbol=None, quantity=float(m["balance"]),
                      price=float(m["avg_cost"] or 0) or None, trade_date=None, side="BUY", note=None, fees=None,
                      scheme_code=None, doubt=doubt) for m in account["mutual_funds"]]
        bonds += len(account["bonds"])
    pensions = len(parsed["nps"]["schemes"]) if parsed["nps"] else 0      # null without an NPS account
    left = [f"{kind} ({n})" for kind, n in (("bonds", bonds), ("NPS pension schemes", pensions)) if n]
    warnings = [f"This statement's {' and '.join(left)} are left out: a portfolio here holds only shares and "
                "funds."] if left else []
    return Statement(source, as_of, stocks, funds, warnings)
