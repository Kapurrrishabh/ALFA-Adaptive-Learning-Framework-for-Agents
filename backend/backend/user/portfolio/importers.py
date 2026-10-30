"""Broker exports (Zerodha, Groww) and our CSV template, read into rows and matched to NSE's listings.

The layouts come from third-party descriptions, not the brokers, so a file that fits none is refused, never
guessed at. Every row is kept and a gap in the file stays a gap. Only a statement's title, its header and the
rows below it are used, so no PAN, name or address of the account holder is read.
"""
from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from backend.config import canonical_symbol
from backend.database.sources.amfi import name_words
from backend.database.sources.panel import Listing

# provisional: 1,000 trades unzip to 0.7 MB; 50 MB is about 75,000 trades, which took 4 s and 62 MB to read on a
# Mac (2026-10-30). It is there to stop a small upload unpacking into gigabytes
MAX_UNZIPPED = 50_000_000
MAX_CHOICES = 3
STATES = ("matched", "check", "not_found")
ISIN = re.compile(r"^IN[A-Z0-9]{10}$")
SERIES = re.compile(r"-(BE|BZ)$")            # Zerodha writes the trading series after the symbol: "SYNTHB-BE"
COMPANY_WORDS = {"ltd", "limited"}          # brokers and NSE write these differently, so they never decide a match
ZERODHA_SET_ASIDE = {"Quantity Pledged (Margin)": "pledged for margin", "Quantity Pledged (Loan)": "pledged for a loan",
                     "Quantity Discrepant": "with a discrepancy"}

Cell = Callable[[str], object]


class ImportUnsupported(Exception):
    """A kind of file this server does not read; the route answers 501."""


@dataclass(frozen=True)
class Row:
    line: int
    name: Optional[str]
    isin: Optional[str]
    symbol: Optional[str]
    quantity: float
    price: Optional[float]
    trade_date: Optional[date]      # None in holdings files: they say what is held, not when it was bought
    side: str
    note: Optional[str]


@dataclass(frozen=True)
class Statement:
    source: str
    as_of: Optional[date]
    rows: List[Row]
    warnings: List[str]


@dataclass(frozen=True)
class Match:
    row: Row
    state: str
    listing: Optional[Listing]
    choices: List[Listing]


@dataclass(frozen=True)
class Layout:
    source: str
    columns: Tuple[str, ...]
    title: Optional[Tuple[str, str]]    # how a cell above the header starts, before the date; the date's format
    read: Callable[[Cell, int], Row]


def parse(data: bytes) -> Statement:
    """A broker export or our template, told apart by its first bytes and its header, never its file name."""
    if data.startswith(b"%PDF"):
        raise ImportUnsupported("PDF statements aren't switched on for this server")
    if data.startswith(b"PK\x03\x04"):
        return _statement(_sheets(data))
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("This file is not an Excel file (.xlsx), a PDF or a CSV file saved as UTF-8.") from None
    return _statement({"": list(csv.reader(io.StringIO(text)))})


def match(rows: Sequence[Row], directory: Dict[str, Listing], names: Dict[str, str]) -> List[Match]:
    """Each row to an NSE listing by ISIN, else by symbol. A name alone is never enough: it only offers choices,
    from `names` (symbol -> company name)."""
    by_symbol: Dict[str, Listing] = {}
    for listing in directory.values():
        by_symbol.setdefault(listing.symbol, listing)
    out = []
    for row in rows:
        listing = directory.get(row.isin) or by_symbol.get(SERIES.sub("", canonical_symbol(row.symbol or "")))
        choices = [] if listing or not row.name else _choices(row.name, by_symbol, names)
        out.append(Match(row, "matched" if listing else "check" if choices else "not_found", listing, choices))
    return out


def counts(matches: Sequence[Match]) -> Dict[str, int]:
    return {state: sum(m.state == state for m in matches) for state in STATES}


def _choices(name: str, by_symbol: Dict[str, Listing], names: Dict[str, str]) -> List[Listing]:
    """Listed companies with a word starting with each word of `name`, as fund search finds funds; shortest first."""
    wanted = set(name_words(name)) - COMPANY_WORDS
    if not wanted:
        return []
    found = []
    for symbol, company in names.items():
        have = name_words(company)
        if symbol in by_symbol and all(any(w.startswith(t) for w in have) for t in wanted):
            found.append((len(have), symbol))
    return [by_symbol[symbol] for _, symbol in sorted(found)[:MAX_CHOICES]]


def _sheets(data: bytes) -> Dict[str, List[Sequence[object]]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            unzipped = sum(entry.file_size for entry in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError(f"This file starts like an Excel file but cannot be opened ({exc}).") from exc
    # zipfile never inflates an entry past its declared size, so their sum bounds the work
    if unzipped > MAX_UNZIPPED:
        raise ValueError(f"This Excel file unpacks to {unzipped / 1e6:,.0f} MB; files over "
                         f"{MAX_UNZIPPED / 1e6:,.0f} MB are refused.")
    import openpyxl
    try:
        book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except KeyError as exc:                 # how openpyxl reports a zip that lacks a workbook's parts
        raise ValueError(f"This file is a zip archive but not an Excel workbook ({exc}).") from exc
    try:
        sheets = {}
        for sheet in book.worksheets:
            sheet.reset_dimensions()        # some exports state a wrong size, which read-only mode would trust
            sheets[sheet.title] = list(sheet.iter_rows(values_only=True))
        return sheets
    finally:
        book.close()


def _statement(sheets: Dict[str, List[Sequence[object]]]) -> Statement:
    fits = [(layout, title, rows, fit) for title, rows in sheets.items() for layout in LAYOUTS
            if (fit := _fit(layout, rows)) is not None]
    if not fits:
        raise ValueError(f"This file doesn't match a layout we read. We read {'; '.join(map(_describe, LAYOUTS))}.")
    if len(fits) > 1:
        raise ValueError("This file matches more than one layout: "
                         + ", ".join(f"{layout.source} on sheet {title!r}" for layout, title, *_ in fits)
                         + ". Upload one statement at a time.")
    layout, title, rows, (as_of, header, columns) = fits[0]
    read = [layout.read(_cells(row, columns), line) for line, row in enumerate(rows[header + 1:], start=header + 2)
            if any(_text(value) for value in row)]
    others = [t for t in sheets if t != title]
    warnings = [f"Only the {title!r} sheet was read, not {' or '.join(map(repr, others))}."] if others else []
    return Statement(layout.source, as_of, read, warnings)


def _fit(layout: Layout, rows: List[Sequence[object]]) -> Optional[Tuple[Optional[date], int, Dict[str, int]]]:
    """(the statement's date, the header's index, each column's position) when the rows have the layout's header,
    below its title if it has one."""
    for header, row in enumerate(rows):
        cells = [_text(value) for value in row]
        if all(column in cells for column in layout.columns):
            break
    else:
        return None
    columns = {column: cells.index(column) for column in layout.columns}
    if layout.title is None:
        return None, header, columns
    title, form = layout.title
    found = next((c for row in rows[:header] for c in map(_text, row) if c and c.startswith(title + " ")), None)
    if found is None:
        return None
    try:
        return datetime.strptime(found[len(title) + 1:], form).date(), header, columns
    except ValueError:
        raise ValueError(f"The title {found!r} does not end in a date we can read.") from None


def _describe(layout: Layout) -> str:
    title = f"a sheet titled '{layout.title[0]} <date>' with " if layout.title else ""
    return f"{layout.source}: {title}the columns {', '.join(layout.columns)}"


def _cells(row: Sequence[object], columns: Dict[str, int]) -> Cell:
    return lambda column: row[columns[column]] if columns[column] < len(row) else None


def _text(value: object) -> Optional[str]:
    text = "" if value is None else " ".join(str(value).split())
    return text or None


def _number(cell: Cell, column: str, line: int, required: bool) -> Optional[float]:
    value = cell(column)
    if _text(value) is None:
        if required:
            raise ValueError(f"Row {line}: {column} is empty.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not 0 <= number < math.inf:
        raise ValueError(f"Row {line}: {column} is {value!r}, not a number of 0 or more.")
    return number


def _price(cell: Cell, column: str, line: int) -> Optional[float]:
    # a holdings file can give 0 (bonus or gifted shares), which is no cost a portfolio can use, so it is asked for
    return _number(cell, column, line, required=False) or None


def _date(cell: Cell, column: str, line: int) -> Optional[date]:
    value = cell(column)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _text(value)
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        raise ValueError(f"Row {line}: {column} is {text!r}, not a date written as YYYY-MM-DD.") from None


def _side(cell: Cell, column: str, line: int) -> str:
    side = (_text(cell(column)) or "").upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Row {line}: {column} is {cell(column)!r}, not buy or sell.")
    return side


def _zerodha_holding(cell: Cell, line: int) -> Row:
    held_long = _number(cell, "Quantity Long Term", line, required=False)
    aside = [f"{n:.10g} {label}" for column, label in ZERODHA_SET_ASIDE.items()
             if (n := _number(cell, column, line, required=False))]
    notes = ([f"{held_long:.10g} of these were held over a year."] if held_long else []) + \
        ([f"The file also shows {' and '.join(aside)}. Check the quantity is right."] if aside else [])
    return Row(line, name=None, isin=_text(cell("ISIN")), symbol=_text(cell("Symbol")),
               quantity=_number(cell, "Quantity Available", line, required=True),
               price=_price(cell, "Average Price", line), trade_date=None, side="BUY", note=" ".join(notes) or None)


def _groww_holding(cell: Cell, line: int) -> Row:
    return Row(line, name=_text(cell("Stock Name")), isin=_text(cell("ISIN")), symbol=None,
               quantity=_number(cell, "Quantity", line, required=True), price=_price(cell, "Average buy price", line),
               trade_date=None, side="BUY", note=None)


def _trade(cell: Cell, line: int) -> Row:
    return Row(line, name=None, isin=_text(cell("isin")), symbol=_text(cell("symbol")),
               quantity=_number(cell, "quantity", line, required=True), price=_price(cell, "price", line),
               trade_date=_date(cell, "trade_date", line), side=_side(cell, "trade_type", line), note=None)


def _template_row(cell: Cell, line: int) -> Row:
    given = (_text(cell("symbol_or_isin")) or "").upper()
    is_isin = bool(ISIN.match(given))
    return Row(line, name=None, isin=given if is_isin else None, symbol=None if is_isin else given or None,
               quantity=_number(cell, "quantity", line, required=True), price=_price(cell, "price", line),
               trade_date=_date(cell, "trade_date", line), side=_side(cell, "side", line), note=None)


LAYOUTS = (
    Layout("Zerodha holdings",
           ("Symbol", "ISIN", "Quantity Available", "Quantity Long Term", "Average Price", *ZERODHA_SET_ASIDE),
           ("Equity Holdings Statement as on", "%Y-%m-%d"), _zerodha_holding),
    Layout("Groww stocks", ("Stock Name", "ISIN", "Quantity", "Average buy price"),
           ("Holdings statement for stocks as on", "%d-%m-%Y"), _groww_holding),
    Layout("Zerodha tradebook", ("symbol", "isin", "trade_date", "trade_type", "quantity", "price"), None, _trade),
    Layout("CSV template", ("symbol_or_isin", "side", "quantity", "price", "trade_date"), None, _template_row),
)
