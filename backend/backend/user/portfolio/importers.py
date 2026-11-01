"""Broker exports (Zerodha, Groww) and our CSV template, read into rows and matched to NSE's listings and AMFI's funds.

The layouts come from third-party descriptions, not the brokers, so a file that fits none is refused, never
guessed at. Every row is kept and a gap in the file stays a gap. Only a statement's title, its header and the
rows below it are used, so no PAN, name or address of the account holder is read.
"""
from __future__ import annotations

import csv
import io
import itertools
import math
import re
import zipfile
import zlib
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from backend.config import canonical_symbol
from backend.database.sources.amfi import DIRECT, Scheme, name_words, search
from backend.database.sources.panel import Listing

# provisional (2026-10-30): the worst hostile .xlsx we built took 120 times its unzipped size in memory (a styles
# part of empty formats); that is a measurement, not a bound. A trade takes about 680 bytes unzipped and 110 as CSV,
# so about 1,400 and 8,800 trades fit
MAX_TEXT_BYTES = 1_000_000
MAX_ROWS = 20_000           # more rows than 1 MB of real trades holds
MAX_COLUMNS = 26            # A to Z: every layout's columns sit well inside it
MAX_SHEETS = 10             # broker workbooks have one to three
MAX_CELL_CHARS = 500        # provisional: the headers, titles and names we read run under 100
SHOWN_CHARS = 100           # of a file's own text in a message: Excel stops sheet names at 31, made-up files don't
# what damaged workbooks raised when fuzzed (2026-10-30); anything else is our bug, so it stays a server error
DAMAGED = (LookupError, SyntaxError, TypeError, OSError, NotImplementedError, zipfile.BadZipFile, zlib.error)
SHORTER_RANGE = "Download a shorter date range and upload each file."
PDF_SWITCHED_OFF = "PDF statements aren't switched on for this server. Upload your broker's Excel or CSV file instead."
MAX_CHOICES = 3
# provisional: one fund search took 20 to 40 ms on AMFI's list of 2026-10-28, so this many take about 3 s
MAX_FUND_NAMES = 100
STATES = ("matched", "check", "not_found")
SIDES = {"BUY": "BUY", "SELL": "SELL"}
GROWW_SIDES = {"PURCHASE": "BUY", "REDEEM": "SELL"}
ISO_DATE = "%Y-%m-%d"
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
    line: Optional[int]             # None in a PDF statement
    name: Optional[str]
    isin: Optional[str]
    symbol: Optional[str]
    quantity: float
    price: Optional[float]
    trade_date: Optional[date]      # None in holdings files: they say what is held, not when it was bought
    side: str
    note: Optional[str]
    fees: Optional[float]           # None where the file lists no charges
    scheme_code: Optional[int]      # a fund's AMFI code, where the file gives one
    doubt: Optional[str]            # why the row is only offered even when it is found


@dataclass(frozen=True)
class Statement:
    source: str
    as_of: Optional[date]
    rows: List[Row]                 # shares, ETFs and REITs, for `match`
    funds: List[Row]                # mutual funds, for `match_funds`
    warnings: List[str]


@dataclass(frozen=True)
class Match:
    row: Row
    state: str
    listing: Optional[Listing]
    choices: List[Listing]


@dataclass(frozen=True)
class FundMatch:
    row: Row
    state: str
    scheme: Optional[Scheme]
    choices: List[Scheme]


@dataclass(frozen=True)
class Layout:
    source: str
    columns: Tuple[str, ...]
    title: Optional[Tuple[str, str]]    # how a cell above the header starts, before the date; the date's format
    read: Callable[[Cell, int], Row]
    is_fund: bool


def parse(data: bytes) -> Statement:
    """A broker export or our template, told apart by its first bytes and its header, never its file name. A file it
    cannot read raises ValueError, which the route answers with 400; any other error is a bug."""
    if data.startswith(b"%PDF"):
        raise ImportUnsupported(PDF_SWITCHED_OFF)
    if data.startswith(b"PK\x03\x04"):
        return _statement(_sheets(data))
    if len(data) > MAX_TEXT_BYTES:
        raise ValueError(f"This CSV file is {math.ceil(len(data) / 1e3):,} KB; we read files of up to "
                         f"{MAX_TEXT_BYTES // 1000:,} KB. {SHORTER_RANGE}")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("This file is not an Excel file (.xlsx), a PDF or a CSV file saved as UTF-8.") from None
    reader = csv.reader(io.StringIO(text))
    try:
        rows = _first_rows(reader, before=0)
    except csv.Error as exc:
        raise ValueError(f"Line {reader.line_num} of this CSV file cannot be read: {exc}.") from exc
    return _statement({"": rows})


def match(rows: Sequence[Row], directory: Dict[str, Listing], names: Dict[str, str]) -> List[Match]:
    """Each row to an NSE listing by its ISIN, or by its symbol when it has none. Anything less sure is only offered
    in `choices`: a symbol whose ISIN the lists lack, a name (from `names`, symbol -> company name), any row
    with a quantity of 0, which Zerodha shows when every share is pledged, and any row with a doubt."""
    by_symbol: Dict[str, Listing] = {}
    for listing in directory.values():
        by_symbol.setdefault(listing.symbol, listing)
    out = []
    for row in rows:
        by_its_symbol = by_symbol.get(SERIES.sub("", canonical_symbol(row.symbol or "")))
        listing = directory.get(row.isin) if row.isin else by_its_symbol
        if listing and row.quantity and not row.doubt:
            out.append(Match(row, "matched", listing, []))
            continue
        # an ISIN the lists lack can be an old one (a split changes it) or another share that now has the symbol
        offered = listing or by_its_symbol
        choices = [offered] if offered else _choices(row.name, by_symbol, names) if row.name else []
        out.append(Match(row, "check" if choices else "not_found", None, choices))
    return out


def match_funds(rows: Sequence[Row], schemes: List[Scheme]) -> List[FundMatch]:
    """Each fund row to one of AMFI's current `schemes` by its ISIN. As in `match`, anything less sure is only
    offered: an ISIN whose row gives an AMFI code of another scheme, or a name that says the other plan (which is
    then offered first); a name alone; a quantity of 0; and a doubt. A file naming more than MAX_FUND_NAMES funds
    to find by name raises ValueError, which the route answers with 400."""
    by_isin = {isin: s for s in schemes for isin in (s.isin_growth, s.isin_reinvestment) if isin}
    by_code = {s.code: s for s in schemes}
    plans: Dict[Tuple[str, str, Optional[str]], List[Scheme]] = {}
    for s in schemes:
        plans.setdefault((s.amc, s.fund, s.option), []).append(s)
    by_name: Dict[str, List[Scheme]] = {}
    out = []
    for row in rows:
        scheme = by_isin.get(row.isin) if row.isin else None
        offered = [scheme] if scheme else []
        if row.scheme_code is not None:
            # casparser finds a CAMS scheme's ISIN and code from its name, and can give the other plan a full score
            coded = by_code.get(row.scheme_code)
            offered += [coded] if coded and coded is not scheme else []
            says_direct = "direct" in name_words(row.name or "")
            if coded is not scheme:
                scheme = None
            elif scheme and says_direct != (scheme.plan == DIRECT):
                offered = sorted(plans[scheme.amc, scheme.fund, scheme.option],
                                 key=lambda s: (s.plan == DIRECT) != says_direct)[:MAX_CHOICES]
                scheme = None
        if scheme and row.quantity and not row.doubt:
            out.append(FundMatch(row, "matched", scheme, []))
            continue
        if not offered and row.name:
            if row.name not in by_name:
                if len(by_name) == MAX_FUND_NAMES:
                    raise ValueError(f"This file names more than {MAX_FUND_NAMES} funds that we have to find by "
                                     "name. Upload a statement with fewer funds.")
                by_name[row.name] = _fund_choices(row.name, schemes, by_code)
            offered = by_name[row.name]
        out.append(FundMatch(row, "check" if offered else "not_found", None, offered))
    return out


def counts(matches: Sequence[Union[Match, FundMatch]]) -> Dict[str, int]:
    return {state: sum(m.state == state for m in matches) for state in STATES}


def _fund_choices(name: str, schemes: List[Scheme], by_code: Dict[int, Scheme]) -> List[Scheme]:
    """For each of the first funds that fund search finds for `name`, the plan it lists first."""
    try:
        funds = search(schemes, name, limit=MAX_CHOICES)
    except ValueError:      # too little of a name to search by
        return []
    return [by_code[fund["plans"][0]["code"]] for fund in funds]


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
            locked = [entry.filename for entry in archive.infolist() if entry.flag_bits & 1]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"This file starts like an Excel file but cannot be opened ({exc}).") from exc
    if locked:      # Excel's own passwords make a different kind of file, so only a hand-made zip gets here
        raise ValueError(f"This Excel file is damaged or is not a workbook ({_short(locked[0])!r} is encrypted).")
    # zipfile never inflates an entry past its declared size, so their sum bounds each pass over the file
    if unzipped > MAX_TEXT_BYTES:
        raise ValueError(f"This Excel file unpacks to {math.ceil(unzipped / 1e3):,} KB; we read files of up to "
                         f"{MAX_TEXT_BYTES // 1000:,} KB. {SHORTER_RANGE}")
    from openpyxl.reader.excel import ExcelReader
    try:
        reader = ExcelReader(io.BytesIO(data), read_only=True, data_only=True, keep_links=False)
        reader.read_manifest()
        reader.read_workbook()
        if len(reader.parser.sheets) > MAX_SHEETS:
            raise ValueError(f"This workbook has {len(reader.parser.sheets):,} sheets; we read files with up to "
                             f"{MAX_SHEETS}. Upload the file as your broker gave it.")
        # openpyxl parses a part once for each sheet that names it, and even in read-only mode reads a chart
        # sheet's drawings once per reference, with no cap on either
        found = list(reader.parser.find_sheets())
        if len({rel.target for _, rel in found}) < len(found):
            raise ValueError("This Excel file is damaged or is not a workbook (two of its sheets name one part).")
        chart = next((sheet.name for sheet, rel in found if "chartsheet" in rel.Type), None)
        if chart is not None:
            raise ValueError(f"Sheet {_short(chart)!r} is a chart, not a table. "
                             "Upload the file as your broker gave it.")
        reader.read()
        book = reader.wb
        try:
            sheets, rows_read = {}, 0
            for sheet in book.worksheets:
                sheet.reset_dimensions()    # some exports state a wrong size, which read-only mode would trust
                # without max_col, openpyxl pads every row out to its last cell, even one in column XFD
                rows = sheet.iter_rows(max_col=MAX_COLUMNS, values_only=True)
                sheets[sheet.title] = _first_rows(_readable(rows, sheet.title), before=rows_read)
                rows_read += len(sheets[sheet.title])
            return sheets
        finally:
            book.close()
    except DAMAGED as exc:
        raise ValueError(f"This Excel file is damaged or is not a workbook ({type(exc).__name__}: "
                         f"{_short(str(exc))}).") from exc


def _readable(rows: Iterable[Sequence[object]], title: str) -> Iterator[Sequence[object]]:
    try:
        yield from rows
    except ValueError as exc:       # from openpyxl reading a cell; Python's own message can name a Python setting
        raise ValueError(f"This Excel file is damaged or is not a workbook (sheet {_short(title)!r} has a cell "
                         "that cannot be read).") from exc


def _first_rows(rows: Iterable[Sequence[object]], before: int) -> List[Sequence[object]]:
    """The rows, refused once they and the `before` rows of earlier sheets pass MAX_ROWS; stops reading there, as
    openpyxl makes a blank row for every row number a sheet skips."""
    kept = list(itertools.islice(rows, MAX_ROWS + 1 - before))
    if before + len(kept) > MAX_ROWS:
        raise ValueError(f"This file has more than {MAX_ROWS:,} rows, counting blank ones. {SHORTER_RANGE}")
    return kept


def _statement(sheets: Dict[str, List[Sequence[object]]]) -> Statement:
    texts = {title: [[_text(value) for value in row] for row in rows] for title, rows in sheets.items()}
    fits = [(layout, title, rows, fit) for title, rows in sheets.items() for layout in LAYOUTS
            if (fit := _fit(layout, texts[title])) is not None]
    if not fits:
        raise ValueError(f"This file doesn't match a layout we read. We read {'; '.join(map(_describe, LAYOUTS))}.")
    statements = {(layout.source, as_of) for layout, _, _, (as_of, *_) in fits}
    titles = [title for _, title, *_ in fits]
    # one statement can take two sheets: Zerodha puts shares on one and funds on another, both dated alike
    if len(statements) > 1 or len({layout for layout, *_ in fits}) < len(fits) or len(set(titles)) < len(fits):
        raise ValueError("This file matches more than one layout: "
                         + ", ".join(f"{layout.source} on sheet {_short(title)!r}" for layout, title, *_ in fits)
                         + ". Upload one statement at a time.")
    (source, as_of), = statements
    read: Dict[bool, List[Row]] = {False: [], True: []}
    for layout, title, rows, (_, header, columns) in fits:
        try:
            read[layout.is_fund] += [layout.read(_cells(row, columns, line), line) for line, row
                                     in enumerate(rows[header + 1:], start=header + 2) if any(texts[title][line - 1])]
        except ValueError as exc:   # a row number alone doesn't say which sheet of a workbook it is on
            if len(sheets) == 1:
                raise
            raise ValueError(f"Sheet {_short(title)!r}, {exc}") from exc
    others = [repr(_short(t)) for t in sheets if t not in titles]
    sheets_read = f"{' and '.join(repr(_short(t)) for t in titles)} sheet{'s were' if len(titles) > 1 else ' was'}"
    warnings = [f"Only the {sheets_read} read, not {' or '.join(others)}."] if others else []
    return Statement(source, as_of, read[False], read[True], warnings)


def _fit(layout: Layout, texts: List[List[Optional[str]]]) -> Optional[Tuple[Optional[date], int, Dict[str, int]]]:
    """(the statement's date, the header's index, each column's position) when the rows' `texts` have the layout's
    header, below its title if it has one."""
    for header, cells in enumerate(texts):
        if all(column in cells for column in layout.columns):
            break
    else:
        return None
    columns = {column: cells.index(column) for column in layout.columns}
    if layout.title is None:
        return None, header, columns
    title, form = layout.title
    found = next((c for row in texts[:header] for c in row if c and c.startswith(title + " ")), None)
    if found is None:
        return None
    try:
        return datetime.strptime(found[len(title) + 1:], form).date(), header, columns
    except ValueError:
        raise ValueError(f"The title {_short(found)!r} does not end in a date we can read.") from None


def _describe(layout: Layout) -> str:
    title = f"a sheet titled '{layout.title[0]} <date>' with " if layout.title else ""
    return f"{layout.source}: {title}the columns {', '.join(layout.columns)}"


def _cells(row: Sequence[object], columns: Dict[str, int], line: int) -> Cell:
    def cell(column: str) -> object:
        value = row[columns[column]] if columns[column] < len(row) else None
        if len(str(value)) > MAX_CELL_CHARS:
            raise ValueError(f"Row {line}: {column} has {len(str(value)):,} characters; we read cells of up to "
                             f"{MAX_CELL_CHARS}. Shorten it and upload the file again.")
        return value
    return cell


def _short(text: str) -> str:
    return text if len(text) <= SHOWN_CHARS else text[:SHOWN_CHARS] + "..."


def _text(value: object) -> Optional[str]:
    text = "" if value is None else str(value)
    # _cells refuses a long cell before it is read; tidying one that many .xlsx cells share costs its length per cell
    return text if len(text) > MAX_CELL_CHARS else " ".join(text.split()) or None


def _number(cell: Cell, column: str, line: int, required: bool) -> Optional[float]:
    value = cell(column)
    if _text(value) is None:
        if required:
            raise ValueError(f"Row {line}: {column} is empty.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):     # a long .xlsx number arrives as an int too big for a float
        number = math.nan
    if not 0 <= number < math.inf:
        raise ValueError(f"Row {line}: {column} is {value!r}, not a number of 0 or more.")
    return number


def _price(cell: Cell, column: str, line: int) -> Optional[float]:
    # a holdings file can give 0 (bonus or gifted shares), which is no cost a portfolio can use, so it is asked for
    return _number(cell, column, line, required=False) or None


def _date(cell: Cell, column: str, line: int, form: str) -> Optional[date]:
    value = cell(column)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _text(value)
    try:
        return datetime.strptime(text, form).date() if text else None
    except ValueError:
        raise ValueError(f"Row {line}: {column} is {text!r}, not a date written like "
                         f"{datetime(2026, 1, 15).strftime(form)}.") from None


def _side(cell: Cell, column: str, line: int, sides: Dict[str, str]) -> str:
    side = (_text(cell(column)) or "").upper()
    if side not in sides:
        raise ValueError(f"Row {line}: {column} is {cell(column)!r}, not {' or '.join(sides).lower()}.")
    return sides[side]


def _set_aside(cell: Cell, line: int) -> List[str]:
    """The note on what Zerodha holds back, which its shares and funds sheets list alike."""
    aside = [f"{n:.10g} {label}" for column, label in ZERODHA_SET_ASIDE.items()
             if (n := _number(cell, column, line, required=False))]
    return [f"The file also shows {' and '.join(aside)}. Check the quantity is right."] if aside else []


def _zerodha_holding(cell: Cell, line: int) -> Row:
    held_long = _number(cell, "Quantity Long Term", line, required=False)
    notes = ([f"{held_long:.10g} of these were held over a year."] if held_long else []) + _set_aside(cell, line)
    return Row(line, name=None, isin=_text(cell("ISIN")), symbol=_text(cell("Symbol")),
               quantity=_number(cell, "Quantity Available", line, required=True),
               price=_price(cell, "Average Price", line), trade_date=None, side="BUY", note=" ".join(notes) or None,
               fees=None, scheme_code=None, doubt=None)


def _zerodha_fund(cell: Cell, line: int) -> Row:
    return Row(line, name=_text(cell("Symbol")), isin=_text(cell("ISIN")), symbol=None,
               quantity=_number(cell, "Quantity Available", line, required=True),
               price=_price(cell, "Average Price", line), trade_date=None, side="BUY",
               note=" ".join(_set_aside(cell, line)) or None, fees=None, scheme_code=None, doubt=None)


def _groww_holding(cell: Cell, line: int) -> Row:
    return Row(line, name=_text(cell("Stock Name")), isin=_text(cell("ISIN")), symbol=None,
               quantity=_number(cell, "Quantity", line, required=True), price=_price(cell, "Average buy price", line),
               trade_date=None, side="BUY", note=None, fees=None, scheme_code=None, doubt=None)


def _groww_fund_order(cell: Cell, line: int) -> Row:
    return Row(line, name=_text(cell("Scheme Name")), isin=None, symbol=None,
               quantity=_number(cell, "Units", line, required=True), price=_price(cell, "NAV", line),
               trade_date=_date(cell, "Date", line, "%d %b %Y"),
               side=_side(cell, "Transaction Type", line, GROWW_SIDES), note=None, fees=None, scheme_code=None,
               doubt=None)


def _trade(cell: Cell, line: int) -> Row:
    return Row(line, name=None, isin=_text(cell("isin")), symbol=_text(cell("symbol")),
               quantity=_number(cell, "quantity", line, required=True), price=_price(cell, "price", line),
               trade_date=_date(cell, "trade_date", line, ISO_DATE), side=_side(cell, "trade_type", line, SIDES),
               note=None, fees=None, scheme_code=None, doubt=None)


def _template_row(cell: Cell, line: int) -> Row:
    given = (_text(cell("symbol_or_isin")) or "").upper()
    is_isin = bool(ISIN.match(given))
    return Row(line, name=None, isin=given if is_isin else None, symbol=None if is_isin else given or None,
               quantity=_number(cell, "quantity", line, required=True), price=_price(cell, "price", line),
               trade_date=_date(cell, "trade_date", line, ISO_DATE), side=_side(cell, "side", line, SIDES),
               note=None, fees=None, scheme_code=None, doubt=None)


LAYOUTS = (
    Layout("Zerodha holdings",
           ("Symbol", "ISIN", "Quantity Available", "Quantity Long Term", "Average Price", *ZERODHA_SET_ASIDE),
           ("Equity Holdings Statement as on", ISO_DATE), _zerodha_holding, is_fund=False),
    Layout("Zerodha holdings", ("Symbol", "ISIN", "Quantity Available", "Average Price", *ZERODHA_SET_ASIDE),
           ("Mutual Funds Holdings Statement as on", ISO_DATE), _zerodha_fund, is_fund=True),
    Layout("Groww stocks", ("Stock Name", "ISIN", "Quantity", "Average buy price"),
           ("Holdings statement for stocks as on", "%d-%m-%Y"), _groww_holding, is_fund=False),
    # Groww gives funds only as orders, with neither an ISIN nor an AMFI code
    Layout("Groww mutual funds", ("Scheme Name", "Transaction Type", "Units", "NAV", "Date"), None,
           _groww_fund_order, is_fund=True),
    Layout("Zerodha tradebook", ("symbol", "isin", "trade_date", "trade_type", "quantity", "price"), None, _trade,
           is_fund=False),
    Layout("CSV template", ("symbol_or_isin", "side", "quantity", "price", "trade_date"), None, _template_row,
           is_fund=False),
)
