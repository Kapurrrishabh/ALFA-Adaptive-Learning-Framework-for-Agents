"""SEC EDGAR filings. The highest-volume source, and a US Government work, so public domain.

Two measurements shaped this module. A full submission .txt averages 26 MB because it carries
every exhibit and image; the primary document alone averages 4 MB. And the extracted text of one
10-K is about 250 KB. So this fetches the primary document only, extracts on the way in, and
stores the text — roughly 5x less bandwidth and 100x less disk than keeping raw submissions.
"""

from .. import config
from ..common.http import BlockedByHost
from ..extract import html_text

ARCHIVES = "https://www.sec.gov/Archives"
FULL_INDEX = ARCHIVES + "/edgar/full-index/{year}/QTR{quarter}/form.idx"
LICENCE = "US Government work, public domain (17 USC 105)"
SOURCE = "sec_edgar"

_FIELDS_AFTER_COMPANY = 3
# Exhibits and cover pages are separate .htm files. The filing itself is always the largest.
_EXHIBIT_MARKERS = ("index", "ex-", "exhibit", "ex_", "cover")
_MINIMUM_TEXT_CHARACTERS = 2000


def _parse_form_index(text, wanted_forms):
    """Returns (form_type, company, cik, date, path) for the forms we asked for.

    form.idx looks fixed-width but the column starts have moved over the years, so fields are
    read from the ends instead: the last three are always path, date, CIK, and the form type is
    always the first token. Only the company name in between can contain spaces.
    """
    wanted = {form.upper() for form in wanted_forms}
    rows = []
    seen_header = False
    for line in text.splitlines():
        if not seen_header:
            # The data starts after a run of dashes; everything above is preamble.
            seen_header = set(line.strip()) == {"-"}
            continue
        parts = line.split()
        if len(parts) < 2 + _FIELDS_AFTER_COMPANY:
            continue
        form_type = parts[0].upper()
        if form_type not in wanted:
            continue
        path, date, cik = parts[-1], parts[-2], parts[-3]
        rows.append((form_type, " ".join(parts[1:-_FIELDS_AFTER_COMPANY]), cik, date, path))
    return rows


def _primary_document_url(session, cik, accession):
    """The filing's own document, found by size from the filing's directory listing."""
    folder = f"{ARCHIVES}/edgar/data/{cik}/{accession.replace('-', '')}"
    listing = session.get_json(f"{folder}/index.json")

    best_name, best_size = None, 0
    for item in listing["directory"]["item"]:
        name = item["name"]
        if not name.lower().endswith((".htm", ".html")):
            continue
        if any(marker in name.lower() for marker in _EXHIBIT_MARKERS):
            continue
        size = int(item.get("size") or 0)
        if size > best_size:
            best_name, best_size = name, size

    return f"{folder}/{best_name}" if best_name else None


def _select_evenly(rows, limit):
    """Strides through the quarter. Taking the first N would be alphabetical by company."""
    if not limit or len(rows) <= limit:
        return rows
    stride = len(rows) / limit
    return [rows[int(index * stride)] for index in range(limit)]


def _select_by_form(rows, limit, form_shares):
    """Splits the quarter's budget across forms by the shares in config.

    Without this, 8-K filings take about 70% of every quarter because they outnumber the others
    that heavily — and a measured 46 KB of mostly cover-page boilerplate each, against 350 KB of
    narrative in a 10-K.
    """
    chosen = []
    for form, share in form_shares.items():
        for_form = [row for row in rows if row[0] == form.upper()]
        chosen.extend(_select_evenly(for_form, round(limit * share)))
    return chosen


def collect(session, manifest, settings, log):
    destination_root = config.RAW_DIR / SOURCE
    destination_root.mkdir(parents=True, exist_ok=True)
    fetched = 0

    for year, quarter in settings["sec_quarters"]:
        try:
            index_text = session.get_text(FULL_INDEX.format(year=year, quarter=quarter))
        except BlockedByHost:
            raise
        except RuntimeError as error:
            log(f"  {year}Q{quarter}: index unavailable ({error})")
            continue

        rows = _parse_form_index(index_text, settings["sec_forms"])
        chosen = _select_by_form(rows, settings["sec_filings_per_quarter"], settings["sec_forms"])
        log(f"  {year}Q{quarter}: {len(rows)} matching filings, taking {len(chosen)}")

        quarter_dir = destination_root / f"{year}Q{quarter}"
        quarter_dir.mkdir(parents=True, exist_ok=True)
        quarter_fetched = 0

        for form_type, company, cik, date, path in chosen:
            accession = path.rsplit("/", 1)[-1].removesuffix(".txt")
            destination = quarter_dir / f"{cik}_{date}_{form_type.replace('/', '-')}.txt"
            if manifest.already_have(f"{ARCHIVES}/{path}", destination):
                continue

            try:
                document_url = _primary_document_url(session, cik, accession)
                if document_url is None:
                    log(f"    {company}: no primary document in the listing")
                    continue
                text = html_text.extract(session.get_text(document_url))
            except BlockedByHost:
                raise
            except RuntimeError as error:
                log(f"    {company}: {error}")
                continue

            if len(text) < _MINIMUM_TEXT_CHARACTERS:
                log(f"    {company}: only {len(text)} characters extracted, skipped")
                continue
            destination.write_text(text, encoding="utf-8")
            # Recorded against the submission URL, which is the stable identifier for the filing.
            manifest.record(f"{ARCHIVES}/{path}", destination, SOURCE, LICENCE)
            fetched += 1
            quarter_fetched += 1

        log(f"  {year}Q{quarter}: {quarter_fetched} stored")

    return fetched
