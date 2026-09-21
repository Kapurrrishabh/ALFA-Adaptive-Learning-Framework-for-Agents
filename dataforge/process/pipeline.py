"""raw -> text -> corpus.

Extraction is cached in data/text/ because it is the slow half; cleaning and deduplication rerun
from that cache in seconds, so the filter thresholds can be tuned without refetching or
re-parsing anything.
"""

import json
from pathlib import Path

from .. import config
from ..extract import html_text, pdf_text, sec_text
from . import clean, dedup

# Deduplication is per source. Cross-source repetition is rare and real (a filing quoted in news
# is a different document), while within-source repetition is the boilerplate problem.
_MINIMUM_DOCUMENT_CHARACTERS = 500

# Publishers write headlines; there is no boilerplate to strip and the prose filter would drop
# them for being short. These sources get whitespace normalisation only.
_ALREADY_PROSE = frozenset({"news_rss"})


def _extract(path):
    """Text of one raw file, chosen by how the source stored it."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_text.extract(path)
    if suffix == ".jsonl":
        # RSS: each line is one article. Headline and summary become one short document.
        parts = []
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    article = json.loads(line)
                    body = html_text.extract(article["summary"]) if article["summary"] else ""
                    parts.append(f"{article['title']}. {body}".strip())
        return "\n\n".join(parts)

    raw = path.read_text(encoding="utf-8", errors="replace")
    if "<SEC-DOCUMENT>" in raw[:5000] or "<DOCUMENT>" in raw[:5000]:
        return sec_text.extract(raw)
    if "<html" in raw[:5000].lower():
        return html_text.extract(raw)
    return raw


def _cached_text(entry, log):
    """Extracted text for one manifest entry, extracting only on a cache miss."""
    raw_path = Path(entry["path"])
    if not raw_path.exists():
        return None

    cache_path = config.TEXT_DIR / entry["source"] / (entry["sha256"][:16] + ".txt")
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    try:
        text = _extract(raw_path)
    except Exception as error:
        # One malformed PDF or filing must not end a run over tens of thousands of files.
        log(f"    unreadable: {raw_path.name} ({type(error).__name__}: {error})")
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def _budget(source, other_characters):
    """Characters a capped source may contribute, given what every other source produced.

    A share of s means budget = other * s / (1 - s), so the source ends up at s of the total.
    """
    share = config.CORPUS_SHARE_CAP.get(source)
    if share is None:
        return None
    return round(other_characters * share / (1 - share))


_BUDGET_SAMPLE = 50


def _average_cleaned_characters(entries, to_prose, log):
    """Cleaned size of a typical document, sampled across the whole list.

    The budget is spent in cleaned characters, and cleaning removes about two thirds of a filing,
    so measuring extracted text left sec_edgar at 24% of the corpus instead of its configured 50%.
    """
    step = max(1, len(entries) // _BUDGET_SAMPLE)
    texts = [_cached_text(entry, log) for entry in entries[::step][:_BUDGET_SAMPLE]]
    sizes = [len(to_prose(text)) for text in texts if text]
    return max(1, sum(sizes) // len(sizes)) if sizes else 1


def _stride(entries, budget, characters_per_entry):
    """Thins entries to fit the budget, spread across the whole list rather than truncated.

    Taking the first N would take only the oldest filings, and the point of a decade of them is
    that they span a decade.
    """
    affordable = int(budget / characters_per_entry)
    if affordable >= len(entries):
        return entries
    step = len(entries) / affordable
    return [entries[int(index * step)] for index in range(affordable)]


def run(manifest, log):
    """Rebuilds data/corpus/ from data/raw/. Safe to rerun; output is replaced, not appended."""
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # Capped sources go last so their budget can be derived from what the others actually yielded.
    sources = sorted(manifest.sources(), key=lambda name: name in config.CORPUS_SHARE_CAP)
    uncapped_characters = 0

    for source in sources:
        entries = manifest.entries_for(source)
        if source == "yahoo_prices":
            continue

        to_prose = clean.normalize if source in _ALREADY_PROSE else clean.clean

        budget = _budget(source, uncapped_characters)
        if budget is not None:
            before = len(entries)
            entries = _stride(entries, budget, _average_cleaned_characters(entries, to_prose, log))
            log(
                f"  {source}: capped to {config.CORPUS_SHARE_CAP[source]:.0%} of the corpus — "
                f"{len(entries)} of {before} documents ({budget // 1_000_000} MB budget)"
            )

        paragraph_filter = dedup.ParagraphFilter()
        near_duplicates = dedup.NearDuplicateIndex()
        written, dropped_short, characters = 0, 0, 0

        with (config.CORPUS_DIR / f"{source}.txt").open("w", encoding="utf-8") as output:
            for index, entry in enumerate(entries, start=1):
                text = _cached_text(entry, log)
                if text is None:
                    continue

                cleaned = to_prose(text)
                if len(cleaned) < _MINIMUM_DOCUMENT_CHARACTERS:
                    dropped_short += 1
                    continue
                if not near_duplicates.add_if_new(cleaned):
                    continue
                cleaned = paragraph_filter.keep_new(cleaned)
                if len(cleaned) < _MINIMUM_DOCUMENT_CHARACTERS:
                    dropped_short += 1
                    continue

                output.write(cleaned + "\n\n")
                written += 1
                characters += len(cleaned)
                if index % 500 == 0:
                    log(f"    {source}: {index}/{len(entries)} read, {written} kept")

        if budget is None:
            uncapped_characters += characters

        log(
            f"  {source}: kept {written} of {len(entries)} documents "
            f"({near_duplicates.dropped} near-duplicate, {dropped_short} too short, "
            f"{paragraph_filter.dropped} repeated paragraphs removed)"
        )
