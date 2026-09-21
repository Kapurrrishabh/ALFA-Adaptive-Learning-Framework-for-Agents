"""What we actually collected, in the units that decide whether it is enough.

Token counts here are estimates from character counts, not tokenizer output. They are for deciding
whether to keep collecting; the real count comes from the tokenizer in Phase 3.
"""

import csv

from .. import config


def summarize(manifest):
    """Returns (rows, totals) where a row is one source's raw and corpus footprint."""
    rows = []
    for source in manifest.sources():
        entries = manifest.entries_for(source)
        corpus_file = config.CORPUS_DIR / f"{source}.txt"
        corpus_bytes = corpus_file.stat().st_size if corpus_file.exists() else 0
        rows.append(
            {
                "source": source,
                "files": len(entries),
                "raw_mb": sum(entry["bytes"] for entry in entries) / 1e6,
                "corpus_mb": corpus_bytes / 1e6,
                "tokens": corpus_bytes / config.ESTIMATED_CHARS_PER_TOKEN,
            }
        )

    totals = {
        "files": sum(row["files"] for row in rows),
        "raw_mb": sum(row["raw_mb"] for row in rows),
        "corpus_mb": sum(row["corpus_mb"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
    }
    return rows, totals


def _price_summary():
    files = sorted(config.PRICES_DIR.glob("*.csv"))
    total_days = 0
    for path in files:
        with path.open() as handle:
            total_days += max(0, sum(1 for _ in csv.reader(handle)) - 1)
    return len(files), total_days


def render(manifest):
    rows, totals = summarize(manifest)
    lines = [
        "",
        f"{'source':<16}{'files':>9}{'raw MB':>10}{'corpus MB':>12}{'est. tokens':>14}",
        "-" * 61,
    ]
    for row in rows:
        lines.append(
            f"{row['source']:<16}{row['files']:>9,}{row['raw_mb']:>10,.1f}"
            f"{row['corpus_mb']:>12,.1f}{row['tokens']:>14,.0f}"
        )
    lines.append("-" * 61)
    lines.append(
        f"{'total':<16}{totals['files']:>9,}{totals['raw_mb']:>10,.1f}"
        f"{totals['corpus_mb']:>12,.1f}{totals['tokens']:>14,.0f}"
    )

    tickers, days = _price_summary()
    lines.append("")
    lines.append(f"prices: {tickers} tickers, {days:,} ticker-days")
    lines.append(
        f"text: ~{totals['tokens'] / 1e6:.1f}M estimated tokens "
        f"(BabyLM tracks are 10M and 100M; BERT-base used 3,300M)"
    )
    return "\n".join(lines)
