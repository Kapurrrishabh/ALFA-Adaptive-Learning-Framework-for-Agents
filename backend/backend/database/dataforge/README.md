# dataforge — DELETE THIS FOLDER BEFORE SUBMISSION

Scaffolding that collects the training corpus. It is deliberately separate from `selfagent/`:
nothing in the library imports from here, and deleting this directory breaks nothing except
the ability to re-download data.

What it keeps: `data/raw/` (as fetched), `data/text/` (extracted), `data/corpus/` (cleaned and
deduplicated), `data/prices/` (OHLCV CSV), and `data/manifest.jsonl` (provenance for every file).
Keep `data/corpus/` and `data/prices/`. The rest is reproducible.

## Setup

```bash
python3 -m pip install -r dataforge/requirements.txt
export DATAFORGE_CONTACT="your.email@university.edu"
```

`DATAFORGE_CONTACT` is required, not decorative. The SEC's fair-access rules require a real
contact address in the User-Agent, and Wikimedia's policy requires the same. Scripts refuse to
run without it rather than sending anonymous traffic.

## Run

```bash
python3 dataforge/collect.py --list                      # what is available
python3 dataforge/collect.py --source openstax           # one source
python3 dataforge/collect.py --tier smoke                # ~5 min, proves the pipeline
python3 dataforge/collect.py --tier standard             # ~15M tokens target
python3 dataforge/collect.py --tier bulk                 # hours; run overnight
python3 dataforge/collect.py --process                   # extract, clean, dedup, report
```

Every stage is resumable. A file already present with a recorded hash is not fetched again, so
an interrupted run can be restarted with the same command.

## Being a good citizen

This matters, and not only for politeness — SEC and Wikimedia both block IPs that ignore their
rules, which would cost you the corpus.

- `robots.txt` is fetched and honoured per host before any request.
- One request at a time per host, with a floor on the gap between them. SEC allows 10/second; the
  default here is 5/second.
- Retries use exponential backoff and stop. A 403 or 429 aborts that source loudly instead of
  hammering.
- Only public bulk endpoints, official APIs, and RSS feeds. No login walls, no paywalls, no
  CAPTCHA circumvention.
- Every file's source URL and licence note is recorded in `data/manifest.jsonl`.

## Sources

| Source | Content | Licence note | Volume |
|---|---|---|---|
| `sec_edgar` | 10-K / 10-Q / 8-K filings | US Government work, public domain | Very high — the main lever |
| `openstax` | Finance and economics textbooks (PDF) | CC BY | ~10 MB, high quality prose |
| `yahoo_prices` | Daily OHLCV | Personal/research use per Yahoo terms | Small, dense |
| `news_rss` | Finance news headlines and summaries | Publisher terms; headlines only | Grows over time |
| `wikipedia` | Finance and economics articles | CC BY-SA | ~50 MB |

Yahoo's endpoint is undocumented and unofficial. It works and needs no key, but it can change
without notice — if `yahoo_prices` breaks, that is why, and the fix is a different price source
rather than a bug in this repository.
