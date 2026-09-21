# Prompt to run collection in another agent space

Written for: a separate coding agent, running on a different machine or network.

Use this when a source is blocked from one environment but reachable from another, or to fetch
several sources at once. Paste the block below, filling in the two placeholders.

## Coordination rules — read before splitting work

- **One agent per source.** Two agents on the same source means two request streams at the same
  host, at double the rate the source module thinks it is using. That is what gets an IP blocked.
- **Only ever one agent on `sec_edgar`.** The SEC's fair-access limit is per IP, not per process.
- **`data/manifest.jsonl` is the shared ledger.** Each agent loads it once at start, so an agent
  cannot see what another added during the run. Disjoint sources are what keeps this correct.
- **Processing runs once, at the end, by one agent.** `--process` rebuilds `data/corpus/` from the
  whole manifest, so running it while another agent is still fetching produces a partial corpus.
- **Never pass `--tier` without `--source`** in a multi-agent run; that starts every source.

## The prompt

```
You are collecting a finance training corpus for a research project. The collection code already
exists and is tested — do not rewrite it, run it.

Repository: <PATH-OR-GIT-URL>
Your assigned source: <SOURCE>        # one of: sec_edgar, openstax, yahoo_prices, news_rss, wikipedia

Setup:
  python3 -m pip install -r dataforge/requirements.txt
  export DATAFORGE_CONTACT="<a real email address>"

The contact variable is required, not decorative — the SEC and Wikimedia both block traffic whose
User-Agent has no contact in it, and the scripts refuse to start without it.

Run only your assigned source:
  python3 dataforge/collect.py --source <SOURCE> --tier bulk

Rules:
- Do not run any other source. Another agent has it. Two agents on one host doubles the request
  rate and gets the IP blocked.
- Do not run --process. The coordinator does that once, after every source has finished.
- Do not lower the rate limits in dataforge/config.py, and do not set obey_robots=False.
- The run is resumable. If it dies, rerun the same command; files already recorded with a matching
  size on disk are not fetched again.
- If a source fails with BlockedByHost, stop that source and report it. Do not retry, do not
  change the User-Agent, do not route around it.

Report back, in this order:
1. The command you ran and its exit code.
2. The last 30 lines of output, including the summary table.
3. `du -sh data/raw data/corpus` and the line count of `data/manifest.jsonl`.
4. Every failure message verbatim. Do not summarise or soften them — a source that silently
   fetched nothing looks identical to success in the totals.

Then stop. Do not start training, do not touch anything under selfagent/, and do not modify
dataforge/ except to fix a genuine bug in your own source — if you do fix one, say exactly what
was wrong and what you changed.
```

## After every agent reports

On the machine that holds the combined `data/` directory:

```bash
python3 dataforge/collect.py --process     # rebuild data/corpus from data/raw
python3 dataforge/collect.py --report      # what we ended up with
```

Keep `data/corpus/` and `data/prices/`. Everything else is reproducible from the manifest.
