#!/usr/bin/env python3
"""Collect the training corpus. See dataforge/README.md.

A source that fails is reported and the run continues, but the exit code is non-zero so a failure
inside an overnight run is visible the next morning instead of looking like success.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataforge import config
from dataforge.common.http import PoliteSession
from dataforge.common.manifest import Manifest
from dataforge.process import pipeline, report
from dataforge.sources import (
    fed_press,
    gutenberg,
    ncert,
    news_rss,
    openstax,
    rbi,
    sebi,
    sec_edgar,
    stackexchange,
    wikipedia,
    yahoo_prices,
)

SOURCES = {
    module.SOURCE: module
    for module in (
        sec_edgar,
        openstax,
        yahoo_prices,
        news_rss,
        fed_press,
        wikipedia,
        sebi,
        rbi,
        ncert,
        gutenberg,
        stackexchange,
    )
}


def log(message):
    print(message, flush=True)


def _run_source(name, manifest, settings):
    module = SOURCES[name]
    rate = config.REQUESTS_PER_SECOND.get(name, config.DEFAULT_REQUESTS_PER_SECOND)
    session = PoliteSession(requests_per_second=rate)
    started = time.monotonic()

    log(f"\n{name} (at {rate}/s)")
    fetched = module.collect(session, manifest, settings, log)
    log(f"  {fetched} new files in {time.monotonic() - started:.0f}s")
    return fetched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show sources and tiers, fetch nothing")
    parser.add_argument("--source", action="append", help="fetch only this source; repeatable")
    parser.add_argument("--tier", choices=sorted(config.TIERS), help="how much to fetch")
    parser.add_argument("--process", action="store_true", help="rebuild data/corpus from data/raw")
    parser.add_argument("--report", action="store_true", help="print what is held and stop")
    arguments = parser.parse_args()

    if arguments.list:
        log("sources: " + ", ".join(sorted(SOURCES)))
        for name, settings in config.TIERS.items():
            quarters = len(settings["sec_quarters"])
            log(
                f"  {name:<9} {quarters} SEC quarters x {settings['sec_filings_per_quarter']} "
                f"filings, {settings['wikipedia_articles']} wiki articles, "
                f"prices {settings['price_range']}"
            )
        return 0

    config.ensure_directories()
    manifest = Manifest(config.MANIFEST_PATH)

    if arguments.report:
        log(report.render(manifest))
        return 0

    failures = []
    if arguments.source or arguments.tier:
        settings = config.tier(arguments.tier or "smoke")
        wanted = arguments.source or sorted(SOURCES)
        unknown = [name for name in wanted if name not in SOURCES]
        if unknown:
            parser.error(f"unknown source(s) {unknown}; choose from {sorted(SOURCES)}")

        for name in wanted:
            try:
                _run_source(name, manifest, settings)
            except Exception as error:
                # Sources are independent. One blocked host should not cost the rest of the run.
                log(f"  FAILED: {type(error).__name__}: {error}")
                failures.append(name)

    if arguments.process:
        log("\nprocessing")
        pipeline.run(manifest, log)

    log(report.render(manifest))
    if failures:
        log(f"\n{len(failures)} source(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
