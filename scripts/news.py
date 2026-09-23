#!/usr/bin/env python3
"""Reads the collected headlines, scores them, and checks the score against what the price did next.

C4's gate is provenance: a scored article has to trace to its source URL and licence. That is
enforced in the type rather than tested here -- `feed.read` refuses a file the manifest does not
describe -- and this prints one scored article in full so the trace can be read by eye.

The second half is the part worth arguing about. A sentiment score that nothing checks is a number
with an opinion in it. So every article with a date, a tagged instrument and a next bar is scored,
and the sign of the score is compared with the sign of the next day's return. Close to close on the
bar after publication, so the day the news broke is never part of the return being predicted.

Read the result as a headline test, not a trading signal: the feeds hold about a month of history,
so the sample is small and one week of market direction moves it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.news import as_of, feed, sentiment  # noqa: E402
from selfagent.data import prices  # noqa: E402

# Bars are daily, so the earliest return that cannot contain the news itself is the next one.
HORIZON = 1


def scored(articles):
    """(article, polarity) for every article, in the order read."""
    return [(article, sentiment.polarity(f"{article.title} {article.summary}"))
            for article in articles]


def outcomes(rows, price_dir, same_day=False):
    """(polarity, return) for every scored article that has a date, a ticker and a usable bar pair.

    An article is joined to the last bar on or before it was published, which is the last price a
    reader could have acted at, and scored against the move from there to the following close.

    With `same_day` the join moves back one bar, so the return being scored is the publication day's
    own move. That is not a forecast and cannot be traded; it separates the two ways this can fail. A
    scorer that disagrees with the move the news already caused is reading the headline wrong. One
    that agrees with it and still cannot call tomorrow has met the harder half of the problem.
    """
    bars_by_ticker, joined, no_date, no_bar = {}, [], 0, 0
    for article, polarity in rows:
        when = as_of(article)
        if when is None:
            no_date += 1
            continue
        for ticker in sorted(article.tickers):
            if ticker not in bars_by_ticker:
                path = Path(price_dir) / f"{ticker.lower()}.csv"
                bars_by_ticker[ticker] = prices.load_bars(path) if path.exists() else None
            if bars_by_ticker[ticker] is None:
                continue
            dates, bars = bars_by_ticker[ticker]
            try:
                end = prices.index_on_or_before(dates, when) - int(same_day)
            except ValueError:
                # Published before this instrument's first bar, so there is no price to join it to.
                no_bar += 1
                continue
            # A headline newer than the last bar has no future to be judged on.
            if end < 0 or end + HORIZON >= len(dates):
                no_bar += 1
                continue
            joined.append((polarity, prices.label_at(bars, end, HORIZON)))
    return joined, no_date, no_bar


def report(joined, moved):
    """Prints how often the sign of the score matched the sign of the return it is scored against."""
    spoken = [(p, r) for p, r in joined if p != 0.0]
    print(f"\n{len(joined)} article-instrument pairs with a usable bar pair, "
          f"{len(spoken)} of them scored non-zero")
    if not spoken:
        print("  nothing to measure: the lexicon found no polarity word in any headline")
        return

    polarity = np.array([p for p, _ in spoken])
    forward = np.array([r for _, r in spoken])
    agreed = np.sign(polarity) == np.sign(forward)
    # The base rate is the score to beat: guessing "up" every time already scores this much.
    base = float((forward > 0).mean())
    rate = float(agreed.mean())
    error = float(np.sqrt(rate * (1 - rate) / len(agreed)))
    print(f"  sign agreement {rate:.1%} +- {error:.1%}, against {max(base, 1 - base):.1%} "
          f"for always saying the same thing")
    for name, picked in (("positive", polarity > 0), ("negative", polarity < 0)):
        if picked.any():
            print(f"  mean {moved} log return on a {name} headline: "
                  f"{forward[picked].mean():+.4f} over {int(picked.sum())}")
    gap = forward[polarity > 0].mean() - forward[polarity < 0].mean() \
        if (polarity > 0).any() and (polarity < 0).any() else float("nan")
    spread = float(np.sqrt(forward[polarity > 0].var(ddof=1) / max((polarity > 0).sum(), 2)
                           + forward[polarity < 0].var(ddof=1) / max((polarity < 0).sum(), 2)))
    print(f"  positive minus negative: {gap:+.4f} ({abs(gap) / spread:.1f} standard errors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", default="data/raw/news_rss")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--symbols", default="data/raw/sec_edgar/company_tickers.json")
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--show", type=int, default=3, help="scored articles to print in full")
    args = parser.parse_args()

    tickers = sorted(path.stem.upper() for path in Path(args.prices).glob("*.csv"))
    names = feed.company_names(args.symbols, tickers)
    articles = feed.read(args.news, args.manifest, names)
    rows = scored(articles)
    tagged = [row for row in rows if row[0].tickers]
    print(f"{len(articles)} articles after dedupe from {len(feed.provenance(args.manifest))} feeds, "
          f"{len(names)} of {len(tickers)} instruments nameable, {len(tagged)} articles tagged")
    print(f"scorer {sentiment.VERSION}")

    for article, polarity in rows[:args.show]:
        print(f"\n  {article.title[:96]}")
        print(f"    polarity {polarity:+.2f} {sentiment.matches(article.title)}"
              f"  tickers {sorted(article.tickers) or '-'}  as of {as_of(article)}")
        print(f"    from {article.feed}\n    licence: {article.licence}")

    for same_day, moved in ((False, "next-day"), (True, "same-day")):
        joined, no_date, no_bar = outcomes(tagged, args.prices, same_day)
        print(f"\nscored against the {'publication day' if same_day else 'day after publication'}: "
              f"{no_date} articles had no readable date, {no_bar} pairs had no usable bar")
        report(joined, moved)


if __name__ == "__main__":
    sys.exit(main())
