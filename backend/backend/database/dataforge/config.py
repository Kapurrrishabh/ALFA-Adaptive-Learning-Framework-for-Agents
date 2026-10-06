"""Where data lands, and how much of it to fetch per tier."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

RAW_DIR = DATA_ROOT / "raw"
TEXT_DIR = DATA_ROOT / "text"
CORPUS_DIR = DATA_ROOT / "corpus"
PRICES_DIR = DATA_ROOT / "prices"
# Question-and-answer structure, kept apart from RAW_DIR because the processing pipeline turns
# everything under there into corpus text and these records would land in it a second time.
QA_DIR = DATA_ROOT / "qa"
MANIFEST_PATH = DATA_ROOT / "manifest.jsonl"

# The Board publishes its whole press-release index as one JSON file, so the archive needs no
# crawling of year-by-year listing pages.
FED_PRESS_BASE = "https://www.federalreserve.gov"
FED_PRESS_INDEX = FED_PRESS_BASE + "/json/ne-press.json"

# Rough conversion for reporting only. Phase 3 replaces this with measured subword fertility.
ESTIMATED_CHARS_PER_TOKEN = 4.0

# Large, liquid names across sectors, so price and filing data cover more than mega-cap tech.
TICKERS = (
    "AAPL MSFT NVDA GOOGL AMZN META TSLA BRK-B JPM V UNH XOM JNJ WMT PG MA HD CVX ABBV KO "
    "PEP COST MRK BAC AVGO PFE TMO CSCO ADBE ACN MCD ABT CRM LIN NFLX DHR VZ TXN NKE WFC "
    "PM DIS INTC AMD CAT GS IBM BA MMM GE"
).split()

# Nifty 50 constituents plus the two headline indices. Yahoo suffixes NSE symbols with .NS, and
# carries them back to 1996, so these have as much history as the US names.
INDIA_TICKERS = (
    "RELIANCE.NS TCS.NS HDFCBANK.NS ICICIBANK.NS INFY.NS HINDUNILVR.NS ITC.NS SBIN.NS "
    "BHARTIARTL.NS BAJFINANCE.NS KOTAKBANK.NS LT.NS AXISBANK.NS ASIANPAINT.NS MARUTI.NS "
    "TITAN.NS SUNPHARMA.NS ULTRACEMCO.NS NESTLEIND.NS WIPRO.NS ONGC.NS NTPC.NS POWERGRID.NS "
    "TATAMOTORS.NS TATASTEEL.NS ADANIENT.NS ADANIPORTS.NS COALINDIA.NS HCLTECH.NS TECHM.NS "
    "GRASIM.NS JSWSTEEL.NS INDUSINDBK.NS BAJAJFINSV.NS DRREDDY.NS CIPLA.NS DIVISLAB.NS "
    "EICHERMOT.NS HEROMOTOCO.NS BRITANNIA.NS BPCL.NS SHREECEM.NS HINDALCO.NS SBILIFE.NS "
    "HDFCLIFE.NS APOLLOHOSP.NS TATACONSUM.NS BAJAJ-AUTO.NS UPL.NS M&M.NS ^NSEI ^BSESN"
).split()

# Requests per second per source. SEC publishes a 10/s ceiling; Yahoo and Wikimedia publish none
# and start returning 429 well below that, so they are throttled hard on purpose.
# rbi.org.in answers /robots.txt with 418, so there is no directive to read; it is throttled
# hardest of all on that basis. SEBI publishes an ordinary robots.txt that allows everything.
REQUESTS_PER_SECOND = {
    "sec_edgar": 5.0,
    "openstax": 2.0,
    "yahoo_prices": 0.5,
    "news_rss": 1.0,
    # federalreserve.gov answers /robots.txt with 404, so there is no directive to read and the
    # archive is walked at one page a second rather than faster.
    "fed_press": 1.0,
    "wikipedia": 1.0,
    "sebi": 1.0,
    "rbi": 0.5,
    "ncert": 1.0,
    "gutenberg": 1.0,
    "stackexchange": 0.5,
}
DEFAULT_REQUESTS_PER_SECOND = 1.0

# Press releases and market news. Regulator feeds are the most useful: formal finance language,
# and no licence ambiguity.
# Per-instrument feeds, one request per symbol. The topic feeds above are macro and regulator news:
# over 147 of their headlines, name matching tags only 4 to an instrument the agent holds prices for,
# so nothing scored from them can be checked against what the price did next. These carry the symbol
# in the URL, which makes the tag the publisher's own rather than a guess.
TICKER_FEED = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"

RSS_FEEDS = (
    "https://www.sec.gov/news/pressreleases.rss",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.federalreserve.gov/feeds/speeches.xml",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "https://www.investing.com/rss/news.rss",
)

# Roots for the Wikipedia crawl. Subcategories are followed to a bounded depth. The India roots
# carry the vocabulary no US source uses at all: demat, SEBI, GST, repo corridor, Nifty.
WIKIPEDIA_CATEGORIES = (
    "Category:Finance",
    "Category:Financial markets",
    "Category:Corporate finance",
    "Category:Investment",
    "Category:Banking",
    "Category:Macroeconomics",
    "Category:Financial risk",
    "Category:Accounting",
    "Category:Monetary policy",
    "Category:Financial ratios",
    "Category:Economy of India",
    "Category:Banking in India",
    "Category:Financial services companies of India",
    "Category:Stock exchanges in India",
    "Category:Taxation in India",
    "Category:Companies listed on the National Stock Exchange of India",
    "Category:Economic history of India",
    "Category:Government finances in India",
)
WIKIPEDIA_CATEGORY_DEPTH = 2

# SEBI's listing pages page in 25s through an ordinary GET parameter. Circular bodies are PDFs
# linked from an iframe, not from the page text, so the collector reads the iframe.
SEBI_SECTIONS = {"circulars": 7, "regulations": 3, "acts": 1}
SEBI_MAX_LISTING_PAGES = 120

# RBI's RSS window is only 10 items per feed, so re-running grows this source. The bulletin index
# is the one archive page that answers a plain GET, and its articles are the long-form ones.
RBI_FEEDS = (
    "https://rbi.org.in/speeches_rss.xml",
    "https://rbi.org.in/pressreleases_rss.xml",
    "https://rbi.org.in/notifications_rss.xml",
    "https://rbi.org.in/Publication_rss.xml",
    "https://rbi.org.in/AnnualReportMain_rss.xml",
)
RBI_BULLETIN_INDEX = "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx"

# NCERT Classes 11-12. These are how finance is actually taught in India, in teaching language:
# k = class 11, l = class 12; ac accountancy, bs business studies, ec economics, st statistics.
NCERT_BOOK_CODES = (
    "keac1", "keac2", "leac1", "leac2",
    "kebs1", "lebs1", "lebs2",
    "keec1", "leec1", "leec2", "kest1",
)
NCERT_MAX_CHAPTERS = 15

# Public-domain economics, the theory the textbooks are built on. Gutenberg ids, chosen rather
# than searched because /ebooks/search is the one path its robots.txt disallows.
GUTENBERG_BOOK_IDS = (
    3300,   # Smith, The Wealth of Nations
    1232,   # Machiavelli, The Prince
    33310,  # Marx, Capital Vol 1
    4341,   # Malthus, Principle of Population
    15776,  # Bagehot, Lombard Street
    1653,   # Veblen, Theory of the Leisure Class
    34081,  # Mill, Principles of Political Economy
    16960,  # Ricardo, Principles of Political Economy and Taxation
    38194,  # Fisher, The Purchasing Power of Money
    14268,  # Hazlitt-era public finance
    26217,  # Keynes, The Economic Consequences of the Peace
    23755,  # Mitchell, Business Cycles
)

# Stack Exchange publishes full dumps on archive.org under CC BY-SA. This is the only informal
# register in the corpus: real people asking real money questions, and being answered.
STACKEXCHANGE_SITES = (
    "money.stackexchange.com",
    "quant.stackexchange.com",
    "economics.stackexchange.com",
)
STACKEXCHANGE_DUMP = "https://archive.org/download/stackexchange/{site}.7z"
# A thread shorter than this is a one-line question with no useful answer.
STACKEXCHANGE_MINIMUM_CHARACTERS = 400

# Caps one source's share of the finished corpus. US filings otherwise supply 95% of all tokens
# and drown both the Indian and the conversational text. A share rather than a character count so
# it stays correct as sources are added: the budget is derived from what everything else yielded.
CORPUS_SHARE_CAP = {"sec_edgar": 0.5}

# Tiers control how much is fetched. Sizing comes from 4,000 filings actually collected: extracted
# text averages 350 KB for a 10-K, 165 KB for a 10-Q, 46 KB for an 8-K. At the shares below that is
# roughly 54k tokens per filing, so "standard" is about 100M nominal tokens and "bulk" about 540M.
# Both clear the 100M BabyLM track, which is already more than a laptop-scale model can use.
TIERS = {
    "smoke": {
        "sec_quarters": [(2024, 1)],
        "sec_filings_per_quarter": 5,
        "sec_forms": {"10-K": 1.0},
        "price_range": "2y",
        "wikipedia_articles": 20,
        "openstax_books": 1,
        "sebi_circulars": 5,
        "rbi_documents": 5,
        "gutenberg_books": 1,
        "stackexchange_sites": 1,
        "fed_releases": 5,
    },
    "standard": {
        "sec_quarters": [(year, quarter) for year in (2022, 2023, 2024) for quarter in (1, 2, 3, 4)],
        "sec_filings_per_quarter": 150,
        "sec_forms": {"10-K": 0.5, "10-Q": 0.5},
        "price_range": "10y",
        "wikipedia_articles": 1500,
        "openstax_books": None,
        "sebi_circulars": 500,
        "rbi_documents": 100,
        "gutenberg_books": 6,
        "stackexchange_sites": 2,
        "fed_releases": 300,
    },
    "bulk": {
        "sec_quarters": [
            (year, quarter) for year in range(2015, 2025) for quarter in (1, 2, 3, 4)
        ],
        "sec_filings_per_quarter": 250,
        # 8-K keeps a small share for the earnings releases and material-event announcements that
        # the annual and quarterly reports do not carry.
        "sec_forms": {"10-K": 0.4, "10-Q": 0.4, "8-K": 0.2},
        "price_range": "max",
        "wikipedia_articles": 12000,
        "openstax_books": None,
        "sebi_circulars": None,
        "rbi_documents": None,
        "gutenberg_books": None,
        "stackexchange_sites": None,
        "fed_releases": None,
    },
}


def tier(name):
    if name not in TIERS:
        raise ValueError(f"unknown tier '{name}'. choose from {sorted(TIERS)}")
    return TIERS[name]


def ensure_directories():
    for directory in (RAW_DIR, TEXT_DIR, CORPUS_DIR, PRICES_DIR, QA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
