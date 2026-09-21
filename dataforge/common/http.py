"""One polite HTTP session for every source.

Rate limiting and robots.txt are enforced here rather than in each source, so a new source
cannot accidentally skip them.
"""

import os
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

CONTACT_VARIABLE = "DATAFORGE_CONTACT"
DEFAULT_REQUESTS_PER_SECOND = 5.0
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0
TIMEOUT_SECONDS = 60

# 403 and 429 mean we are being told to stop. Retrying those earns an IP ban.
FATAL_STATUS_CODES = frozenset({401, 403, 429})


class BlockedByHost(RuntimeError):
    """The host refused us. Stop this source instead of retrying."""


def user_agent():
    contact = os.environ.get(CONTACT_VARIABLE, "").strip()
    if not contact or "@" not in contact:
        raise RuntimeError(
            f"set {CONTACT_VARIABLE} to a real email address before collecting data. "
            "The SEC and Wikimedia both require a contact in the User-Agent and block "
            f"traffic without one.  export {CONTACT_VARIABLE}='you@university.edu'"
        )
    return f"FinanceResearchCorpus/0.1 (academic research; {contact}) python-requests"


class PoliteSession:
    def __init__(self, requests_per_second=DEFAULT_REQUESTS_PER_SECOND, obey_robots=True):
        self.minimum_gap = 1.0 / requests_per_second
        self.obey_robots = obey_robots
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent()
        self._last_request_at = {}
        self._robots = {}

    def _wait_turn(self, host):
        previous = self._last_request_at.get(host)
        if previous is not None:
            remaining = self.minimum_gap - (time.monotonic() - previous)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at[host] = time.monotonic()

    def _fetch_robots(self, scheme, host):
        """robots.txt for one host, or None if it has none we can read.

        RobotFileParser.read() sends its own default User-Agent, which these hosts answer with
        403 — and the parser reads a 403 as "everything is disallowed". So the file is fetched
        with our declared User-Agent and only then handed to the parser.
        """
        self._wait_turn(host)
        try:
            response = self.session.get(
                f"{scheme}://{host}/robots.txt", timeout=TIMEOUT_SECONDS
            )
        except requests.RequestException:
            return None
        if not response.ok:
            # No readable robots.txt is not permission to ignore rate limits, but it is not a
            # refusal either. Proceed at the configured rate.
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser

    def _robots_allow(self, url):
        if not self.obey_robots:
            return True
        parsed = urlparse(url)
        host = parsed.netloc
        if host not in self._robots:
            self._robots[host] = self._fetch_robots(parsed.scheme, host)
        parser = self._robots[host]
        return True if parser is None else parser.can_fetch(self.session.headers["User-Agent"], url)

    def get(self, url, params=None, stream=False):
        if not self._robots_allow(url):
            raise BlockedByHost(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        last_error = None

        for attempt in range(MAX_ATTEMPTS):
            self._wait_turn(host)
            try:
                response = self.session.get(
                    url, params=params, timeout=TIMEOUT_SECONDS, stream=stream
                )
            except requests.RequestException as error:
                last_error = error
            else:
                if response.status_code in FATAL_STATUS_CODES:
                    raise BlockedByHost(
                        f"{response.status_code} from {host} for {url}. "
                        "Slow down or stop; do not retry."
                    )
                if response.ok:
                    return response
                if response.status_code == 404:
                    # Deterministic. Retrying costs three sleeps to learn the same thing, and
                    # sources that probe for what exists hit this on purpose.
                    raise RuntimeError(f"404 for {url}")
                last_error = requests.HTTPError(f"{response.status_code} for {url}")
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_SECONDS * (2**attempt))

        raise RuntimeError(f"gave up on {url} after {MAX_ATTEMPTS} attempts: {last_error}")

    def get_text(self, url, params=None):
        response = self.get(url, params=params)
        # For text/* with no charset, requests assumes ISO-8859-1, so a UTF-8 body decodes to
        # mojibake — including the BOM, which then survives as three characters and breaks the
        # XML parser. The Federal Reserve feeds are served exactly this way.
        if "charset=" not in response.headers.get("content-type", "").lower():
            response.encoding = response.apparent_encoding
        return response.text

    def get_json(self, url, params=None):
        return self.get(url, params=params).json()
