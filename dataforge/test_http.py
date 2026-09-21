"""Pins how fetched bytes become text. Deleted along with the rest of dataforge/.

Separate from test_extract.py because the bug this pins is in decoding, not extraction: two of six
news feeds were silently lost for a whole collection run, and the extractors never saw them.
"""

import pytest
import requests

from dataforge.common.http import CONTACT_VARIABLE, PoliteSession


@pytest.fixture(autouse=True)
def _contact(monkeypatch):
    monkeypatch.setenv(CONTACT_VARIABLE, "tests@example.test")

# What the Federal Reserve feeds actually return: UTF-8 with a BOM, and no charset in the header.
_FEED_BYTES = '﻿<?xml version="1.0" encoding="utf-8" ?><rss><channel>café</channel></rss>'.encode()


def _response(body, content_type):
    response = requests.Response()
    response.status_code = 200
    response._content = body
    response.headers["content-type"] = content_type
    # requests fills this in from the headers on a real response, and that is what get_text reacts
    # to: ISO-8859-1 for text/* with no charset.
    response.encoding = requests.utils.get_encoding_from_headers(response.headers)
    return response


def test_utf8_without_a_declared_charset_is_not_read_as_latin1(monkeypatch):
    session = PoliteSession()
    monkeypatch.setattr(session, "get", lambda url, params=None: _response(_FEED_BYTES, "text/xml"))
    text = session.get_text("https://example.test/feed.xml")
    assert "café" in text
    # Decoded as ISO-8859-1 the BOM becomes three characters that the XML parser rejects.
    assert not text.startswith("ï")


def test_a_declared_charset_is_left_alone(monkeypatch):
    session = PoliteSession()
    body = "café".encode("iso-8859-1")
    monkeypatch.setattr(
        session, "get", lambda url, params=None: _response(body, "text/html; charset=iso-8859-1")
    )
    assert session.get_text("https://example.test/page.html") == "café"
