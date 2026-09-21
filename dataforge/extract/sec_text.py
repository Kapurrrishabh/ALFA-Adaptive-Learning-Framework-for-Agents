"""SEC full-submission files to the text of the primary document.

A submission is an SGML wrapper holding several documents: the filing itself, exhibits, XBRL, and
base64 images. Only the first is prose, and pulling the rest in would fill the corpus with
markup and encoded binary.
"""

import re

from . import html_text

_DOCUMENT = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)
_TYPE = re.compile(r"<TYPE>\s*([^\s<]+)", re.IGNORECASE)
_BODY = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)

# Exhibit and data documents. Anything matching this is not the filing narrative.
_SKIP_TYPE = re.compile(r"^(EX-|GRAPHIC|ZIP|EXCEL|JSON|XML|COVER|CORRESP)", re.IGNORECASE)


def extract(submission_text):
    """Returns the primary document's text, or '' if the submission holds no prose document."""
    for block in _DOCUMENT.findall(submission_text):
        type_match = _TYPE.search(block)
        if type_match and _SKIP_TYPE.match(type_match.group(1)):
            continue
        body_match = _BODY.search(block)
        if not body_match:
            continue

        # Since inline XBRL, the primary document is wrapped in <XBRL> — that tag marks the
        # document we want, not one to skip. Standalone XBRL data is excluded by TYPE above.
        body = body_match.group(1)
        return html_text.extract(body) if "<" in body else body

    return ""
