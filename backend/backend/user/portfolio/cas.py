"""CAS statements (CAMS, KFintech, NSDL, CDSL), read by casparser (the optional `cas` extra) in a process of its own,
as it runs native PDF code on files from users. The route sends a PDF and its password here, not to importers.parse,
so importers never needs to import this module.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

from backend.user.portfolio.importers import PDF_SWITCHED_OFF, SHORTER_RANGE, ImportUnsupported

CAS_TIMEOUT_S = 30          # provisional: no real statement has been timed; confirm with a long CAMS one on the server
WORKER = Path(__file__).with_name("cas_worker.py")


def read(data: bytes, password: str) -> Dict[str, object]:
    """Holdings and transactions as casparser writes them (its keys, decimals as text), plus parse_warning_count. A
    file casparser cannot read raises ValueError, which the route answers with 400."""
    request = json.dumps({"pdf": base64.b64encode(data).decode(), "password": password}).encode()
    try:
        # env={} only keeps the server's secrets out of its environment; it runs as the same user, so it is no sandbox
        done = subprocess.run([sys.executable, "-I", str(WORKER)], input=request, capture_output=True,
                              timeout=CAS_TIMEOUT_S, env={})
    except subprocess.TimeoutExpired:
        raise ValueError(f"This statement took more than {CAS_TIMEOUT_S} seconds to read, so we stopped. "
                         f"{SHORTER_RANGE}") from None
    if done.returncode:     # stderr is left out: the PDF libraries write there, and nothing checks it for the statement
        raise RuntimeError(f"The CAS reader stopped with exit code {done.returncode}.")
    try:
        reply = json.loads(done.stdout)
    except ValueError as exc:   # our fault, not the file's, so not a 400
        raise RuntimeError(f"The CAS reader's reply is not JSON ({exc}). "
                           "Check that nothing in the reader prints to stdout.") from None
    error = reply.get("error")
    if error == "missing":
        raise ImportUnsupported(PDF_SWITCHED_OFF)
    if error == "password":
        raise ValueError("That password didn't open the file. For CAMS or KFintech it is the one you chose; "
                         "for NSDL or CDSL it is usually your PAN in capitals.")
    if error == "unreadable":
        raise ValueError(f"This PDF isn't a CAS statement we can read: {reply['detail']}")
    if error == "failed":
        raise RuntimeError(f"The CAS reader failed with {reply['detail']}.")
    return reply["statement"]
