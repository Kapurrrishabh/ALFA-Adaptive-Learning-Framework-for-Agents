"""CAS statements (CAMS, KFintech, NSDL, CDSL) read with casparser, the optional `cas` extra.

casparser runs native PDF code on files from users, so the server never imports it: cas_worker.py does, in a process
of its own with a time and memory limit.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

from backend.user.portfolio.importers import SHORTER_RANGE, ImportUnsupported

CAS_TIMEOUT_S = 30          # provisional: no real statement has been timed; confirm with a long CAMS one on the server
WORKER = Path(__file__).with_name("cas_worker.py")


def read(data: bytes, password: str) -> Dict[str, object]:
    """The statement as casparser writes it in JSON (its keys, decimals as text), keeping only holdings and
    transactions, plus parse_warning_count. A file casparser cannot read raises ValueError, which the route answers
    with 400. The password goes to the worker on stdin, never on its command line."""
    request = json.dumps({"pdf": base64.b64encode(data).decode(), "password": password}).encode()
    try:
        # no environment: the worker runs native code on a file from a user, so it gets none of the server's secrets
        done = subprocess.run([sys.executable, "-I", str(WORKER)], input=request, capture_output=True,
                              timeout=CAS_TIMEOUT_S, env={})
    except subprocess.TimeoutExpired:
        raise ValueError(f"This statement took more than {CAS_TIMEOUT_S} seconds to read, so we stopped. "
                         f"{SHORTER_RANGE}") from None
    if done.returncode:     # its stderr is left out, as a traceback can quote the statement
        raise RuntimeError(f"The CAS reader stopped with exit code {done.returncode}.")
    reply = json.loads(done.stdout)
    error = reply.get("error")
    if error == "missing":
        raise ImportUnsupported("PDF statements aren't switched on for this server. "
                                "Upload your broker's Excel or CSV file instead.")
    if error == "password":
        raise ValueError("That password didn't open the file. For CAMS or KFintech it is the one you chose; "
                         "for NSDL or CDSL it is usually your PAN in capitals.")
    if error == "unreadable":
        raise ValueError(f"This PDF isn't a CAS statement we can read: {reply['detail']}")
    if error == "failed":
        raise RuntimeError(f"The CAS reader failed with {reply['detail']}.")
    return reply["statement"]
