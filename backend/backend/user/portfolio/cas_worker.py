"""Reads one CAS PDF with casparser. Only cas.read runs it, in a process of its own under `python -I`; it imports
nothing but the standard library and casparser.

It takes {"pdf": base64, "password": text} on stdin and writes one JSON reply to stdout: {"statement": ...} or
{"error": ...}.
"""
import base64
import io
import json
import sys

# provisional: this caps address space, which runs well above use; importing casparser took a Mac from 14 to 44 MB
# of use. Confirm it with real statements on the Linux server.
MAX_MEMORY_BYTES = 512 * 2**20
# casparser's JSON keys that may leave this process, nested as it nests them; None keeps the value. The rest (PAN,
# names, address, folio and account numbers, nominees, the text of each transaction) stays here.
KEPT = {
    "file_type": None, "cas_type": None, "statement_period": {"from": None, "to": None},
    "folios": {"schemes": {"scheme": None, "isin": None, "amfi": None, "open": None, "close": None,
                           "transactions": {"date": None, "type": None, "units": None, "nav": None, "amount": None}}},
    "accounts": {"equities": {"isin": None, "name": None, "num_shares": None, "value": None},
                 "mutual_funds": {"isin": None, "name": None, "amfi": None, "balance": None, "nav": None,
                                  "avg_cost": None},
                 "bonds": {"isin": None, "name": None, "num_bonds": None, "value": None}},
    "nps": {"schemes": {"scheme": None, "units": None, "nav": None, "value": None}},
}


def kept(value, keys):
    if keys is None or value is None:
        return value
    if isinstance(value, list):
        return [kept(item, keys) for item in value]
    return {key: kept(value[key], inner) for key, inner in keys.items() if key in value}


def reply(request):
    try:
        from casparser import read_cas_pdf
        from casparser.exceptions import CASParseError, IncorrectPasswordError
    except ModuleNotFoundError as exc:
        if exc.name != "casparser":
            raise
        return {"error": "missing"}
    try:
        statement = json.loads(read_cas_pdf(io.BytesIO(base64.b64decode(request["pdf"])), request["password"],
                                            output="json"))
        # casparser's warnings can quote the statement, so only their number leaves
        return {"statement": {**kept(statement, KEPT), "parse_warning_count": len(statement["parse_warnings"])}}
    except IncorrectPasswordError:
        return {"error": "password"}
    except CASParseError as exc:        # casparser's own fixed sentences, such as "Could not identify the CAS issuer"
        return {"error": "unreadable", "detail": str(exc)}
    except Exception as exc:            # any other message can quote the statement, so only its type leaves
        return {"error": "failed", "detail": type(exc).__name__}


if __name__ == "__main__":
    if sys.platform == "linux":         # macOS does not enforce RLIMIT_AS
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    json.dump(reply(json.loads(sys.stdin.buffer.read())), sys.stdout)
