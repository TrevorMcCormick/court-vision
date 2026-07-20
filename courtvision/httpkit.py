"""Small HTTP helpers for the charting app's stdlib server.

Deliberately duplicates the frozen review.py bench's private logic
(json body guard, Range-capable file serving) — the bench is a frozen
cv-18 artifact and must not be edited; this module is the living
copy. Helpers take the BaseHTTPRequestHandler as first argument."""

import json
import re

_RANGE = re.compile(r"bytes=(\d+)-(\d*)")


def send_json(handler, obj, status=200):
    body = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler):
    """Parsed dict body, or None for malformed JSON, non-dict JSON,
    or a non-numeric Content-Length."""
    try:
        n = int(handler.headers.get("Content-Length", 0))
    except (TypeError, ValueError):
        return None
    try:
        body = json.loads(handler.rfile.read(n) or b"{}")
    except json.JSONDecodeError:
        return None
    return body if isinstance(body, dict) else None


def send_file_ranged(handler, path, ctype):
    if not path.exists():
        handler.send_error(404)
        return
    data = path.read_bytes()
    total = len(data)
    m = _RANGE.match(handler.headers.get("Range") or "")
    if m:
        a = int(m.group(1))
        if a >= total:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{total}")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        b = int(m.group(2)) if m.group(2) else total - 1
        b = min(b, total - 1)
        chunk = data[a:b + 1]
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {a}-{b}/{total}")
    else:
        chunk = data
        handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(len(chunk)))
    handler.end_headers()
    handler.wfile.write(chunk)
