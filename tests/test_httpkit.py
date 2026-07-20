"""httpkit serves the chart app's HTTP needs; review.py stays frozen
with its private copies. A stub handler exercises the helpers over a
real socketless interface (BytesIO wfile)."""

import io
import json

from courtvision import httpkit


class Stub:
    def __init__(self, headers=None, body=b""):
        self.headers = headers or {}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.status = None
        self.hdrs = {}

    def send_response(self, code):
        self.status = code

    def send_error(self, code, msg=""):
        self.status = code

    def send_header(self, k, v):
        self.hdrs[k] = v

    def end_headers(self):
        pass


def test_send_json_roundtrip():
    h = Stub()
    httpkit.send_json(h, {"ok": True})
    assert h.status == 200
    assert json.loads(h.wfile.getvalue()) == {"ok": True}
    assert h.hdrs["Content-Type"] == "application/json"


def test_read_json_guards_malformed_and_non_dict():
    good = Stub({"Content-Length": "13"}, b'{"a": 1}     ')
    assert httpkit.read_json(good) == {"a": 1}
    bad = Stub({"Content-Length": "8"}, b"{not js}")
    assert httpkit.read_json(bad) is None
    arr = Stub({"Content-Length": "5"}, b"[1,2]")
    assert httpkit.read_json(arr) is None
    badlen = Stub({"Content-Length": "abc"}, b"{}")
    assert httpkit.read_json(badlen) is None


def test_send_file_ranged(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(bytes(range(256)) * 4)          # 1024 bytes
    full = Stub()
    httpkit.send_file_ranged(full, p, "video/mp4")
    assert full.status == 200 and len(full.wfile.getvalue()) == 1024

    part = Stub({"Range": "bytes=10-19"})
    httpkit.send_file_ranged(part, p, "video/mp4")
    assert part.status == 206
    assert part.hdrs["Content-Range"] == "bytes 10-19/1024"
    assert part.wfile.getvalue() == bytes(range(256))[10:20]

    past = Stub({"Range": "bytes=5000-"})
    httpkit.send_file_ranged(past, p, "video/mp4")
    assert past.status == 416
    assert past.hdrs["Content-Range"] == "bytes */1024"

    gone = Stub()
    httpkit.send_file_ranged(gone, tmp_path / "nope.mp4", "video/mp4")
    assert gone.status == 404
