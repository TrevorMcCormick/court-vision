import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import courtvision.review as review
from courtvision.review import ReviewSession, make_server


def _client(cfg, mode="review", name="srv"):
    s = ReviewSession(cfg, mode, name, seed="s", n=2)
    httpd = make_server(s, 0)                    # port 0 = ephemeral
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return s, httpd, HTTPConnection("127.0.0.1", httpd.server_port)


def _post(conn, path, body):
    conn.request("POST", path, json.dumps(body),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def test_root_serves_ui(cfg, monkeypatch, tmp_path):
    stub = tmp_path / "ui.html"
    stub.write_text("<!-- cv-review -->")
    monkeypatch.setattr(review, "UI_PATH", stub)
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/")
    r = conn.getresponse()
    assert r.status == 200 and b"cv-review" in r.read()
    httpd.shutdown()


def test_state_endpoint_matches_session(cfg):
    s, httpd, conn = _client(cfg, mode="cold", name="srv-cold")
    conn.request("GET", "/api/state")
    got = json.loads(conn.getresponse().read())
    assert got == s.state()
    assert got["rows"][0]["confidence"] is None      # cold: stripped
    httpd.shutdown()


def test_clip_range_request_returns_206_partial(cfg):
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/clip/tt_point_01.mp4",
                 headers={"Range": "bytes=10-19"})
    r = conn.getresponse()
    body = r.read()
    assert r.status == 206
    assert r.getheader("Content-Range") == "bytes 10-19/1024"
    assert body == bytes(range(256))[10:20]
    httpd.shutdown()


def test_clip_without_range_returns_full_200(cfg):
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/clip/tt_point_01.mp4")
    r = conn.getresponse()
    assert r.status == 200 and len(r.read()) == 1024
    httpd.shutdown()


def test_lint_event_accept_roundtrip(cfg):
    s, httpd, conn = _client(cfg)
    st, got = _post(conn, "/api/lint", {"first": "6b2Z*", "second": ""})
    assert st == 200 and len(got["issues"]) == 1
    st, _ = _post(conn, "/api/event",
                  {"ts_ms": 1, "row": "tt_point_01",
                   "event": "row_open", "payload": {}})
    assert st == 200
    st, _ = _post(conn, "/api/accept",
                  {"clip": "tt_point_01", "corrected_1st": "4b3f1w@",
                   "corrected_2nd": "", "notes": "", "skip_reason": ""})
    assert st == 200
    events = (s.dir / "events.jsonl").read_text()
    assert '"row_open"' in events and '"accept"' in events
    assert "tt_point_01" in (s.dir / "corrected.csv").read_text()
    httpd.shutdown()


def test_malformed_json_returns_400_not_crash(cfg):
    _, httpd, conn = _client(cfg)
    conn.request("POST", "/api/lint", b"{not json",
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    r.read()
    assert r.status == 400
    conn2 = HTTPConnection("127.0.0.1", httpd.server_port)
    conn2.request("POST", "/api/accept", json.dumps({"corrected_1st": "x"}),
                  {"Content-Type": "application/json"})
    r = conn2.getresponse()
    r.read()
    assert r.status == 400                      # missing clip key
    httpd.shutdown()


def test_range_past_eof_returns_416(cfg):
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/clip/tt_point_01.mp4",
                 headers={"Range": "bytes=2000-3000"})
    r = conn.getresponse()
    r.read()
    assert r.status == 416
    assert r.getheader("Content-Range") == "bytes */1024"
    httpd.shutdown()


def test_missing_clip_404(cfg):
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/clip/nope.mp4")
    r = conn.getresponse()
    r.read()
    assert r.status == 404
    httpd.shutdown()
