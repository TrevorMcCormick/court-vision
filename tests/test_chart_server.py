"""Server-half tests for the chart app. This file starts with the
conformance plumbing; Task 7 appends the route tests."""

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

import courtvision.chartapp as chartapp
from courtvision.chartapp import ChartSession, make_chart_server


def test_conformance_path_points_at_generated_fixture():
    p = chartapp.CONFORMANCE_PATH
    assert p.name == "score_conformance.json"
    assert p.exists() and p.stat().st_size > 10000


SETUP = {"player1": "A", "player2": "B", "best_of": 3,
         "final_set": "tb7", "first_server": 1, "video": "m.mp4"}


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(chartapp, "CHARTING_ROOT", tmp_path)
    video = tmp_path / "m.mp4"
    video.write_bytes(bytes(range(256)) * 4)
    s = ChartSession("mm", SETUP)
    httpd = make_chart_server(s, video, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    conn = HTTPConnection("127.0.0.1", httpd.server_port)
    yield s, conn
    httpd.shutdown()


def _post(conn, path, body):
    conn.request("POST", path, json.dumps(body),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def test_root_injects_chart_mode_and_serves_grammar(served):
    _, conn = served
    conn.request("GET", "/")
    body = conn.getresponse().read().decode()
    assert 'SERVER_MODE="chart"' in body.split("</script>")[0]
    conn.request("GET", "/grammar.json")
    g = json.loads(conn.getresponse().read())
    assert g["version"] == 1
    conn.request("GET", "/conformance.json")
    r = conn.getresponse()
    body = r.read()
    assert r.status == 200 and len(body) > 10000


def test_video_range_and_point_roundtrip(served):
    s, conn = served
    conn.request("GET", "/video", headers={"Range": "bytes=0-9"})
    r = conn.getresponse()
    assert r.status == 206 and len(r.read()) == 10

    st, _ = _post(conn, "/api/point",
                  {"action": "add", "first": "6*", "second": "",
                   "winner": None, "start_s": 1.0, "end_s": 4.2})
    assert st == 200
    st, _ = _post(conn, "/api/point", {"action": "unseen",
                                       "winner": 2})
    assert st == 200
    conn.request("GET", "/api/chart-state")
    state = json.loads(conn.getresponse().read())
    assert len(state["points"]) == 2
    assert state["points"][0]["Pts"] == "0-0"

    st, _ = _post(conn, "/api/point",
                  {"action": "update", "idx": 0,
                   "first": "42n@", "second": ""})
    assert st == 200
    conn.request("GET", "/api/chart-state")
    state = json.loads(conn.getresponse().read())
    assert state["points"][0]["PtWinner"] == "2"


def test_bad_point_returns_400_not_crash(served):
    _, conn = served
    st, _ = _post(conn, "/api/point",
                  {"action": "add", "first": "4b2f1?",
                   "second": "", "winner": None})
    assert st == 400
