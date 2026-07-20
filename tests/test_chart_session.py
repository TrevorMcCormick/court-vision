"""ChartSession stores raw inputs; every score column is replayed.
The edit test is the point: changing an early winner must change the
replayed context of every later point with no stored-state cleanup."""

import json

import pytest

import courtvision.chartapp as chartapp
from courtvision.chartapp import ChartSession

SETUP = {"player1": "A", "player2": "B", "best_of": 3,
         "final_set": "tb7", "first_server": 1, "video": "m.mp4"}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(chartapp, "CHARTING_ROOT", tmp_path)
    return tmp_path


def test_create_resume_and_setup_guard(root):
    s = ChartSession("m1", SETUP)
    assert (root / "m1" / "manifest.json").exists()
    s2 = ChartSession("m1")                      # resume
    assert s2.setup["player1"] == "A"
    with pytest.raises(ValueError):
        ChartSession("m2")                       # create needs setup


def test_add_point_derives_winner_and_scores(root):
    s = ChartSession("m1", SETUP)
    s.add_point("6*", "")                        # ace, server (P1)
    s.add_point("4b2f1*", "")                    # 3 shots * -> server
    st = s.state()
    assert st["points"][0]["Pts"] == "0-0"
    assert st["points"][1]["Pts"] == "15-0"
    assert st["points"][1]["PtWinner"] == "1"
    assert st["next_server"] == 1                # still game 1


def test_unknown_ending_requires_winner(root):
    s = ChartSession("m1", SETUP)
    with pytest.raises(ValueError):
        s.add_point("4b2f1?", "")
    s.add_point("4b2f1?", "", winner=2)
    assert s.state()["points"][0]["PtWinner"] == "2"


def test_unseen_point_and_export_flags(root):
    s = ChartSession("m1", SETUP)
    s.add_point("6*", "")
    s.insert_unseen(winner=2)
    rows = s.export_rows()
    assert rows[1]["1st"] == "" and rows[1]["PtWinner"] == "2"
    assert "unseen" in rows[1]["Notes"]


def test_edit_replays_downstream_scores(root):
    s = ChartSession("m1", SETUP)
    s.add_point("6*", "")                        # P1 wins
    s.add_point("6*", "")                        # P1 wins -> 30-0
    assert s.state()["points"][1]["Pts"] == "15-0"
    s.update_point(0, first="4b2n*", second="")  # now P2 won pt 1
    st = s.state()
    assert st["points"][0]["PtWinner"] == "2"
    assert st["points"][1]["Pts"] == "0-15"      # downstream moved


def test_segments_and_events(root):
    s = ChartSession("m1", SETUP)
    s.add_point("6*", "", start_s=3.5, end_s=9.1)
    s.add_point("6*", "")                        # no stamps
    segs = s.segments()
    assert segs == [{"clip": "m1_point_001",
                     "start_s": 3.5, "end_s": 9.1}]
    s.add_point("6*", "", start_s=0, end_s=0)      # degenerate stamp
    s.add_point("6*", "", start_s=9.0, end_s=3.0)  # inverted stamp
    assert s.segments() == segs                    # both filtered out
    s.append_event({"ts_ms": 1, "row": "m1_point_001",
                    "event": "palette_key", "payload": {"k": "6"}})
    line = json.loads(
        (root / "m1" / "events.jsonl").read_text().splitlines()[0])
    assert line["mode"] == "chart" and "server_ts_ms" in line


def test_resume_preserves_points(root):
    s = ChartSession("m1", SETUP)
    s.add_point("6*", "")
    s2 = ChartSession("m1")
    assert len(s2.state()["points"]) == 1
