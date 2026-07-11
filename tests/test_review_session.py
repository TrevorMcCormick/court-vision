import json

from courtvision.review import (load_export_rows, sample_cold_rows,
                                ReviewSession)


def test_load_export_rows_preserves_order(cfg):
    rows = load_export_rows(cfg)
    assert [r["clip"] for r in rows] == [
        "tt_point_01", "tt_point_02", "tt_point_03", "tt_point_04"]


def test_sample_excludes_blank_pt_and_is_deterministic(cfg):
    rows = load_export_rows(cfg)
    picked = sample_cold_rows(rows, 2, "cv18-a")
    assert picked == sample_cold_rows(rows, 2, "cv18-a")
    assert "tt_point_03" not in picked          # blank Pt: ungradeable
    # match order preserved
    idx = {r["clip"]: i for i, r in enumerate(rows)}
    assert picked == sorted(picked, key=idx.get)


def test_create_writes_manifest_and_resume_roundtrips(cfg):
    s = ReviewSession(cfg, "cold", "block-a", seed="cv18-a", n=2)
    man = json.loads((s.dir / "manifest.json").read_text())
    assert man["mode"] == "cold" and man["match"] == "tt"
    assert len(man["rows"]) == 2
    s.accept("tt_point_01", "4b3f1w@", "", "shaky serve read")
    s2 = ReviewSession(cfg, "cold", "block-a", seed="cv18-a", n=2)
    row = next(r for r in s2.state()["rows"]
               if r["clip"] == "tt_point_01")
    assert row["done"] and row["corrected_1st"] == "4b3f1w@"


def test_resume_mode_mismatch_raises(cfg):
    ReviewSession(cfg, "cold", "block-x", seed="s", n=2)
    try:
        ReviewSession(cfg, "review", "block-x")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cold_state_strips_machine_fields(cfg):
    s = ReviewSession(cfg, "cold", "block-b", seed="cv18-a", n=2)
    for r in s.state()["rows"]:
        assert r["first_draft"] == ""
        assert r["confidence"] is None and r["conf_p"] is None
        assert r["serve_s"] is None and r["n_shots"] is None


def test_review_state_keeps_draft_and_all_rows(cfg):
    s = ReviewSession(cfg, "review", "full")
    rows = s.state()["rows"]
    assert len(rows) == 4
    assert rows[0]["first_draft"] == "s4b3f1w@"
    assert rows[0]["confidence"] == "high"


def test_events_append_and_inject_context(cfg):
    s = ReviewSession(cfg, "review", "full")
    s.append_event({"ts_ms": 5, "row": "tt_point_01",
                    "event": "row_open", "payload": {}})
    s.append_event({"ts_ms": 9, "row": "tt_point_01",
                    "event": "accept", "payload": {}})
    lines = [json.loads(l) for l in
             (s.dir / "events.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["row_open", "accept"]
    assert lines[0]["session"] == "full" and lines[0]["mode"] == "review"
    assert "server_ts_ms" in lines[0]


def test_skip_records_flag(cfg):
    s = ReviewSession(cfg, "review", "full")
    s.accept("tt_point_02", "", "", "", skip_reason="broken clip")
    row = next(r for r in s.state()["rows"]
               if r["clip"] == "tt_point_02")
    assert row["skipped"] == "broken clip"
