"""import-bundle: synthetic charter bundles -> evaluate-shaped scaffolding.

The fixture is a tiny 3-point session-shape bundle (point 2 is a fault
point played on the second-serve string) built in tmp_path; every
assertion targets the exact shapes evaluate()/align read. The live
ao26f session is deliberately NOT touched here.
"""

import csv
import json
import shutil
import subprocess

import pytest

from courtvision import bundle, config
from courtvision.chartapp import EXPORT_FIELDS, RAW_FIELDS
from courtvision.config import EvalCfg
from courtvision.evaluate import p1_end
from courtvision.mcp import parse_mcp

SETUP = {"player1": "Ann", "player2": "Bea", "best_of": 3,
         "final_set": "tb7", "first_server": 1, "video": "tb.mp4"}

POINTS = [
    {"first": "4f2b3*", "second": "", "notes": "", "winner": "1",
     "start_s": "1.0", "end_s": "4.0", "flags": ""},
    {"first": "6n", "second": "4f1*", "notes": "", "winner": "2",
     "start_s": "6.0", "end_s": "9.5", "flags": ""},        # the fault pt
    {"first": "5x", "second": "4d", "notes": "", "winner": "2",
     "start_s": "11.0", "end_s": "13.0", "flags": ""},      # double fault
]


def make_bundle(tmp_path, points=POINTS, setup=SETUP, name="bundle"):
    b = tmp_path / name
    b.mkdir()
    (b / "manifest.json").write_text(json.dumps(
        {"setup": setup, "created_ts_ms": 0, "grammar_version": 1}))
    with open(b / "points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(points)
    return b


@pytest.fixture
def imported(tmp_path):
    b = make_bundle(tmp_path)
    root = tmp_path / "repo"
    bundle.import_bundle(b, "tb", root=root)
    return root


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_points_csv_is_mcp_shape(imported):
    rows = read(imported / "data" / "mcp" / "tb_points.csv")
    assert list(rows[0]) == EXPORT_FIELDS
    assert [r["Pt"] for r in rows] == ["1", "2", "3"]
    # score state is REPLAYED, server-first, state-before-the-point
    assert [r["Pts"] for r in rows] == ["0-0", "15-0", "15-15"]
    assert all(r["Svr"] == "1" for r in rows)
    assert [r["PtWinner"] for r in rows] == ["1", "2", "2"]


def test_map_is_matched_1_to_1_and_parses(imported):
    rows = read(imported / "data" / "mcp" / "tb_mcp_map.csv")
    assert list(rows[0]) == bundle.MAP_FIELDS
    assert [r["clip"] for r in rows] == [f"tb_point_{k:02d}"
                                         for k in (1, 2, 3)]
    assert [r["mcp_pt"] for r in rows] == ["1", "2", "3"]
    assert all(r["status"] == "matched" for r in rows)
    assert rows[0]["gms"] == "0-0" and rows[1]["pts"] == "15-0"
    # the fault point: parse_mcp must land on the second-serve string
    serve_d, strokes, played = parse_mcp(rows[1]["first"],
                                         rows[1]["second"])
    assert (serve_d, played) == ("4", "4f1*") and strokes == ["f"]


def test_alignment_feeds_p1_end(imported):
    rows = read(imported / "data" / "mcp" / "tb_clip_alignment.csv")
    assert list(rows[0]) == bundle.ALIGN_FIELDS
    ev = EvalCfg()                      # start_end 'far', no priors
    for a in rows:                      # game 0: no swaps yet
        assert p1_end(a, ev) == "far"


def test_alignment_pts_is_player1_first(tmp_path):
    # first_server=2: MCP Pts stays server-first, alignment flips
    setup = dict(SETUP, first_server=2)
    points = [dict(POINTS[0], winner="2"),  # server is P2 now,
              dict(POINTS[1], winner="1")]  # so winners flip too
    b = make_bundle(tmp_path, points=points, setup=setup)
    root = tmp_path / "repo"
    bundle.import_bundle(b, "tb", root=root)
    pts = read(root / "data" / "mcp" / "tb_points.csv")
    ali = read(root / "data" / "mcp" / "tb_clip_alignment.csv")
    assert pts[1]["Svr"] == "2" and pts[1]["Pts"] == "15-0"
    assert ali[1]["pts"] == "0-15"


def test_unseen_point_kept_in_points_but_not_aligned(tmp_path):
    points = POINTS + [{"first": "", "second": "", "notes": "",
                        "winner": "1", "start_s": "", "end_s": "",
                        "flags": "unseen"}]
    b = make_bundle(tmp_path, points=points)
    root = tmp_path / "repo"
    bundle.import_bundle(b, "tb", root=root)
    pts = read(root / "data" / "mcp" / "tb_points.csv")
    ali = read(root / "data" / "mcp" / "tb_clip_alignment.csv")
    assert len(pts) == 4 and pts[3]["Notes"].startswith("unseen;")
    assert len(ali) == 3                # no window -> no clip row


def test_export_shape_bundle_matches_session_shape(imported, tmp_path):
    # rebuild the frozen /export/bundle shape from the session import's
    # own output; both paths must generate identical join files
    b2 = tmp_path / "bundle2"
    b2.mkdir()
    shutil.copy(imported / "data" / "mcp" / "tb_points.csv",
                b2 / "points.csv")
    (b2 / "manifest.json").write_text(json.dumps(
        {"match_id": "tb", "setup": SETUP, "points": 3}))
    segs = ["clip,start_s,end_s"] + [
        f"tb_point_{k + 1:03d},{p['start_s']},{p['end_s']}"
        for k, p in enumerate(POINTS)]
    (b2 / "segments.csv").write_text("\n".join(segs) + "\n")
    root2 = tmp_path / "repo2"
    bundle.import_bundle(b2, "tb", root=root2)
    for name in ("tb_clip_alignment.csv", "tb_mcp_map.csv"):
        assert ((root2 / "data" / "mcp" / name).read_text()
                == (imported / "data" / "mcp" / name).read_text())


def test_yaml_scaffold_loads_as_match_config(imported, monkeypatch):
    monkeypatch.setattr(config, "ROOT", imported)
    monkeypatch.setattr(config, "MATCH_DIR",
                        imported / "data" / "matches")
    cfg = config.load("tb")
    assert cfg.id == "tb"
    assert cfg.eval.mcp_points.exists()
    assert cfg.eval.alignment.exists()
    assert cfg.eval.mcp_map.exists()
    # parity-dependent fields must ship as loud TODOs, not answers
    text = (imported / "data" / "matches" / "tb.yaml").read_text()
    assert "TODO(parity)" in text
    assert cfg.eval.set_priors == {} and cfg.eval.tiebreak_states == []


def test_dry_run_writes_nothing(tmp_path, capsys):
    b = make_bundle(tmp_path)
    root = tmp_path / "repo"
    bundle.import_bundle(b, "tb", root=root, dry_run=True)
    assert not root.exists()
    assert "would write" in capsys.readouterr().out


def test_refuses_overwrite_without_force(imported, tmp_path):
    b = tmp_path / "bundle"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        bundle.import_bundle(b, "tb", root=imported)
    bundle.import_bundle(b, "tb", root=imported, force=True)


def test_conflicting_point_blocks_import(tmp_path):
    # string says the server (P1) won, attested winner says P2 —
    # the app's export gate, mirrored
    points = [dict(POINTS[0], winner="2")]
    b = make_bundle(tmp_path, points=points)
    with pytest.raises(SystemExit, match="contradicts"):
        bundle.import_bundle(b, "tb", root=tmp_path / "repo")


@pytest.mark.skipif(shutil.which("ffmpeg") is None,
                    reason="ffmpeg not on PATH")
def test_clips_cut_from_video_beside_bundle(tmp_path):
    b = make_bundle(tmp_path)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=duration=15:size=64x64:rate=10",
         "-g", "10", str(b / "tb.mp4")], check=True)
    root = tmp_path / "repo"
    bundle.import_bundle(b, "tb", root=root)
    clips = sorted((root / "clips" / "points_tb").glob("*.mp4"))
    assert [c.name for c in clips] == [f"tb_point_{k:02d}.mp4"
                                       for k in (1, 2, 3)]
    assert all(c.stat().st_size > 0 for c in clips)
