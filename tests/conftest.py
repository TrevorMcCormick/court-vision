"""A minimal on-disk match for review/session tests: a MatchConfig
pointing at tmp dirs, a 4-row export CSV (one blank-Pt row), and two
tiny fake 'clips' (bytes on disk are enough for Range tests)."""

import csv

import pytest

from courtvision.config import MatchConfig, Staging, EvalCfg

EXPORT_FIELDS = ["match_id", "Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts",
                 "Svr", "1st", "2nd", "confidence", "conf_p", "clip",
                 "serve_s", "n_shots"]

ROWS = [
    {"match_id": "m", "Pt": "2", "Set1": "0", "Set2": "0", "Gm1": "0",
     "Gm2": "0", "Pts": "0-15", "Svr": "1", "1st": "s4b3f1w@", "2nd": "",
     "confidence": "high", "conf_p": "0.966", "clip": "tt_point_01",
     "serve_s": "3.2", "n_shots": "3"},
    {"match_id": "m", "Pt": "3", "Set1": "0", "Set2": "0", "Gm1": "0",
     "Gm2": "0", "Pts": "0-30", "Svr": "1", "1st": "s?f2?3?", "2nd": "",
     "confidence": "low", "conf_p": "0.51", "clip": "tt_point_02",
     "serve_s": "3.0", "n_shots": "3"},
    {"match_id": "m", "Pt": "", "Set1": "0", "Set2": "0", "Gm1": "1",
     "Gm2": "0", "Pts": "0-0", "Svr": "", "1st": "s6*", "2nd": "",
     "confidence": "low", "conf_p": "0.40", "clip": "tt_point_03",
     "serve_s": "0.8", "n_shots": "1"},
    {"match_id": "m", "Pt": "5", "Set1": "0", "Set2": "0", "Gm1": "1",
     "Gm2": "0", "Pts": "15-0", "Svr": "2", "1st": "s5b2f1*", "2nd": "",
     "confidence": "high", "conf_p": "0.91", "clip": "tt_point_04",
     "serve_s": "1.5", "n_shots": "3"},
]


@pytest.fixture
def cfg(tmp_path):
    clips = tmp_path / "clips"
    out = tmp_path / "out"
    (out / "export").mkdir(parents=True)
    clips.mkdir()
    for r in ROWS:
        (clips / f"{r['clip']}.mp4").write_bytes(
            bytes(range(256)) * 4)                 # 1024 known bytes
    with open(out / "export" / "tt_mcp_draft.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
        w.writeheader()
        w.writerows(ROWS)
    return MatchConfig(
        id="tt", title="tt", match="", clips_dir=clips, out_dir=out,
        ball_dir=out / "ball", players_dir=out / "players",
        charts_dir=out / "charts", homography=out / "H.npy",
        serves=out / "serves.csv", clip_offsets=None,
        lefty={"near": False, "far": False}, staging=Staging(),
        eval=EvalCfg(), serve_detect={}, players_detect={},
        video=None, court_detect={},
    )
