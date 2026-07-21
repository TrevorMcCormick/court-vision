"""Import a charter training bundle as benchmark-match scaffolding.

The charting app's export (docs/specs/2026-07-19-charting-app-design.md,
"Export: the training bundle") is an import-ready benchmark match: the
charter attested every point's string, winner, and video window, so the
clip->MCP join that costs a by-eye bug transcription pass on broadcast
matches (extract.bug_sheets + align) is 1:1 BY CONSTRUCTION here —
every clip maps 'matched' to its own Pt.

Two accepted bundle shapes:

  export shape   points.csv in MCP columns (chartapp.EXPORT_FIELDS) +
                 segments.csv (clip,start_s,end_s) — the /export/bundle
                 files saved into a directory
  session shape  a chart-along session dir (outputs/charting/<id>/):
                 points.csv in raw columns (first,second,notes,winner,
                 start_s,end_s,flags). Score columns are replayed here
                 through score.Score exactly as the app replays them —
                 raw inputs are the storage contract, bookkeeping is
                 never trusted from disk — including the app's export
                 gate (refuse while any point contradicts the replay).

Produces (repo-relative, refusing to overwrite unless --force):

  data/mcp/<id>_points.csv            MCP points file (evaluate shape)
  data/mcp/<id>_clip_alignment.csv    score state per clip, player-1-first
  data/mcp/<id>_mcp_map.csv           every clip 'matched' to its own Pt
  clips/points_<id>/<id>_point_NN.mp4 cut from the source video
  data/matches/<id>.yaml              config scaffold; staging knobs and
                                      the parity-dependent eval fields
                                      (start_end, set_priors) stay TODO

Usage:
    uv run python -m courtvision import-bundle outputs/charting/m1 --id t8
"""

import csv
import json
import subprocess

from .config import ROOT
from .chartapp import EXPORT_FIELDS
from .score import Score, winner_from_strings

ALIGN_FIELDS = ["clip", "set1", "set2", "gm1", "gm2", "pts",
                "server_bug", "note"]
MAP_FIELDS = ["clip", "note", "mcp_pt", "svr", "first", "second",
              "winner", "status", "gms", "pts"]

# Charter-stamped windows are tight (accept-to-accept, human reflexes),
# and ffmpeg stream-copy opens on the keyframe at/before -ss anyway —
# a small symmetric pad keeps the serve toss and the ball's last bounce
# in frame. A convention margin, not a tuned constant.
PAD_PRE_S = 0.5
PAD_POST_S = 0.5


def _p1_first(pts, svr):
    """MCP Pts is SERVER-first; the alignment CSVs are player-1-first
    (align.py's t1 trap) — flip whenever player 2 serves."""
    if svr == "2" and "-" in pts:
        a, _, b = pts.partition("-")
        return f"{b}-{a}"
    return pts


def _replay_raw(points, setup):
    """Session-shape points -> MCP rows, replaying score.Score the way
    chartapp.ChartSession does (raw inputs only; every score column
    recomputed; the attested winner is truth and any string that
    contradicts it under the replayed server blocks the import, same
    as the app's export gate)."""
    sc = Score(best_of=int(setup["best_of"]),
               final_set=setup["final_set"],
               first_server=int(setup["first_server"]))
    rows = []
    for i, p in enumerate(points):
        server = sc.server
        rel = winner_from_strings(p["first"], p["second"])
        if p.get("winner"):
            w = int(p["winner"])
        elif rel is not None:
            w = server if rel == 1 else (2 if server == 1 else 1)
        else:
            raise SystemExit(f"point {i + 1}: no winner stored or "
                             f"derivable — reconcile in the app first")
        if (rel is not None and p.get("flags") != "unseen"
                and (server if rel == 1 else (2 if server == 1 else 1)) != w):
            raise SystemExit(f"point {i + 1}: string contradicts the "
                             f"attested winner under the replayed "
                             f"server — reconcile in the app first")
        ctx = sc.point(w)
        notes = p.get("notes", "")
        if p.get("flags") == "unseen":
            notes = f"unseen;{notes}" if notes else "unseen;"
        rows.append({"Pt": str(i + 1), "Svr": ctx["Svr"],
                     "Set1": ctx["Set1"], "Set2": ctx["Set2"],
                     "Gm1": ctx["Gm1"], "Gm2": ctx["Gm2"],
                     "Pts": ctx["Pts"], "Gm#": ctx["Gm#"],
                     "TbSet": ctx["TbSet"], "1st": p["first"],
                     "2nd": p["second"], "Notes": notes,
                     "PtWinner": str(w)})
    return rows


def _load(bundle_dir):
    """-> (mcp_rows, {pt: (start_s, end_s)}, setup, source match_id)."""
    man = json.loads((bundle_dir / "manifest.json").read_text())
    setup = man["setup"]
    with open(bundle_dir / "points.csv") as f:
        pts = list(csv.DictReader(f))
    if not pts:
        raise SystemExit(f"{bundle_dir}/points.csv is empty")

    timing = {}
    if "Pt" in pts[0]:                       # export shape
        rows = pts
        src_id = man.get("match_id", pts[0].get("match_id", ""))
        seg_p = bundle_dir / "segments.csv"
        if not seg_p.exists():
            raise SystemExit(f"{bundle_dir}: MCP-shape points.csv has no "
                             f"timing columns and segments.csv is missing")
        for s in csv.DictReader(open(seg_p)):
            pt = int(s["clip"].rsplit("_", 1)[1])
            timing[pt] = (float(s["start_s"]), float(s["end_s"]))
    elif "first" in pts[0]:                  # session shape
        rows = _replay_raw(pts, setup)
        src_id = man.get("match_id", bundle_dir.name)
        for i, p in enumerate(pts):
            if (p.get("start_s") and p.get("end_s")
                    and float(p["start_s"]) < float(p["end_s"])):
                timing[i + 1] = (float(p["start_s"]), float(p["end_s"]))
    else:
        raise SystemExit(f"{bundle_dir}/points.csv: unrecognized columns "
                         f"{list(pts[0])[:4]}...")
    for r in rows:
        r["match_id"] = src_id
    return rows, timing, setup, src_id


def _yaml_scaffold(match_id, setup, src_id, video_name, n_pts, n_clips):
    p1, p2 = setup.get("player1", "?"), setup.get("player2", "?")
    return f"""\
# {match_id} — {p1}-{p2}, imported charter bundle ({n_pts} points,
# {n_clips} charter-stamped clips). Scaffold from `courtvision
# import-bundle`; the eval join files are complete, the pipeline
# staging below is the usual manual pass.
id: {match_id}
title: "{match_id} {p1}-{p2}"
match: "{p1}-{p2} (imported charter bundle)"
video: clips/{video_name}
clips_dir: clips/points_{match_id}
out_dir: outputs/{match_id}

lefty: {{near: false, far: false}}   # TODO: look up handedness

# staging:                  # TODO: manual staging knobs (docs/USAGE.md)
#   lock_serve: true
# court_detect:             # TODO: hull bands + fit window (fitcourt,
#   hull_lo: [...]          #       probe) before track-ball/players

eval:
  title: "{match_id}: imported charter bundle"
  mcp_map: data/mcp/{match_id}_mcp_map.csv
  alignment: data/mcp/{match_id}_clip_alignment.csv
  mcp_points: data/mcp/{match_id}_points.csv
  mcp_match_id: "{src_id}"
  # TODO(parity): start_end and set_priors need the staging parity
  # pass — pixel-verify serve ends, then experiments/parity_audit.py.
  # The 2026-07-20 parity-truth lesson: after any ODD-total set the
  # prior is NOT the game sum (the set-end change and the game-1
  # change are consecutive — a 6-3 set is 9 games but the next set's
  # prior is 10, not 9; and a TB set contributes EVEN parity).
  start_end: far            # TODO: parity vote
  set_priors: {{}}            # TODO: after the parity audit
  tiebreak_states: []       # TODO: "s1,s2" states where 6-6 is a TB
"""


def import_bundle(bundle_dir, match_id, root=ROOT, dry_run=False,
                  force=False):
    """Turn a training bundle into benchmark-match scaffolding under
    root (data/mcp CSVs, per-point clips, data/matches/<id>.yaml)."""
    rows, timing, setup, src_id = _load(bundle_dir)
    timed = [r for r in rows if int(r["Pt"]) in timing]

    align_rows, map_rows = [], []
    for r in timed:
        clip = f"{match_id}_point_{int(r['Pt']):02d}"
        align_rows.append({
            "clip": clip, "set1": r["Set1"], "set2": r["Set2"],
            "gm1": r["Gm1"], "gm2": r["Gm2"],
            "pts": _p1_first(r["Pts"], r["Svr"]),
            "server_bug": r["Svr"], "note": ""})
        map_rows.append({
            "clip": clip, "note": "", "mcp_pt": r["Pt"], "svr": r["Svr"],
            "first": r["1st"], "second": r["2nd"],
            "winner": r["PtWinner"], "status": "matched",
            "gms": f"{r['Gm1']}-{r['Gm2']}", "pts": r["Pts"]})

    mcp_dir = root / "data" / "mcp"
    clips_dir = root / "clips" / f"points_{match_id}"
    targets = {
        mcp_dir / f"{match_id}_points.csv": (EXPORT_FIELDS, rows),
        mcp_dir / f"{match_id}_clip_alignment.csv": (ALIGN_FIELDS,
                                                     align_rows),
        mcp_dir / f"{match_id}_mcp_map.csv": (MAP_FIELDS, map_rows),
    }
    yaml_path = root / "data" / "matches" / f"{match_id}.yaml"
    clip_paths = {m["clip"]: clips_dir / f"{m['clip']}.mp4"
                  for m in map_rows}

    existing = [p for p in (*targets, yaml_path, *clip_paths.values())
                if p.exists()]
    if existing and not force:
        listing = "\n  ".join(str(p) for p in existing)
        raise SystemExit(f"refusing to overwrite (use --force):\n"
                         f"  {listing}")

    # the video: beside the bundle, or already staged in clips/
    video_name = setup.get("video", "")
    video = next((p for p in (bundle_dir / video_name,
                              root / "clips" / video_name) if video_name
                  and p.exists()), None)

    tag = "would write" if dry_run else "->"
    for path, (fields, out) in targets.items():
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields,
                                   extrasaction="ignore")
                w.writeheader()
                w.writerows(out)
        print(f"{tag} {path} ({len(out)} rows)")
    if not dry_run:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(_yaml_scaffold(
            match_id, setup, src_id, video_name, len(rows), len(timed)))
    print(f"{tag} {yaml_path} (start_end/set_priors TODO: parity pass)")

    if video is None:
        print(f"video '{video_name}' not found beside the bundle or in "
              f"{root / 'clips'} — skipping clip cuts (rerun with the "
              f"file in place, or cut manually per the alignment CSV)")
    else:
        if not dry_run:
            clips_dir.mkdir(parents=True, exist_ok=True)
        for m in map_rows:
            start, end = timing[int(m["mcp_pt"])]
            a = max(0.0, start - PAD_PRE_S)
            dur = (end - a) + PAD_POST_S
            out = clip_paths[m["clip"]]
            if not dry_run:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}",
                     "-i", str(video), "-t", f"{dur:.3f}",
                     "-c", "copy", "-an", str(out)], check=True)
        print(f"{tag} {len(map_rows)} clips in {clips_dir} "
              f"(stream-copy from {video})")

    n_untimed = len(rows) - len(timed)
    if n_untimed:
        print(f"note: {n_untimed} point(s) without a video window "
              f"(unseen/unstamped) are in the points file but have no "
              f"clip/alignment row")
    if dry_run:
        print("dry run: nothing written")
