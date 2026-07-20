"""Diagnosis 2d — server-end recovery from signals already on disk.

Truth: MCP svr + changeover parity. For t5 the set-2 prior is corrected
9 -> 10 in-memory (the pixel-verified eval bug; yaml untouched).

Rules measured per match (t5, t6, t7):
  R0  current pipeline (match_chart_v2 server_used)
  R1  stance-settle vote alone (serve.py stance-variant recipe)
  R1n stance-settle without the toss requirement
  R2  hybrid: R0, overridden by stance-settle end when it commits & disagrees
  R2n same with the no-toss settle
  R3  hybrid: R0, stance fills only the '?' clips
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import cv2

from courtvision.config import load
from courtvision.evaluate import p1_end, OTHER
from courtvision.court import CENTER_X
from courtvision.serve import (BASELINE_TOL, COVERAGE_MIN, TOSS_RATIO,
                               SETTLE_WIN_S, SETTLE_STEP_S,
                               SERVE_AFTER_SETTLE_S, toss_peak)


def stance_vote(cfg, stem, fps, center_tol, need_toss=True):
    """serve.py stance-variant recipe, replayed from the players CSV."""
    tpath = cfg.players_dir / f"players_{stem}.csv"
    if not tpath.exists():
        return None
    per = {"near": {}, "far": {}}
    with open(tpath) as f:
        for r in csv.DictReader(f):
            per[r["player"]][int(r["frame"])] = r
    if not per["near"] and not per["far"]:
        return None
    Hm = np.load(cfg.homography)
    odx, ody = cfg.load_offsets().get(stem, (0.0, 0.0))
    nfr = max([max(d) for d in per.values() if d], default=0) + 1

    def court_xy(r):
        pt = np.float32([[float(r["foot_x"]) - odx,
                          float(r["foot_y"]) - ody]]).reshape(-1, 1, 2)
        xy = cv2.perspectiveTransform(pt, Hm).reshape(2)
        return float(xy[0]), float(xy[1])

    win = int(SETTLE_WIN_S * fps)
    step = max(1, int(SETTLE_STEP_S * fps))
    candidates = []
    for side in ("near", "far"):
        settle = None
        for start in range(0, max(1, nfr - win), step):
            rows = [per[side][fi] for fi in sorted(per[side])
                    if start <= fi < start + win]
            if len(rows) / win < COVERAGE_MIN:
                continue
            xys = [court_xy(r) for r in rows]
            mx = float(np.median([p[0] for p in xys]))
            my = float(np.median([p[1] for p in xys]))
            lo, hi = BASELINE_TOL[side]
            if abs(mx - CENTER_X) <= center_tol and lo <= my <= hi:
                settle = (start, mx, my)
                break
        if settle is None:
            continue
        s0, mx, my = settle
        tr = 0.0
        tp = toss_peak(per[side], s0, s0 + int(SERVE_AFTER_SETTLE_S * fps))
        if tp is not None:
            tr = tp[1]
        if need_toss and (tp is None or tr < TOSS_RATIO):
            continue
        candidates.append({"side": side, "settle": s0, "toss_ratio": tr})
    if not candidates:
        return None
    best = min(candidates, key=lambda c: (c["settle"], -c["toss_ratio"]))
    return best["side"]


def truth_for(mid, prior_patch=None):
    cfg = load(mid)
    ev = cfg.eval
    if prior_patch:
        for k, v in prior_patch.items():
            ev.set_priors[k] = v
    mapd = {r["clip"]: r for r in csv.DictReader(open(ev.mcp_map))}
    align = {r["clip"]: r for r in csv.DictReader(open(ev.alignment))}
    match = {r["clip"]: r
             for r in csv.DictReader(open(cfg.charts_dir / "match_chart_v2.csv"))}
    out = []
    for clip, mc in match.items():
        m, a = mapd.get(clip), align.get(clip)
        if m is None or m["status"] != "matched":
            continue
        n_end = p1_end(a, ev)
        true_end = n_end if m["svr"] == "1" else OTHER[n_end]
        out.append((clip, true_end, mc["server_used"]))
    return cfg, out


for mid, patch in (("t5", {"1,0": 10}), ("t6", None), ("t7", None)):
    cfg, rows = truth_for(mid, patch)
    center_tol = float(cfg.serve_detect.get("center_tol_m", 2.0))
    fps_cache = {}
    res = {k: [0, 0] for k in ("R0", "R1", "R1n", "R2", "R2n", "R3")}
    cover = {"R1": 0, "R1n": 0}
    flips = {"R2": [], "R2n": []}
    for clip, true_end, used in rows:
        if clip not in fps_cache:
            cap = cv2.VideoCapture(str(cfg.clip_path(clip)))
            fps_cache[clip] = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        fps = fps_cache[clip]
        sv = stance_vote(cfg, clip, fps, center_tol, need_toss=True)
        svn = stance_vote(cfg, clip, fps, center_tol, need_toss=False)

        def score(key, end):
            res[key][0] += (end == true_end)
            res[key][1] += 1
        score("R0", used)
        if sv is not None:
            cover["R1"] += 1
            score("R1", sv)
        if svn is not None:
            cover["R1n"] += 1
            score("R1n", svn)
        r2 = sv if (sv is not None and used in ("near", "far", "?")
                    and sv != used) else used
        if r2 != used:
            flips["R2"].append((clip, used, sv, true_end))
        score("R2", r2)
        r2n = svn if (svn is not None and svn != used) else used
        if r2n != used:
            flips["R2n"].append((clip, used, svn, true_end))
        score("R2n", r2n)
        r3 = sv if (used == "?" and sv is not None) else used
        score("R3", r3)

    print(f"\n=== {mid} (n={len(rows)}) ===")
    for k in ("R0", "R1", "R1n", "R2", "R2n", "R3"):
        c, n = res[k]
        cov = f"  coverage {cover[k]}/{len(rows)}" if k in cover else ""
        print(f"{k:4} {c}/{n} ({100*c/max(n,1):.0f}%){cov}")
    for k in ("R2", "R2n"):
        good = sum(1 for _, _, s, t in flips[k] if s == t)
        print(f"{k} flips: {len(flips[k])} ({good} correct)  "
              + "; ".join(f"{c}:{u}->{s}{'✓' if s==t else '✗'}"
                          for c, u, s, t in flips[k][:14]))
