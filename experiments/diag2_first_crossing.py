"""Diagnosis 2e — recovery rule: server end = direction of the EARLIEST
tracked net crossing in the ball CSV (adjacent tracked pair, gap-capped),
instead of the earliest gate-passing sustained run. Measured on t5
(corrected set-2 prior), t6, t7 against MCP truth.

Variants:
  Cg3 / Cg6 / Cg10   first crossing with pair gap <= 3 / 6 / 10 frames
  Cg6b               gap <= 6 AND both endpoints inside court band
                     (-4 .. L_C+4) — rejects airborne-projection junk
  H(g6b)             hybrid: R0 (server_used), but replaced by Cg6b's end
                     when Cg6b commits and disagrees
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import cv2

from courtvision.config import load
from courtvision.evaluate import p1_end, OTHER
from courtvision.court import NET_Y, L_C


def first_crossing(bfr, bcy, gap_max, band=None):
    for i in range(len(bfr) - 1):
        if bfr[i + 1] - bfr[i] > gap_max:
            continue
        a, b = bcy[i], bcy[i + 1]
        if (a - NET_Y) * (b - NET_Y) >= 0:
            continue
        if band and not (band[0] <= a <= band[1] and band[0] <= b <= band[1]):
            continue
        return "near" if b < a else "far"
    return None


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
        out.append((clip, n_end if m["svr"] == "1" else OTHER[n_end],
                    mc["server_used"]))
    return cfg, out


BAND = (-4.0, L_C + 4.0)
for mid, patch in (("t5", {"1,0": 10}), ("t6", None), ("t7", None)):
    cfg, rows = truth_for(mid, patch)
    Hm = np.load(cfg.homography)
    offsets = cfg.load_offsets()
    res = {k: [0, 0] for k in ("R0", "Cg3", "Cg6", "Cg10", "Cg6b", "Hg6b")}
    flips = []
    for clip, true_end, used in rows:
        bpath = cfg.ball_dir / f"ball_{clip}.csv"
        ends = {}
        if bpath.exists():
            ball = list(csv.DictReader(open(bpath)))
            if ball:
                bfr = np.array([int(r["frame"]) for r in ball])
                odx, ody = offsets.get(clip, (0.0, 0.0))
                pts = np.stack([[float(r["x_stab"]) - odx for r in ball],
                                [float(r["y_stab"]) - ody for r in ball]],
                               axis=1).reshape(-1, 1, 2).astype(np.float32)
                bcy = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)[:, 1]
                ends["Cg3"] = first_crossing(bfr, bcy, 3)
                ends["Cg6"] = first_crossing(bfr, bcy, 6)
                ends["Cg10"] = first_crossing(bfr, bcy, 10)
                ends["Cg6b"] = first_crossing(bfr, bcy, 6, BAND)

        def score(k, e):
            if e is not None:
                res[k][0] += (e == true_end)
                res[k][1] += 1
        score("R0", used)
        for k in ("Cg3", "Cg6", "Cg10", "Cg6b"):
            score(k, ends.get(k))
        h = ends.get("Cg6b") or used
        if used in ("near", "far") and ends.get("Cg6b") not in (None, used):
            flips.append((clip, used, ends["Cg6b"], true_end))
        score("Hg6b", h)

    print(f"\n=== {mid} (n={len(rows)}) ===")
    for k in ("R0", "Cg3", "Cg6", "Cg10", "Cg6b", "Hg6b"):
        c, n = res[k]
        print(f"{k:5} {c}/{n} ({100*c/max(n,1):.0f}%)"
              + (f"  [commits {n}/{len(rows)}]" if k.startswith("C") else ""))
    good = sum(1 for _, _, s, t in flips if s == t)
    print(f"Hg6b flips vs R0: {len(flips)} ({good} correct): "
          + "; ".join(f"{c}:{u}->{s}{'✓' if s==t else '✗'}"
                      for c, u, s, t in flips))
