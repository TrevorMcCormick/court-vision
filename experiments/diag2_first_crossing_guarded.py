"""Diagnosis 2f — guard the first-crossing flip so it can't damage t6.

Diagnostics per flip candidate + guarded hybrids:
  Hearly  flip only when the first crossing ends >= EARLY_S before the
          detected serve launch (evidence the launch gate skipped an
          earlier crossing — the return-flip signature)
  Hfast   flip only when the crossing pair moves >= FAST_M court-m/frame
  Hboth   both guards
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

BAND = (-4.0, L_C + 4.0)
EARLY_S = 0.4
FAST_M = 0.8   # m per frame across the crossing pair (fast flight)


def first_crossing(bfr, bcy, gap_max=6):
    for i in range(len(bfr) - 1):
        gap = bfr[i + 1] - bfr[i]
        if gap > gap_max:
            continue
        a, b = bcy[i], bcy[i + 1]
        if (a - NET_Y) * (b - NET_Y) >= 0:
            continue
        if not (BAND[0] <= a <= BAND[1] and BAND[0] <= b <= BAND[1]):
            continue
        end = "near" if b < a else "far"
        speed = abs(b - a) / gap
        return end, int(bfr[i]), int(bfr[i + 1]), speed
    return None


def truth_for(mid, prior_patch=None):
    cfg = load(mid)
    ev = cfg.eval
    if prior_patch:
        ev.set_priors.update(prior_patch)
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


for mid, patch in (("t5", {"1,0": 10}), ("t6", None), ("t7", None)):
    cfg, rows = truth_for(mid, patch)
    Hm = np.load(cfg.homography)
    offsets = cfg.load_offsets()
    serves = {r["clip"]: r for r in csv.DictReader(open(cfg.serves))}
    res = {k: [0, 0] for k in ("R0", "Hearly", "Hfast", "Hboth")}
    detail = []
    for clip, true_end, used in rows:
        bpath = cfg.ball_dir / f"ball_{clip}.csv"
        fc = None
        if bpath.exists():
            ball = list(csv.DictReader(open(bpath)))
            if ball:
                bfr = np.array([int(r["frame"]) for r in ball])
                odx, ody = offsets.get(clip, (0.0, 0.0))
                pts = np.stack([[float(r["x_stab"]) - odx for r in ball],
                                [float(r["y_stab"]) - ody for r in ball]],
                               axis=1).reshape(-1, 1, 2).astype(np.float32)
                bcy = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)[:, 1]
                fc = first_crossing(bfr, bcy)
        s = serves.get(clip, {})
        fps = 30.0
        serve_f = int(s["serve_frame"]) if s.get("serve_frame") else None

        def hyb(guard):
            if fc is None or used not in ("near", "far"):
                return fc[0] if (fc and used == "?") else used
            end, f0, f1, speed = fc
            if end == used:
                return used
            early = serve_f is not None and f1 <= serve_f - EARLY_S * fps
            fast = speed >= FAST_M
            ok = {"early": early, "fast": fast, "both": early and fast}[guard]
            return end if ok else used

        res["R0"][0] += used == true_end; res["R0"][1] += 1
        for k, g in (("Hearly", "early"), ("Hfast", "fast"), ("Hboth", "both")):
            e = hyb(g)
            res[k][0] += e == true_end; res[k][1] += 1
            if k == "Hboth" and e != used:
                detail.append((clip, used, e, true_end,
                               fc[1] if fc else None, serve_f,
                               round(fc[3], 2) if fc else None))

    print(f"\n=== {mid} (n={len(rows)}) ===")
    for k in ("R0", "Hearly", "Hfast", "Hboth"):
        c, n = res[k]
        print(f"{k:7} {c}/{n} ({100*c/max(n,1):.0f}%)")
    good = sum(1 for d in detail if d[2] == d[3])
    print(f"Hboth flips: {len(detail)} ({good} correct): "
          + "; ".join(f"{c}:{u}->{e}{'✓' if e==t else '✗'}(f{f0}<srv{sf},v{v})"
                      for c, u, e, t, f0, sf, v in detail))
