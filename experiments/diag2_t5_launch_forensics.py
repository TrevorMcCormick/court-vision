"""Diagnosis 2b — for each wrong-end t5 clip: was the TRUE serve flight
in the ball track before the detected launch, and why did find_launch
reject it? Classifies (a) untracked-serve vs (c) gate-rejected-serve.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import cv2

from courtvision.config import load
from courtvision.court import NET_Y, L_C
from courtvision.serve import LAUNCH_WIN_S, LAUNCH_SPAN_M, LAUNCH_MONO, LAUNCH_GAP

cfg = load("t5")
Hm = np.load(cfg.homography)
offsets = cfg.load_offsets()

diag = list(csv.DictReader(open(
    Path(__file__).resolve().parent.parent / "outputs/diag/t5_server_end_diag.csv")))
wrong = [r for r in diag if r["ok"] == "False" and r["src"] == "ball"]

def load_track(clip):
    ball = list(csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv")))
    bfr = np.array([int(r["frame"]) for r in ball])
    odx, ody = offsets.get(clip, (0.0, 0.0))
    pts = np.stack([[float(r["x_stab"]) - odx for r in ball],
                    [float(r["y_stab"]) - ody for r in ball]],
                   axis=1).reshape(-1, 1, 2).astype(np.float32)
    c = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    return ball, bfr, c[:, 1], c[:, 0]

print("true-direction net crossings BEFORE the detected launch, per wrong clip")
print("(sign: near serve = cy decreasing across net; far serve = cy increasing)")
for r in wrong:
    clip = r["clip"]
    ball, bfr, bcy, bcx = load_track(clip)
    f_launch = int(r["serve_f"])
    true_end = r["true_end"]
    want = -1 if true_end == "near" else +1     # sign of cy step across net

    # every adjacent-tracked-pair crossing of NET_Y before the launch
    crossings = []
    for i in range(len(bfr) - 1):
        if bfr[i + 1] > f_launch:
            break
        d = bcy[i + 1] - bcy[i]
        if (bcy[i] - NET_Y) * (bcy[i + 1] - NET_Y) < 0:
            crossings.append((int(bfr[i]), int(bfr[i + 1] - bfr[i]),
                              round(float(bcy[i]), 1), round(float(bcy[i + 1]), 1),
                              "true-dir" if np.sign(d) == want else "wrong-dir"))
    # coverage in the 1.5 s before the launch
    fps = 30.0
    pre_lo = max(0, f_launch - int(1.5 * fps))
    n_pre = int(np.sum((bfr >= pre_lo) & (bfr < f_launch)))
    cov_pre = n_pre / max(1, f_launch - pre_lo)
    tag = "NO-TRACKED-CROSSING-PRE" if not any(c[4] == "true-dir" for c in crossings) else "TRUE-DIR-CROSSING-PRESENT"
    print(f"\n{clip} true={true_end} det={r['det']} launch_f={f_launch} "
          f"cov_pre1.5s={cov_pre:.2f} first_tracked={r['first_tracked_f']}  [{tag}]")
    for c in crossings[:8]:
        print(f"   f{c[0]} gap={c[1]} cy {c[2]}->{c[3]}  {c[4]}")

    # for the true-dir crossings: why did find_launch reject the run there?
    for c in crossings:
        if c[4] != "true-dir":
            continue
        i0 = int(np.searchsorted(bfr, c[0]))
        # rebuild the run find_launch would test starting from a few
        # points before the crossing
        for istart in range(max(0, i0 - 6), i0 + 1):
            j = istart
            maxgap = 0
            while (j + 1 < len(bfr) and bfr[j + 1] - bfr[j] <= LAUNCH_GAP
                   and bfr[j + 1] - bfr[istart] <= LAUNCH_WIN_S * fps):
                j += 1
            seg = bcy[istart:j + 1]
            if len(seg) < 2:
                continue
            crosses = (seg[0] - NET_Y) * (seg[-1] - NET_Y) < 0
            span = abs(seg[-1] - seg[0])
            dd = np.diff(seg)
            mono = float(np.mean(np.sign(dd) == np.sign(seg[-1] - seg[0]))) if len(dd) else 0
            gaps = np.diff(bfr[istart:j + 1])
            print(f"   run@f{bfr[istart]}(n={len(seg)}) crossnet={crosses} "
                  f"span={span:.1f} mono={mono:.2f} maxgap={int(gaps.max()) if len(gaps) else 0} "
                  f"gap_after_run={int(bfr[j+1]-bfr[j]) if j+1 < len(bfr) else -1}")
            break
        break  # only the first true-dir crossing
