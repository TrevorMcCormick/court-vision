"""T1 (Nadal–Shapovalov test match) — fit the court homography.

The dev reel's homography does NOT transfer: the projected quad lands
~20 px off on t1 wide frames (t1_homography_check.png) — different
production day, remounted camera. And NOTHING automated survived night:

  attempt 1  M1 recipe verbatim: the blue->green paint boundary reads
             (V~155, S~55) under floodlights — same as grazing-angle
             real lines — and the extreme-cluster fit grabbed the paint
             edge instead of the baselines.
  attempt 2  tighter white (V>200, S<50): kept only camera-facing lines
             and lost the sidelines and far baseline entirely.
  attempt 3  guided bands around the dev-H prior: the far-baseline band
             locked onto ground-paint/banner whites 70 px off, and a
             sideline band silently mixed in doubles-line pixels — the
             fit passed residual checks while putting the center line
             32 px off its actual pixels. Plausible-but-wrong, caught
             only by LOOKING at the reprojection.

So: FOUR MANUAL CORNERS, read off gridded 6x zooms of the median plate
(corner_*.png), exactly one human input per match — the same bootstrap
cost this project accepted for M0's ball click. Everything automated
here is validation now: the tight-mask lines (center, service lines,
near baseline) took no part in the fit and must land on the projection.

All t1 artifacts go to outputs/t1/ — the dev reel's outputs stay frozen.

Usage:
    uv run experiments/t1_fit_homography.py clips/t1_nadal_shapo_30fps.mp4 --lo 11150 --hi 11350
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t1"
sys.path.insert(0, str(ROOT / "experiments"))
from m1_fit_court import (MODEL_LINES, seg_angle, fit_line,
                          line_y_at, line_x_at, cluster_1d)

# four manual correspondences from gridded zooms (corner_*.png), with a
# twist the crops taught the hard way: at the NEAR baseline the doubles
# corners sit at the frame edges (x~74 / ~1141, unreadable), and the
# crisp diagonals visible in the crops are the SINGLES sidelines. First
# two reads mislabeled them as doubles corners and the fit put the
# service line 15-23 px off. The perspective-scale cross-check (px/m
# linear in image y: 48 px/m at the far baseline, 80 at the near service
# line, extrapolates to 100 at the near baseline -> doubles span ~1095 px
# = the actual white extent 74..1141) settled the labels.
# far DOUBLES corners + near SINGLES corners:
MODEL_PTS = np.float32([(0, 0), (10.97, 0), (1.372, 23.77), (9.598, 23.77)])
CORNERS_IMG = np.float32([
    (338, 199), (868, 196),
    (182, 534), (1017, 532),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--lo", type=int, required=True)
    parser.add_argument("--hi", type=int, required=True)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- plate: median over the court-view run ----
    cap = cv2.VideoCapture(args.video)
    frames = []
    for fi in range(args.lo, args.hi, 4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    plate = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "plate_fit.png"), plate)
    print(f"plate from {len(frames)} frames ({args.lo}-{args.hi})")

    # ---- the fit: four manual correspondences ----
    H = cv2.getPerspectiveTransform(MODEL_PTS, CORNERS_IMG).astype(np.float64)
    np.save(OUT_DIR / "H_court_to_img.npy", H)
    np.save(OUT_DIR / "H_img_to_court.npy", np.linalg.inv(H))

    # ---- validation: tight-mask lines vs the projection ----
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    h, w = plate.shape[:2]
    blue = cv2.inRange(hsv, (95, 60, 40), (135, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    court_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(court_mask, [cv2.convexHull(max(contours, key=cv2.contourArea))],
                     -1, 255, -1)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    white = cv2.bitwise_and(white, court_mask)
    cv2.imwrite(str(OUT_DIR / "white_mask.png"), white)

    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=80,
                           minLineLength=60, maxLineGap=12)
    segs = segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4), int)
    angles = np.array([seg_angle(s) for s in segs])
    horiz = segs[(angles < 30) | (angles > 150)]
    vert = segs[(angles >= 30) & (angles <= 150)]

    def measured(lines, at, fn):
        vals = np.array([fn(fit_line([s[:2], s[2:]]), at) for s in lines])
        return [round(float(np.mean(vals[c])), 1) for c in cluster_1d(vals, tol=15)]

    hy = measured(horiz, w / 2, line_y_at) if len(horiz) else []
    vx = measured(vert, h * 0.55, line_x_at) if len(vert) else []
    print(f"tight-mask horizontals at mid-column y = {hy}")
    print(f"tight-mask verticals   at mid-row    x = {vx}")

    checks = {                       # model point -> should match a measured value
        "service_near y": ((5.485, 18.285), 1, hy),
        "baseline_near y": ((5.485, 23.77), 1, hy),
        "center_service x": ((5.485, 15.0), 0, vx),
    }
    for name, (pt, axis, obs) in checks.items():
        proj = cv2.perspectiveTransform(np.float32([pt]).reshape(-1, 1, 2), H).reshape(2)
        best = min(obs, key=lambda v: abs(v - proj[axis])) if obs else None
        d = abs(best - proj[axis]) if best is not None else float("nan")
        print(f"validate {name}: projected {proj[axis]:.1f}, "
              f"nearest measured {best}, err {d:.1f} px")

    overlay = plate.copy()
    for name, (p, q) in MODEL_LINES.items():
        pts = np.float32([p, q]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        color = (0, 255, 255) if "baseline" in name or "doubles" in name else (0, 128, 255)
        cv2.line(overlay, tuple(proj[0].astype(int)), tuple(proj[1].astype(int)), color, 2)
    cv2.imwrite(str(OUT_DIR / "model_reprojection.png"), overlay)
    print(f"-> {OUT_DIR}/model_reprojection.png, H_court_to_img.npy")


if __name__ == "__main__":
    main()
