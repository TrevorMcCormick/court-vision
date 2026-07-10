"""T2 (Federer–Haase test match) — fit the court homography.

T2 is the near-transfer control: day session, same court and broadcast
family as the M1 dev reel. So unlike t1 (night, where nothing automated
survived), the FIRST attempt here is the M1 recipe — blue-hull white
mask, Hough, extreme clusters = baselines + doubles sidelines, four
intersections, done. t1's manual-corner path stays in the script as the
fallback (--manual), with the same arithmetic cross-checks:

  * far-corner midpoint x must ~= the measured center-service-line x
  * px/m lateral scale must be linear in image y (near-baseline reads
    can be SINGLES corners in disguise — t1 lost two fits to that)

Validation is held out either way: tight-mask lines (center, service
lines, near baseline for the manual path; singles/service/center for
the auto path) took no part in the fit, and the reprojection overlay
gets LOOKED at — residuals can lie, overlays cannot.

All t2 artifacts go to outputs/t2/.

Usage:
    uv run experiments/t2_fit_homography.py clips/t2_federer_haase_30fps.mp4 --lo 1650 --hi 1930
    uv run experiments/t2_fit_homography.py ... --manual   # corner-constant fallback
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t2"
sys.path.insert(0, str(ROOT / "experiments"))
from m1_fit_court import (MODEL_LINES, MODEL_CORNERS, seg_angle, fit_line,
                          line_y_at, line_x_at, intersect, cluster_1d)

W_COURT, L_COURT = 10.97, 23.77

# manual fallback (t1 recipe): read off gridded zoom crops corner_*.png.
# Fill these in only if the auto fit fails the eyeball check.
MODEL_PTS_MANUAL = np.float32([(0, 0), (10.97, 0), (0, 23.77), (10.97, 23.77)])
CORNERS_IMG_MANUAL = np.float32([
    (0, 0), (0, 0),
    (0, 0), (0, 0),
])


def corner_crops(plate, corners, zoom=6):
    """Gridded zoom crops of the four corner regions for manual reads."""
    names = ["farL", "farR", "nearL", "nearR"]
    for name, (cx, cy) in zip(names, corners):
        x1, y1 = int(cx) - 55, int(cy) - 40
        x2, y2 = int(cx) + 55, int(cy) + 40
        x1, y1 = max(0, x1), max(0, y1)
        crop = plate[y1:y2, x1:x2]
        big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        for gx in range(x1, x2, 10):
            px = (gx - x1) * zoom
            cv2.line(big, (px, 0), (px, big.shape[0]), (0, 255, 255), 1)
            cv2.putText(big, str(gx), (px + 2, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
        for gy in range(y1, y2, 10):
            py = (gy - y1) * zoom
            cv2.line(big, (0, py), (big.shape[1], py), (0, 255, 255), 1)
            cv2.putText(big, str(gy), (2, py + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
        cv2.imwrite(str(OUT_DIR / f"corner_{name}.png"), big)
    print(f"-> corner crops in {OUT_DIR} (corner_farL/farR/nearL/nearR.png)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--lo", type=int, required=True)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--manual", action="store_true")
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

    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    h, w = plate.shape[:2]

    # blue court hull (shared by both paths)
    blue = cv2.inRange(hsv, (95, 60, 40), (135, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    court = max(contours, key=cv2.contourArea)
    court_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(court_mask, [cv2.convexHull(court)], -1, 255, -1)

    if args.manual:
        H = cv2.getPerspectiveTransform(MODEL_PTS_MANUAL, CORNERS_IMG_MANUAL).astype(np.float64)
        corners_img = CORNERS_IMG_MANUAL
    else:
        # ---- M1 recipe: loose white inside the dilated hull ----
        hull_dil = cv2.dilate(court_mask, np.ones((25, 25), np.uint8))
        white_loose = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
        white_loose = cv2.bitwise_and(white_loose, hull_dil)
        segs = cv2.HoughLinesP(white_loose, 1, np.pi / 360, threshold=80,
                               minLineLength=60, maxLineGap=12)
        segs = segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4), int)
        angles = np.array([seg_angle(s) for s in segs])
        horiz = segs[(angles < 30) | (angles > 150)]
        vert = segs[(angles >= 30) & (angles <= 150)]
        print(f"auto fit: {len(horiz)} horizontal segments, {len(vert)} vertical")

        hy = np.array([line_y_at(fit_line([s[:2], s[2:]]), w / 2) for s in horiz])
        vx = np.array([line_x_at(fit_line([s[:2], s[2:]]), h * 0.5) for s in vert])
        h_clusters = cluster_1d(hy, tol=15)
        v_clusters = cluster_1d(vx, tol=25)
        print(f"h clusters at y~{[round(float(np.mean(hy[c]))) for c in h_clusters]}")
        print(f"v clusters at x~{[round(float(np.mean(vx[c]))) for c in v_clusters]}")

        def merged_line(seg_group):
            pts = np.concatenate([[s[:2], s[2:]] for s in seg_group])
            return fit_line(pts)

        baseline_far = merged_line(horiz[h_clusters[0]])
        baseline_near = merged_line(horiz[h_clusters[-1]])
        doubles_left = merged_line(vert[v_clusters[0]])
        doubles_right = merged_line(vert[v_clusters[-1]])
        corners_img = np.float32([
            intersect(baseline_far, doubles_left),
            intersect(baseline_far, doubles_right),
            intersect(baseline_near, doubles_left),
            intersect(baseline_near, doubles_right),
        ])
        for name, pt in zip(["far-L", "far-R", "near-L", "near-R"], corners_img):
            print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")
        H = cv2.getPerspectiveTransform(MODEL_CORNERS, corners_img).astype(np.float64)

    corner_crops(plate, corners_img)

    # ---- arithmetic cross-checks (the t1 lessons) ----
    tight = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    tight = cv2.bitwise_and(tight, court_mask)
    cv2.imwrite(str(OUT_DIR / "white_mask.png"), tight)
    segs_t = cv2.HoughLinesP(tight, 1, np.pi / 360, threshold=80,
                             minLineLength=60, maxLineGap=12)
    segs_t = segs_t.reshape(-1, 4) if segs_t is not None else np.zeros((0, 4), int)
    ang_t = np.array([seg_angle(s) for s in segs_t])
    horiz_t = segs_t[(ang_t < 30) | (ang_t > 150)]
    vert_t = segs_t[(ang_t >= 30) & (ang_t <= 150)]

    def measured(lines, at, fn):
        vals = np.array([fn(fit_line([s[:2], s[2:]]), at) for s in lines])
        return [round(float(np.mean(vals[c])), 1) for c in cluster_1d(vals, tol=15)]

    hy_t = measured(horiz_t, w / 2, line_y_at) if len(horiz_t) else []
    vx_t = measured(vert_t, h * 0.55, line_x_at) if len(vert_t) else []
    print(f"tight-mask horizontals at mid-column y = {hy_t}")
    print(f"tight-mask verticals   at mid-row    x = {vx_t}")

    mid_far = (corners_img[0][0] + corners_img[1][0]) / 2
    print(f"cross-check: far-corner midpoint x = {mid_far:.1f} "
          f"(must ~= measured center-service x)")
    span_far = corners_img[1][0] - corners_img[0][0]
    span_near = corners_img[3][0] - corners_img[2][0]
    print(f"cross-check: doubles span far {span_far:.0f}px @y~{corners_img[0][1]:.0f}, "
          f"near {span_near:.0f}px @y~{corners_img[2][1]:.0f} "
          f"({span_far / W_COURT:.1f} / {span_near / W_COURT:.1f} px/m)")

    # ---- validation: tight-mask lines vs the projection ----
    checks = {
        "service_near y": ((5.485, 18.285), 1, hy_t),
        "service_far y": ((5.485, 5.485), 1, hy_t),
        "baseline_near y": ((5.485, 23.77), 1, hy_t),
        "center_service x": ((5.485, 15.0), 0, vx_t),
    }
    for name, (pt, axis, obs) in checks.items():
        proj = cv2.perspectiveTransform(np.float32([pt]).reshape(-1, 1, 2), H).reshape(2)
        best = min(obs, key=lambda v: abs(v - proj[axis])) if obs else None
        d = abs(best - proj[axis]) if best is not None else float("nan")
        print(f"validate {name}: projected {proj[axis]:.1f}, "
              f"nearest measured {best}, err {d:.1f} px")

    # ---- held-out lines vs nearest white pixel (loose mask keeps the
    # service lines the tight mask dropped; m1's dist-transform check) ----
    hull_dil2 = cv2.dilate(court_mask, np.ones((25, 25), np.uint8))
    white_all = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    white_all = cv2.bitwise_and(white_all, hull_dil2)
    dist = cv2.distanceTransform(255 - white_all, cv2.DIST_L2, 5)
    held_out = ["singles_left", "singles_right", "service_far",
                "service_near", "center_service"]
    for name in held_out:
        p, q = MODEL_LINES[name]
        ts = np.linspace(0, 1, 50)
        pts = np.float32([(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
                          for t in ts])
        proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        inside = ((proj[:, 0] >= 0) & (proj[:, 0] < w)
                  & (proj[:, 1] >= 0) & (proj[:, 1] < h))
        d = dist[proj[inside, 1].astype(int), proj[inside, 0].astype(int)]
        print(f"held-out {name}: mean dist to white px = {d.mean():.1f} px "
              f"(max {d.max():.1f})")

    np.save(OUT_DIR / "H_court_to_img.npy", H)
    np.save(OUT_DIR / "H_img_to_court.npy", np.linalg.inv(H))

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
