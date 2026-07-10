"""T4 (Krejcikova-Paolini, Wimbledon 2024 final) — fit the court homography.

Twin of t3_fit_homography (the clay recipe: ECC-stabilized plate,
filled-contour color hull, tophat line mask, symmetric-pair verticals,
scored far-line choice, LS refit on all labeled intersections) with
grass constants:

  * hull color = grass greens PLUS the worn baseline browns (H 15-19 —
    nearly clay-colored; a green-only hull loses the baselines' ground)
    with V >= 70 to exclude the dark-green surround walls (V ~50)
  * on grass the FAR BASELINE IS VISIBLE (white on green survives wear
    better than white on clay dust), so 0.0 joins the horizontal label
    set and the LS refit gets a 4th horizontal

The far-line candidate scoring stays: it must pick the far SERVICE
line (y~305) over the far BASELINE (y~222) for the initial 4-point
basis, and the interior-line scorer is what makes that choice safe.

All t4 artifacts go to outputs/t4/.

Usage:
    uv run experiments/t4_fit_homography.py clips/t4_krejcikova_paolini_30fps.mp4 --lo 15000 --hi 15360
    uv run experiments/t4_fit_homography.py ... --manual   # corner-constant fallback
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t4"
sys.path.insert(0, str(ROOT / "experiments"))
from m1_fit_court import (MODEL_LINES, seg_angle, fit_line,
                          line_y_at, line_x_at, intersect, cluster_1d)

W_COURT, L_COURT = 10.97, 23.77
SERVICE_FAR_Y = 5.485

GRASS_LO, GRASS_HI = (10, 25, 70), (50, 255, 255)
TOPHAT_K = 25
TOPHAT_T = 25
V_MIN = 170
HULL_ERODE = 15
SYM_TOL = 12
MIN_SUPPORT = 3

MODEL_PTS_MANUAL = np.float32([(0, 0), (10.97, 0), (0, 23.77), (10.97, 23.77)])
CORNERS_IMG_MANUAL = np.float32([
    (0, 0), (0, 0),
    (0, 0), (0, 0),
])


def corner_crops(plate, corners, zoom=6):
    names = ["farL", "farR", "nearL", "nearR"]
    for name, (cx, cy) in zip(names, corners):
        x1, y1 = int(cx) - 55, int(cy) - 40
        x2, y2 = int(cx) + 55, int(cy) + 40
        x1, y1 = max(0, x1), max(0, y1)
        crop = plate[y1:y2, x1:x2]
        if crop.size == 0:
            continue
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
    print(f"-> corner crops in {OUT_DIR}")


def stabilized_plate(video, lo, hi):
    cap = cv2.VideoCapture(video)
    frames = []
    for fi in range(lo, hi, 4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    ref = cv2.cvtColor(cv2.resize(frames[0], None, fx=0.5, fy=0.5),
                       cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    out = [frames[0]]
    warp = np.eye(2, 3, dtype=np.float32)
    mx = 0.0
    for fr in frames[1:]:
        g = cv2.cvtColor(cv2.resize(fr, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        try:
            _, warp = cv2.findTransformECC(ref, g, warp, cv2.MOTION_TRANSLATION, crit)
            dx, dy = float(warp[0, 2]) * 2, float(warp[1, 2]) * 2
        except cv2.error:
            dx, dy = 0.0, 0.0
        mx = max(mx, abs(dx), abs(dy))
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out.append(cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0])))
    plate = np.median(np.stack(out), axis=0).astype(np.uint8)
    print(f"plate from {len(out)} ECC-stabilized frames ({lo}-{hi}), "
          f"max drift {mx:.1f}px")
    return plate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--lo", type=int, required=True)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plate = stabilized_plate(args.video, args.lo, args.hi)
    cv2.imwrite(str(OUT_DIR / "plate_fit.png"), plate)

    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    h, w = plate.shape[:2]

    grass = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(grass)
    big = 1 + int(np.argmax(stats[1:, 4]))
    comp = (labels == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    court_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(court_mask, cnts, -1, 255, -1)
    cv2.imwrite(str(OUT_DIR / "hull_mask.png"), court_mask)

    v = hsv[:, :, 2]
    tophat = cv2.morphologyEx(
        v, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K)))
    hull_er = cv2.erode(court_mask, np.ones((HULL_ERODE, HULL_ERODE), np.uint8))
    line_mask = ((tophat > TOPHAT_T) & (v > V_MIN)).astype(np.uint8) * 255
    line_mask = cv2.bitwise_and(line_mask, hull_er)
    cv2.imwrite(str(OUT_DIR / "white_mask.png"), line_mask)

    if args.manual:
        H = cv2.getPerspectiveTransform(MODEL_PTS_MANUAL, CORNERS_IMG_MANUAL).astype(np.float64)
        fit_pts = CORNERS_IMG_MANUAL
    else:
        segs = cv2.HoughLinesP(line_mask, 1, np.pi / 360, threshold=80,
                               minLineLength=60, maxLineGap=12)
        segs = segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4), int)
        angles = np.array([seg_angle(s) for s in segs])
        horiz = segs[(angles < 30) | (angles > 150)]
        vert = segs[(angles >= 30) & (angles <= 150)]
        print(f"auto fit: {len(horiz)} horizontal segments, {len(vert)} vertical")

        hy = np.array([line_y_at(fit_line([s[:2], s[2:]]), w / 2) for s in horiz])
        vx = np.array([line_x_at(fit_line([s[:2], s[2:]]), h * 0.5) for s in vert])
        h_clusters = [c for c in cluster_1d(hy, tol=15) if len(c) >= MIN_SUPPORT]
        v_clusters = [c for c in cluster_1d(vx, tol=25) if len(c) >= MIN_SUPPORT]
        h_pos = [float(np.mean(hy[c])) for c in h_clusters]
        v_pos = [float(np.mean(vx[c])) for c in v_clusters]
        print(f"h clusters (>= {MIN_SUPPORT} segs) at y~{[round(p) for p in h_pos]}")
        print(f"v clusters (>= {MIN_SUPPORT} segs) at x~{[round(p) for p in v_pos]}")

        def merged_line(seg_group):
            pts = np.concatenate([[s[:2], s[2:]] for s in seg_group])
            return fit_line(pts)

        # ---- verticals: labeled by symmetric pairing about the center
        # cluster (widest pair = doubles, next = singles) ----
        ci = int(np.argmin([abs(p - w / 2) for p in v_pos]))
        cx_center = v_pos[ci]
        pairs = []
        for i in range(len(v_pos)):
            for j in range(i + 1, len(v_pos)):
                if abs((v_pos[i] + v_pos[j]) / 2 - cx_center) <= SYM_TOL:
                    pairs.append((v_pos[j] - v_pos[i], i, j))
        if not pairs:
            raise SystemExit("no symmetric vertical pair found — see white_mask.png")
        pairs.sort(reverse=True)
        v_label = {5.485: merged_line(vert[v_clusters[ci]])}
        _, li, ri = pairs[0]
        v_label[0.0] = merged_line(vert[v_clusters[li]])
        v_label[10.97] = merged_line(vert[v_clusters[ri]])
        msg = (f"center x~{cx_center:.1f}; doubles x~{v_pos[li]:.1f}/{v_pos[ri]:.1f}")
        if len(pairs) > 1:
            _, si, sj = pairs[1]
            v_label[1.372] = merged_line(vert[v_clusters[si]])
            v_label[9.598] = merged_line(vert[v_clusters[sj]])
            msg += f"; singles x~{v_pos[si]:.1f}/{v_pos[sj]:.1f}"
        print(msg)

        # ---- horizontals: the 4-point far-line heuristic died here (an
        # off-axis camera + mild lens distortion make single-line errors
        # explode through a 4-point basis). Instead: enumerate every
        # order-preserving assignment of >=3 of the 4 model horizontals
        # onto the measured clusters, LS-fit each on ALL labeled
        # intersections, and let reprojection RMS choose. ----
        # scorer: a wrong-but-self-consistent assignment can LS-fit its
        # own intersections to rms ~2 px (it happened: [175,368,473] as
        # baseline/service/baseline), so fitting error cannot choose.
        # Mask coverage can: the true assignment must land EVERY model
        # line on observed white; impostors strand the lines they
        # didn't use. Score = mean clipped dist of all 9 model lines.
        dist_map = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)

        def mask_score(Hk):
            errs = []
            for name in ("baseline_far", "service_far", "service_near",
                         "baseline_near", "singles_left", "singles_right",
                         "doubles_left", "doubles_right", "center_service"):
                p, q = MODEL_LINES[name]
                ts = np.linspace(0.05, 0.95, 40)
                pts = np.float32([(p[0] + t * (q[0] - p[0]),
                                   p[1] + t * (q[1] - p[1])) for t in ts])
                proj = cv2.perspectiveTransform(
                    pts.reshape(-1, 1, 2), Hk).reshape(-1, 2)
                ok = ((proj[:, 0] >= 0) & (proj[:, 0] < w)
                      & (proj[:, 1] >= 0) & (proj[:, 1] < h))
                if ok.sum() < 10:
                    return float("inf")
                d = dist_map[proj[ok, 1].astype(int), proj[ok, 0].astype(int)]
                errs.append(np.clip(d, 0, 25).mean())
            return float(np.mean(errs))

        from itertools import combinations
        H_YS = [0.0, SERVICE_FAR_Y, 18.285, L_COURT]
        h_lines = [merged_line(horiz[c]) for c in h_clusters]
        best = None
        for r in (4, 3):
            for model_sub in combinations(H_YS, r):
                for clus_sub in combinations(range(len(h_clusters)), r):
                    src, dst = [], []
                    for my_, k in zip(model_sub, clus_sub):
                        for mx_, vl in v_label.items():
                            src.append((mx_, my_))
                            dst.append(intersect(h_lines[k], vl))
                    Hk, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
                    if Hk is None:
                        continue
                    proj = cv2.perspectiveTransform(
                        np.float32(src).reshape(-1, 1, 2), Hk).reshape(-1, 2)
                    rms = float(np.sqrt(np.mean(np.sum(
                        (proj - np.float32(dst)) ** 2, axis=1))))
                    score = mask_score(Hk)
                    if best is None or score < best[0]:
                        best = (score, rms, Hk.astype(np.float64),
                                model_sub, clus_sub, len(src))
        score, rms, H, model_sub, clus_sub, npts = best
        print(f"assignment mask-score {score:.1f} px")
        print(f"horizontal assignment: model y {list(model_sub)} -> cluster y "
              f"{[round(h_pos[k]) for k in clus_sub]}  "
              f"(LS on {npts} intersections, rms {rms:.1f} px)")
        fit_pts = np.float32([
            intersect(h_lines[clus_sub[0]], v_label[0.0]),
            intersect(h_lines[clus_sub[0]], v_label[10.97]),
            intersect(h_lines[clus_sub[-1]], v_label[0.0]),
            intersect(h_lines[clus_sub[-1]], v_label[10.97]),
        ])

    corner_crops(plate, fit_pts)

    # ---- cross-checks + residual report ----
    def measured(lines_, at, fn):
        vals = np.array([fn(fit_line([s[:2], s[2:]]), at) for s in lines_])
        return [round(float(np.mean(vals[c])), 1) for c in cluster_1d(vals, tol=15)]

    segs_t = cv2.HoughLinesP(line_mask, 1, np.pi / 360, threshold=80,
                             minLineLength=60, maxLineGap=12)
    segs_t = segs_t.reshape(-1, 4) if segs_t is not None else np.zeros((0, 4), int)
    ang_t = np.array([seg_angle(s) for s in segs_t])
    horiz_t = segs_t[(ang_t < 30) | (ang_t > 150)]
    vert_t = segs_t[(ang_t >= 30) & (ang_t <= 150)]
    hy_t = measured(horiz_t, w / 2, line_y_at) if len(horiz_t) else []
    vx_t = measured(vert_t, h * 0.55, line_x_at) if len(vert_t) else []
    print(f"line-mask horizontals at mid-column y = {hy_t}")
    print(f"line-mask verticals   at mid-row    x = {vx_t}")

    mid_far = (fit_pts[0][0] + fit_pts[1][0]) / 2
    print(f"cross-check: far-fit-point midpoint x = {mid_far:.1f} "
          f"(must ~= measured center-service x)")
    span_far = fit_pts[1][0] - fit_pts[0][0]
    span_near = fit_pts[3][0] - fit_pts[2][0]
    print(f"cross-check: doubles span far {span_far:.0f}px @y~{fit_pts[0][1]:.0f}, "
          f"near {span_near:.0f}px @y~{fit_pts[2][1]:.0f} "
          f"({span_far / W_COURT:.1f} / {span_near / W_COURT:.1f} px/m)")

    dist_all = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)
    for name in ["singles_left", "singles_right", "baseline_far",
                 "service_near", "service_far", "center_service"]:
        p, q = MODEL_LINES[name]
        ts = np.linspace(0, 1, 50)
        pts = np.float32([(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
                          for t in ts])
        proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        inside = ((proj[:, 0] >= 0) & (proj[:, 0] < w)
                  & (proj[:, 1] >= 0) & (proj[:, 1] < h))
        d = dist_all[proj[inside, 1].astype(int), proj[inside, 0].astype(int)]
        print(f"residual {name}: mean dist to line px = {d.mean():.1f} px "
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
