"""Fit the court homography from the reel — the t3/t4 recipe, config-driven.

The per-surface constants the t*_fit_homography twins carried in code
(hull HSV band, fit-run frame window) move into the match YAML
(court_detect); the recipe itself is the one that survived clay AND
grass, so it is the shared logic:

  1. ECC-stabilized median plate over a hand-picked court-view run
     (an unstabilized plate smears lines into the surface — t3 lesson).
  2. Court hull: configured HSV band, largest connected component,
     FILLED contour (a convex hull bridges into the crowd — t3 lesson).
  3. Line mask: thin-bright TOPHAT inside the eroded hull (absolute
     white dies on dusty/worn lines — t3 lesson).
  4. Verticals labeled by symmetric pairing about the center-service
     cluster (apron junk is asymmetric, court lines aren't).
  5. Horizontals: enumerate every order-preserving assignment of >=3 of
     the 4 model horizontals onto the measured clusters, LS-fit each on
     all labeled intersections, scored by how well ALL NINE model lines
     land on the observed line mask (fit rms cannot choose — a wrong-
     but-self-consistent triple fit its own points to rms 1.7 px on t4).
  6. Arithmetic cross-checks printed, reprojection overlay written —
     residuals can lie, overlays cannot: LOOK at model_reprojection.png.

court_detect keys (data/matches/<id>.yaml):
  hull_lo / hull_hi   HSV band that reads the COURT surface and not the
                      apron (measure from real frames first; on a blue-
                      on-blue court S or V carries the split, not H)
  fit_lo / fit_hi     frame window of a clean, full-court view run

Manual fallback: --manual reads court_detect.manual_corners
(farL, farR, nearL, nearR image px) — the t1 gridded-crop recipe;
corner_*.png crops are written for the reads either way.

Usage:
    uv run python -m courtvision fitcourt t5 [--manual]
"""

from itertools import combinations

import cv2
import numpy as np

from .court import W_C, L_C

SERVICE_FAR_Y = 5.485
SINGLES_INSET = 1.372

MODEL_LINES = {
    "baseline_far": ((0, 0), (W_C, 0)),
    "baseline_near": ((0, L_C), (W_C, L_C)),
    "doubles_left": ((0, 0), (0, L_C)),
    "doubles_right": ((W_C, 0), (W_C, L_C)),
    "singles_left": ((SINGLES_INSET, 0), (SINGLES_INSET, L_C)),
    "singles_right": ((W_C - SINGLES_INSET, 0), (W_C - SINGLES_INSET, L_C)),
    "service_far": ((SINGLES_INSET, SERVICE_FAR_Y), (W_C - SINGLES_INSET, SERVICE_FAR_Y)),
    "service_near": ((SINGLES_INSET, L_C - SERVICE_FAR_Y), (W_C - SINGLES_INSET, L_C - SERVICE_FAR_Y)),
    "center_service": ((W_C / 2, SERVICE_FAR_Y), (W_C / 2, L_C - SERVICE_FAR_Y)),
}

TOPHAT_K = 25
TOPHAT_T = 25
V_MIN = 170
HULL_ERODE = 15
SYM_TOL = 12
MIN_SUPPORT = 3


def seg_angle(s):
    x1, y1, x2, y2 = s
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def fit_line(points):
    pts = np.asarray(points, dtype=np.float64)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return n[0], n[1], -n @ mean


def line_x_at(line, y):
    a, b, c = line
    return (-c - b * y) / a


def line_y_at(line, x):
    a, b, c = line
    return (-c - a * x) / b


def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    return np.array([(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d])


def cluster_1d(values, tol):
    order = np.argsort(values)
    clusters = [[order[0]]]
    for idx in order[1:]:
        if values[idx] - values[clusters[-1][-1]] <= tol:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def stabilized_plate(video, lo, hi):
    cap = cv2.VideoCapture(str(video))
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


def hull_and_lines(plate, cd, out_dir):
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    h, w = plate.shape[:2]
    hull = cv2.inRange(hsv, tuple(cd["hull_lo"]), tuple(cd["hull_hi"]))
    if "hull2_lo" in cd:      # court + apron of a DIFFERENT color family
        hull |= cv2.inRange(hsv, tuple(cd["hull2_lo"]), tuple(cd["hull2_hi"]))
    hull = cv2.morphologyEx(hull, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hull)
    big = 1 + int(np.argmax(stats[1:, 4]))
    comp = (labels == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    court_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(court_mask, cnts, -1, 255, -1)
    cv2.imwrite(str(out_dir / "hull_mask.png"), court_mask)

    v = hsv[:, :, 2]
    tophat = cv2.morphologyEx(
        v, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K)))
    hull_er = cv2.erode(court_mask, np.ones((HULL_ERODE, HULL_ERODE), np.uint8))
    v_min = int(cd.get("v_min", V_MIN))
    tophat_t = int(cd.get("tophat_t", TOPHAT_T))
    line_mask = ((tophat > tophat_t) & (v > v_min)).astype(np.uint8) * 255
    line_mask = cv2.bitwise_and(line_mask, hull_er)
    cv2.imwrite(str(out_dir / "white_mask.png"), line_mask)
    return court_mask, line_mask


def corner_crops(plate, corners, out_dir, zoom=6):
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
        cv2.imwrite(str(out_dir / f"corner_{name}.png"), big)
    print(f"-> corner crops in {out_dir}")


def auto_fit(plate, line_mask):
    h, w = plate.shape[:2]
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

    # verticals: symmetric pairing about the center cluster
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
    v_label = {W_C / 2: merged_line(vert[v_clusters[ci]])}
    _, li, ri = pairs[0]
    v_label[0.0] = merged_line(vert[v_clusters[li]])
    v_label[W_C] = merged_line(vert[v_clusters[ri]])
    msg = f"center x~{cx_center:.1f}; doubles x~{v_pos[li]:.1f}/{v_pos[ri]:.1f}"
    if len(pairs) > 1:
        _, si, sj = pairs[1]
        v_label[SINGLES_INSET] = merged_line(vert[v_clusters[si]])
        v_label[W_C - SINGLES_INSET] = merged_line(vert[v_clusters[sj]])
        msg += f"; singles x~{v_pos[si]:.1f}/{v_pos[sj]:.1f}"
    print(msg)

    # horizontals: mask-scored assignment enumeration
    dist_map = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)

    def mask_score(Hk):
        errs = []
        for name in MODEL_LINES:
            p, q = MODEL_LINES[name]
            ts = np.linspace(0.05, 0.95, 40)
            pts = np.float32([(p[0] + t * (q[0] - p[0]),
                               p[1] + t * (q[1] - p[1])) for t in ts])
            proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hk).reshape(-1, 2)
            ok = ((proj[:, 0] >= 0) & (proj[:, 0] < w)
                  & (proj[:, 1] >= 0) & (proj[:, 1] < h))
            if ok.sum() < 10:
                return float("inf")
            d = dist_map[proj[ok, 1].astype(int), proj[ok, 0].astype(int)]
            errs.append(np.clip(d, 0, 25).mean())
        return float(np.mean(errs))

    H_YS = [0.0, SERVICE_FAR_Y, L_C - SERVICE_FAR_Y, L_C]
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
        intersect(h_lines[clus_sub[0]], v_label[W_C]),
        intersect(h_lines[clus_sub[-1]], v_label[0.0]),
        intersect(h_lines[clus_sub[-1]], v_label[W_C]),
    ])
    return H, fit_pts


def validate(plate, line_mask, H, fit_pts, out_dir):
    h, w = plate.shape[:2]
    segs_t = cv2.HoughLinesP(line_mask, 1, np.pi / 360, threshold=80,
                             minLineLength=60, maxLineGap=12)
    segs_t = segs_t.reshape(-1, 4) if segs_t is not None else np.zeros((0, 4), int)
    ang_t = np.array([seg_angle(s) for s in segs_t])
    horiz_t = segs_t[(ang_t < 30) | (ang_t > 150)]
    vert_t = segs_t[(ang_t >= 30) & (ang_t <= 150)]

    def measured(lines_, at, fn):
        vals = np.array([fn(fit_line([s[:2], s[2:]]), at) for s in lines_])
        return [round(float(np.mean(vals[c])), 1) for c in cluster_1d(vals, tol=15)]

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
          f"({span_far / W_C:.1f} / {span_near / W_C:.1f} px/m)")

    dist_all = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)
    for name, (p, q) in MODEL_LINES.items():
        ts = np.linspace(0, 1, 50)
        pts = np.float32([(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
                          for t in ts])
        proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        inside = ((proj[:, 0] >= 0) & (proj[:, 0] < w)
                  & (proj[:, 1] >= 0) & (proj[:, 1] < h))
        if inside.sum() < 5:
            continue
        d = dist_all[proj[inside, 1].astype(int), proj[inside, 0].astype(int)]
        print(f"residual {name}: mean dist to line px = {d.mean():.1f} px "
              f"(max {d.max():.1f})")

    overlay = plate.copy()
    for name, (p, q) in MODEL_LINES.items():
        pts = np.float32([p, q]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        color = (0, 255, 255) if "baseline" in name or "doubles" in name else (0, 128, 255)
        cv2.line(overlay, tuple(proj[0].astype(int)), tuple(proj[1].astype(int)), color, 2)
    cv2.imwrite(str(out_dir / "model_reprojection.png"), overlay)
    print(f"-> {out_dir}/model_reprojection.png — LOOK AT IT before proceeding")


def fit_match(cfg, manual=False, net=False):
    if net:
        # neural auto-fit + paint referee, straight off the staged plate
        # (no video re-read); abstains to the hand fit when it can't win
        from . import courtfit_auto
        v = courtfit_auto.gated_fit(cfg)
        print(f"[gate] {cfg.id}: {v['decision']}  "
              f"(neural {v['neural_score']} vs hand {v['hand_score']} px, "
              f"kps {v['n_kps']}/14, landmark Δ {v['landmark_delta_px']}px)")
        print(f"-> {cfg.out_dir}/H_*.npy (chosen: {v['chosen']}); "
              f"overlay {v['overlay']}")
        return v

    cd = cfg.court_detect
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plate = stabilized_plate(cfg.video, int(cd["fit_lo"]), int(cd["fit_hi"]))
    cv2.imwrite(str(out_dir / "plate_fit.png"), plate)
    _, line_mask = hull_and_lines(plate, cd, out_dir)

    if manual:
        model_pts = np.float32([(0, 0), (W_C, 0), (0, L_C), (W_C, L_C)])
        fit_pts = np.float32(cd["manual_corners"])
        H = cv2.getPerspectiveTransform(model_pts, fit_pts).astype(np.float64)
    else:
        H, fit_pts = auto_fit(plate, line_mask)

    corner_crops(plate, fit_pts, out_dir)
    validate(plate, line_mask, H, fit_pts, out_dir)
    np.save(out_dir / "H_court_to_img.npy", H)
    np.save(out_dir / "H_img_to_court.npy", np.linalg.inv(H))
    print(f"-> {out_dir}/H_court_to_img.npy, H_img_to_court.npy")
