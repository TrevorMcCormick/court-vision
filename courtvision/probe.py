"""Court-view detection over the reel — the t3/t4 probe, config-driven.

The discriminator that survived every surface so far: interior probes
read the COURT COLOR (the match's hull band, dilated so a probe on a
blown-out white line still counts), and LINE probes read the tophat
thin-bright mask along the projected model lines — geometry a close-up
cannot fake. The apron test is dead (clay apron IS clay, grass run-off
IS grass, and the AO apron is blue like the court); the line test
replaced it.

Pan tolerance (the t3 lesson): a fixed projection walks the line probes
off the real lines, so the line read is a MAX over a small grid of
probe shifts; the winning shift is recorded per frame — downstream
stages need it to map back to the fit camera's coordinates
(clip_offsets.csv, written by courtvision.extract).

Reads court_detect from the match YAML (same hull band as the fit).
Writes outputs/<id>/view_probe.csv + segments.csv.

Usage:
    uv run python -m courtvision probe t5
"""

import csv

import cv2
import numpy as np

from .court import W_C, L_C

DS = 2  # probe at half res

INT_MIN = 0.80
LINE_MIN = 0.35
SHIFT_DX = range(-12, 13, 2)   # half-res px
SHIFT_DY = range(-8, 9, 2)
MIN_RUN_S = 3.0
GAP_S = 0.6
SMOOTH = 9
TOPHAT_K = 13          # 25 at full res
TOPHAT_T = 25
V_MIN = 170

PROBE_LINES = {         # far baseline excluded: buried/occluded on some feeds
    "baseline_near": ((0, 23.77), (10.97, 23.77)),
    "singles_left": ((1.372, 0), (1.372, 23.77)),
    "singles_right": ((9.598, 0), (9.598, 23.77)),
    "doubles_left": ((0, 0), (0, 23.77)),
    "doubles_right": ((10.97, 0), (10.97, 23.77)),
    "service_near": ((1.372, 18.285), (9.598, 18.285)),
    "service_far": ((1.372, 5.485), (9.598, 5.485)),
    "center_service": ((5.485, 5.485), (5.485, 18.285)),
}
PTS_PER_LINE = 12


def probe_points():
    interior = [(x, y)
                for x in np.linspace(1.5, W_C - 1.5, 5)
                for y in np.linspace(2.0, L_C - 2.0, 10)]
    line_pts = []
    for (p, q) in PROBE_LINES.values():
        for t in np.linspace(0.06, 0.94, PTS_PER_LINE):
            line_pts.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return np.float32(interior), np.float32(line_pts)


def probe_match(cfg):
    cd = cfg.court_detect
    hull_lo = tuple(cd["hull_lo"])
    hull_hi = tuple(cd["hull_hi"])
    v_min = int(cd.get("v_min", V_MIN))
    out_dir = cfg.out_dir

    Hc2i = np.load(out_dir / "H_court_to_img.npy")
    interior, line_pts = probe_points()

    def project(pts):
        img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hc2i).reshape(-1, 2)
        return (img / DS).astype(int)

    p_int = project(interior)
    p_lin = project(line_pts)

    cap = cv2.VideoCapture(str(cfg.video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // DS
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // DS
    ok_int = (p_int[:, 0] >= 0) & (p_int[:, 0] < w) & (p_int[:, 1] >= 0) & (p_int[:, 1] < h)
    ok_lin = (p_lin[:, 0] >= 0) & (p_lin[:, 0] < w) & (p_lin[:, 1] >= 0) & (p_lin[:, 1] < h)
    p_int, p_lin = p_int[ok_int], p_lin[ok_lin]
    print(f"{n_total} frames @ {fps:.2f} fps; probes: {len(p_int)} interior, "
          f"{len(p_lin)} line")

    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K))
    shift_grid = [(dx, dy) for dx in SHIFT_DX for dy in SHIFT_DY]
    int_hits, line_hits, shifts = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        court = cv2.inRange(hsv, hull_lo, hull_hi)
        court_dil = cv2.dilate(court, np.ones((7, 7), np.uint8))
        v = hsv[:, :, 2]
        tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kern)
        lines = ((tophat > TOPHAT_T) & (v > v_min)).astype(np.uint8) * 255
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        int_hits.append(float(np.mean(court_dil[p_int[:, 1], p_int[:, 0]] > 0)))
        best_hit, best_sh = 0.0, (0, 0)
        for dx, dy in shift_grid:
            pp = p_lin + [dx, dy]
            okm = ((pp[:, 0] >= 0) & (pp[:, 0] < w)
                   & (pp[:, 1] >= 0) & (pp[:, 1] < h))
            if okm.sum() < len(pp) * 0.8:
                continue
            hit = float(np.mean(lines[pp[okm, 1], pp[okm, 0]] > 0))
            if hit > best_hit:
                best_hit, best_sh = hit, (dx, dy)
        line_hits.append(best_hit)
        shifts.append(best_sh)
        i += 1
        if i % 10000 == 0:
            print(f"  {i}/{n_total}")

    int_hits = np.array(int_hits)
    line_hits = np.array(line_hits)
    raw = (int_hits > INT_MIN) & (line_hits > LINE_MIN)

    k = SMOOTH
    pad = np.pad(raw.astype(np.uint8), k // 2, mode="edge")
    smooth = np.array([np.median(pad[j:j + k]) for j in range(len(raw))]).astype(bool)

    segs = []
    start = None
    for j, val in enumerate(smooth):
        if val and start is None:
            start = j
        elif not val and start is not None:
            segs.append([start, j - 1])
            start = None
    if start is not None:
        segs.append([start, len(smooth) - 1])
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) / fps <= GAP_S:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    keep = [s for s in merged if (s[1] - s[0] + 1) / fps >= MIN_RUN_S]

    with open(out_dir / "view_probe.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "int_hit", "line_hit", "shift_dx", "shift_dy",
                     "court_view"])
        for j in range(len(raw)):
            wr.writerow([j, round(int_hits[j], 3), round(line_hits[j], 3),
                         shifts[j][0] * DS, shifts[j][1] * DS, int(smooth[j])])
    with open(out_dir / "segments.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seg", "start_frame", "end_frame", "start_s", "dur_s"])
        for k2, (a, b) in enumerate(keep, 1):
            wr.writerow([k2, a, b, round(a / fps, 2), round((b - a + 1) / fps, 2)])

    total_s = sum((b - a + 1) for a, b in keep) / fps
    print(f"court-view frames (smoothed): {int(smooth.sum())} ({smooth.mean():.1%})")
    print(f"segments >= {MIN_RUN_S}s: {len(keep)}, total {total_s:.0f}s")
    for k2, (a, b) in enumerate(keep, 1):
        print(f"  seg {k2:>2}: f{a}-f{b}  {a/fps:7.1f}s  dur {(b-a+1)/fps:5.1f}s")
    print(f"-> {out_dir / 'view_probe.csv'}, {out_dir / 'segments.csv'}")
