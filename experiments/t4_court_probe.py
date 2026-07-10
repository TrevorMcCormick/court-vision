"""T4 — court-view detection on grass: twin of the t3 clay probe.

Grass kills the t1/t2 apron test the same way clay did: the run-off IS
grass, so "apron reads a different color" never fires. Same redesign:

  interior probes — the t2 grid projected into the image must read
                    grass (dilated mask; the worn baseline browns at
                    H 15-19 are inside the band on purpose)
  line probes     — points along the projected model lines must read
                    thin-bright (tophat mask). Geometry, not color.

court view  <=>  interior >= INT_MIN and lines >= LINE_MIN,
with the t3 pan-tolerant shift search (max over a +-24/+-16 px grid),
winning shift recorded per frame for downstream coordinate mapping.

Usage:
    uv run experiments/t4_court_probe.py clips/t4_krejcikova_paolini_30fps.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "t4"
ROOT = Path(__file__).resolve().parent.parent

GRASS_LO, GRASS_HI = (10, 25, 70), (50, 255, 255)
W_C, L_C = 10.97, 23.77
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

PROBE_LINES = {         # model lines sampled for the line probes
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    Hc2i = np.load(ROOT / "outputs/t4/H_court_to_img.npy")
    interior, line_pts = probe_points()

    def project(pts):
        img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hc2i).reshape(-1, 2)
        return (img / DS).astype(int)

    p_int = project(interior)
    p_lin = project(line_pts)

    cap = cv2.VideoCapture(args.video)
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
        clay = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
        clay_dil = cv2.dilate(clay, np.ones((7, 7), np.uint8))
        v = hsv[:, :, 2]
        tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kern)
        lines = ((tophat > TOPHAT_T) & (v > V_MIN)).astype(np.uint8) * 255
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        int_hits.append(float(np.mean(clay_dil[p_int[:, 1], p_int[:, 0]] > 0)))
        # pan tolerance: best line read over the shift grid
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
        if i % 5000 == 0:
            print(f"  {i}/{n_total}")

    int_hits = np.array(int_hits)
    line_hits = np.array(line_hits)
    raw = (int_hits > INT_MIN) & (line_hits > LINE_MIN)

    # median smooth
    k = SMOOTH
    pad = np.pad(raw.astype(np.uint8), k // 2, mode="edge")
    smooth = np.array([np.median(pad[j:j + k]) for j in range(len(raw))]).astype(bool)

    # runs -> segments with gap merging
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

    with open(OUT_DIR / "view_probe.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "int_hit", "line_hit", "shift_dx", "shift_dy",
                     "court_view"])
        for j in range(len(raw)):
            wr.writerow([j, round(int_hits[j], 3), round(line_hits[j], 3),
                         shifts[j][0] * DS, shifts[j][1] * DS, int(smooth[j])])
    with open(OUT_DIR / "segments.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seg", "start_frame", "end_frame", "start_s", "dur_s"])
        for k2, (a, b) in enumerate(keep, 1):
            wr.writerow([k2, a, b, round(a / fps, 2), round((b - a + 1) / fps, 2)])

    total_s = sum((b - a + 1) for a, b in keep) / fps
    print(f"court-view frames (smoothed): {int(smooth.sum())} ({smooth.mean():.1%})")
    print(f"segments >= {MIN_RUN_S}s: {len(keep)}, total {total_s:.0f}s")
    for k2, (a, b) in enumerate(keep, 1):
        print(f"  seg {k2:>2}: f{a}-f{b}  {a/fps:7.1f}s  dur {(b-a+1)/fps:5.1f}s")
    print(f"-> {OUT_DIR / 'view_probe.csv'}, {OUT_DIR / 'segments.csv'}")


if __name__ == "__main__":
    main()
