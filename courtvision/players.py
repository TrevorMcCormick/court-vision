"""$0 player tracking — background subtraction against a deep median plate.

Lifted from the t*_bgsub_players.py scripts (frozen outputs). Two passes:

  A  ECC-stabilize each clip against its own frame 0, write the median
     plate and the per-frame shifts (the shifts also serve the ball
     tracker's x_stab/y_stab).
  B  diff each stabilized frame against a DEEP plate (median of the
     clip's plate and its reel neighbors' plates, ECC-aligned), keep
     the largest court-region component per half above a per-half area
     floor, erase static blobs (on > 90% of frames: net tape, bug).

Per-match knob (players_detect in the match YAML): where the two
court-half region polygons stop in court-y. Montreal/Wimbledon split at
the net line; RG excludes the net-tape band — the tape hangs ~1.07 m
above the ground plane so its back-projection lands at court y ~7-11.5,
and under residual camera jitter it out-areas the real far player.

The far corners host crouching ballkids ~2.5 m outside the doubles
lines; the far region is tight so they don't out-diff the motionless
pre-serve far player and hijack his track.

The SAM-3 alternative (players_sam/, same CSV schema) is the measured
buy option for the letter sink — experiments/t3_sam_players.py, ~$0.20/
clip via fal; the shipped default stays bgsub at $0. A match switches
sources via players_dir in its YAML.

Usage:
    uv run python -m courtvision players t3 [clips...] [--pass-a|--pass-b]
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from .court import W_C, L_C

DIFF_T = 30
MIN_AREA_NEAR = 220
MIN_AREA_FAR = 110
STATIC_FRAC = 0.90
ECC_DS = 2
NEIGHBORS = 2            # plates on each side for the deep plate


def court_region_mask(shape, Hc2i, near_top_m, far_bottom_m):
    """Two polygons: generous for the near half, tight for the far half."""
    def poly_mask(pts):
        img = cv2.perspectiveTransform(
            np.float32(pts).reshape(-1, 1, 2), Hc2i).reshape(-1, 2).astype(np.int32)
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [img], 1)
        return m

    near = poly_mask([[-3.5, near_top_m], [W_C + 3.5, near_top_m],
                      [W_C + 4.5, L_C + 4.5], [-4.5, L_C + 4.5]])
    far = poly_mask([[-1.5, -5.0], [W_C + 1.5, -5.0],
                     [W_C + 1.5, far_bottom_m], [-1.5, far_bottom_m]])
    return near | far


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    return frames, fps


def ecc_translation(ref_gray, gray, warp, crit):
    _, warp = cv2.findTransformECC(ref_gray, gray, warp, cv2.MOTION_TRANSLATION, crit)
    return float(warp[0, 2]) * ECC_DS, float(warp[1, 2]) * ECC_DS, warp


def stabilize(frames):
    ref = cv2.cvtColor(cv2.resize(frames[0], None, fx=1 / ECC_DS, fy=1 / ECC_DS),
                       cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    out, shifts = [frames[0]], [(0.0, 0.0)]
    warp = np.eye(2, 3, dtype=np.float32)
    for fr in frames[1:]:
        g = cv2.cvtColor(cv2.resize(fr, None, fx=1 / ECC_DS, fy=1 / ECC_DS),
                         cv2.COLOR_BGR2GRAY)
        try:
            dx, dy, warp = ecc_translation(ref, g, warp, crit)
        except cv2.error:
            dx, dy = shifts[-1]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out.append(cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0])))
        shifts.append((dx, dy))
    return out, shifts


def pass_a(cfg, path):
    plate_dir = cfg.out_dir / "plates"
    frames, fps = read_frames(path)
    frames, shifts = stabilize(frames)
    n = len(frames)
    plate = np.median(np.stack(frames[::max(1, n // 60)]), axis=0).astype(np.uint8)
    cv2.imwrite(str(plate_dir / f"plate_{path.stem}.png"), plate)
    with open(plate_dir / f"shifts_{path.stem}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "dx", "dy"])
        for i, (dx, dy) in enumerate(shifts):
            wr.writerow([i, round(dx, 2), round(dy, 2)])
    mx = max(max(abs(a), abs(b)) for a, b in shifts)
    print(f"  pass A {path.stem}: {n} fr, drift<= {mx:.1f}px")


def deep_plate(plate_dir, stem, all_stems):
    k = all_stems.index(stem)
    own = cv2.imread(str(plate_dir / f"plate_{stem}.png"))
    ref = cv2.cvtColor(cv2.resize(own, None, fx=1 / ECC_DS, fy=1 / ECC_DS),
                       cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    stack = [own]
    for j in range(max(0, k - NEIGHBORS), min(len(all_stems), k + NEIGHBORS + 1)):
        if j == k:
            continue
        nb = cv2.imread(str(plate_dir / f"plate_{all_stems[j]}.png"))
        g = cv2.cvtColor(cv2.resize(nb, None, fx=1 / ECC_DS, fy=1 / ECC_DS),
                         cv2.COLOR_BGR2GRAY)
        try:
            dx, dy, _ = ecc_translation(ref, g, np.eye(2, 3, dtype=np.float32), crit)
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            stack.append(cv2.warpAffine(nb, M, (nb.shape[1], nb.shape[0])))
        except cv2.error:
            continue
    return np.median(np.stack(stack), axis=0).astype(np.uint8), len(stack)


def pass_b(cfg, path, region, all_stems):
    plate_dir = cfg.out_dir / "plates"
    frames, fps = read_frames(path)
    shifts = {}
    with open(plate_dir / f"shifts_{path.stem}.csv") as f:
        for row in csv.DictReader(f):
            shifts[int(row["frame"])] = (float(row["dx"]), float(row["dy"]))
    for i in range(len(frames)):
        dx, dy = shifts.get(i, (0, 0))
        if dx or dy:
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            frames[i] = cv2.warpAffine(frames[i], M,
                                       (frames[i].shape[1], frames[i].shape[0]))
    plate, depth = deep_plate(plate_dir, path.stem, all_stems)

    masks = []
    for fr in frames:
        diff = cv2.absdiff(fr, plate).max(axis=2)
        m = ((diff > DIFF_T).astype(np.uint8)) & region
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        masks.append(m)

    on_frac = np.mean([m.astype(np.float32) for m in masks], axis=0)
    static = (on_frac > STATIC_FRAC).astype(np.uint8)

    Hh, Ww = frames[0].shape[:2]
    rows = []
    for fi, m in enumerate(masks):
        m = m & ~static
        nn, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        halves = {"near": [], "far": []}
        for lab in range(1, nn):
            x, y, w, h, area = stats[lab]
            side = "near" if y + h > Hh / 2 else "far"
            if area < (MIN_AREA_NEAR if side == "near" else MIN_AREA_FAR):
                continue
            halves[side].append((area, x, y, w, h))
        for side in ("near", "far"):
            if not halves[side]:
                continue
            area, x, y, w, h = max(halves[side])
            rows.append({"frame": fi, "player": side,
                         "cx": (x + w / 2) / Ww, "cy": (y + h / 2) / Hh,
                         "w": w / Ww, "h": h / Hh,
                         "foot_x": x + w / 2, "foot_y": y + h, "area": area})
    out = cfg.players_dir / f"players_{path.stem}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    n = len(frames)
    near = sum(1 for r in rows if r["player"] == "near")
    far = sum(1 for r in rows if r["player"] == "far")
    print(f"  pass B {path.stem}: deep plate x{depth}  near {near}/{n}  far {far}/{n}")
    return near / n, far / n


def track_match(cfg, stems=None, do_a=True, do_b=True):
    """Run pass A and/or B over the match's clips."""
    all_paths = sorted(cfg.clips_dir.glob("*.mp4"))
    paths = ([cfg.clip_path(s) for s in stems] if stems else all_paths)
    all_stems = [p.stem for p in all_paths]
    (cfg.out_dir / "plates").mkdir(parents=True, exist_ok=True)
    cfg.players_dir.mkdir(parents=True, exist_ok=True)

    if do_a:
        for p in paths:
            pass_a(cfg, p)
    if do_b:
        probe = cv2.VideoCapture(str(paths[0]))
        ok, fr0 = probe.read()
        Hc2i = np.load(cfg.out_dir / "H_court_to_img.npy")
        pd = cfg.players_detect
        region = court_region_mask(
            fr0.shape, Hc2i,
            float(pd.get("near_top_m", L_C / 2)),
            float(pd.get("far_bottom_m", L_C / 2)))
        covs = []
        for p in paths:
            covs.append(pass_b(cfg, p, region, all_stems))
        if len(covs) > 1:
            print(f"coverage: near median {np.median([c[0] for c in covs]):.0%}, "
                  f"far median {np.median([c[1] for c in covs]):.0%}")
