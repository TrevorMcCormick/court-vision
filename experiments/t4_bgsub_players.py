"""T4 — $0 player tracking: m3_bgsub_players, t4 paths.

SAM tracked the players at $0.15/clip; 60 clips is $9 of what the clean
plate gives away free (M1: the temporal median erases players, so
frame-minus-plate IS a player detector). Getting it to actually work
took three findings:

  1. The "static" camera pans. ECC alignment shows up to ~32 px of REAL
     mid-point camera movement (the operator follows play), plus slow
     drift. Invisible watching the video, fatal to a median plate and to
     far-court coordinates. Every frame is ECC-stabilized (translation)
     to its clip's frame 0 before anything else touches it.
  2. The per-clip plate contains a ghost of the far player. He's small
     and stands nearly still between shots, so the median bakes him in
     and his diff goes ~zero (near player: 100% coverage; far: ~35%).
     Fix: each clip subtracts a DEEP plate — the median of its own plate
     and its neighbors' plates (aligned by ECC). Players stand in
     different spots point to point; ghosts wash out.
  3. Same impostor tricks as the SAM split: court-region polygon (crowd
     never enters), per-clip static-pixel erase (line judges self-erase),
     one component per half.

Pass A per clip: stabilize -> per-clip plate + shifts (cached to disk).
Pass B per clip: deep plate from neighbors -> subtract -> tracks CSV
                 (same schema as SAM's players_traj.csv).

Usage:
    uv run experiments/t4_bgsub_players.py --pass-a [clips...]
    uv run experiments/t4_bgsub_players.py --pass-b [clips...]
    (no flag: both passes; no clips: all of clips/points_t4/)
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "t4"
PLATE_DIR = OUT_BASE / "plates"
TRACK_DIR = OUT_BASE / "players"
ROOT = Path(__file__).resolve().parent.parent

DIFF_T = 30
MIN_AREA_NEAR = 220
MIN_AREA_FAR = 110
STATIC_FRAC = 0.90
ECC_DS = 2
NEIGHBORS = 2            # plates on each side for the deep plate
W_C, L_C = 10.97, 23.77


def court_region_mask(shape):
    """Two polygons: generous for the near half, tight for the far half.

    The far corners host crouching ballkids ~2.5 m outside the doubles
    lines; with a generous margin they out-diff the motionless pre-serve
    far player and hijack his track (court x ~13.5 or -3 in the serve
    gate log — outside the court entirely). The far player himself rarely
    plays from >1.5 m beyond the doubles line at this end.
    """
    Hc2i = np.load(ROOT / "outputs/t4/H_court_to_img.npy")

    def poly_mask(pts):
        img = cv2.perspectiveTransform(
            np.float32(pts).reshape(-1, 1, 2), Hc2i).reshape(-1, 2).astype(np.int32)
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [img], 1)
        return m

    near = poly_mask([[-3.5, L_C / 2], [W_C + 3.5, L_C / 2],
                      [W_C + 4.5, L_C + 4.5], [-4.5, L_C + 4.5]])
    far = poly_mask([[-1.5, -5.0], [W_C + 1.5, -5.0],
                     [W_C + 1.5, L_C / 2], [-1.5, L_C / 2]])
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


def pass_a(path):
    frames, fps = read_frames(path)
    frames, shifts = stabilize(frames)
    n = len(frames)
    plate = np.median(np.stack(frames[::max(1, n // 60)]), axis=0).astype(np.uint8)
    cv2.imwrite(str(PLATE_DIR / f"plate_{path.stem}.png"), plate)
    with open(PLATE_DIR / f"shifts_{path.stem}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "dx", "dy"])
        for i, (dx, dy) in enumerate(shifts):
            wr.writerow([i, round(dx, 2), round(dy, 2)])
    mx = max(max(abs(a), abs(b)) for a, b in shifts)
    print(f"  pass A {path.stem}: {n} fr, drift<= {mx:.1f}px")


def deep_plate(stem, all_stems):
    k = all_stems.index(stem)
    own = cv2.imread(str(PLATE_DIR / f"plate_{stem}.png"))
    ref = cv2.cvtColor(cv2.resize(own, None, fx=1 / ECC_DS, fy=1 / ECC_DS),
                       cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    stack = [own]
    for j in range(max(0, k - NEIGHBORS), min(len(all_stems), k + NEIGHBORS + 1)):
        if j == k:
            continue
        nb = cv2.imread(str(PLATE_DIR / f"plate_{all_stems[j]}.png"))
        g = cv2.cvtColor(cv2.resize(nb, None, fx=1 / ECC_DS, fy=1 / ECC_DS),
                         cv2.COLOR_BGR2GRAY)
        try:
            dx, dy, _ = ecc_translation(ref, g, np.eye(2, 3, dtype=np.float32), crit)
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            stack.append(cv2.warpAffine(nb, M, (nb.shape[1], nb.shape[0])))
        except cv2.error:
            continue
    return np.median(np.stack(stack), axis=0).astype(np.uint8), len(stack)


def pass_b(path, region, all_stems):
    frames, fps = read_frames(path)
    shifts = {}
    with open(PLATE_DIR / f"shifts_{path.stem}.csv") as f:
        for row in csv.DictReader(f):
            shifts[int(row["frame"])] = (float(row["dx"]), float(row["dy"]))
    for i in range(len(frames)):
        dx, dy = shifts.get(i, (0, 0))
        if dx or dy:
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            frames[i] = cv2.warpAffine(frames[i], M,
                                       (frames[i].shape[1], frames[i].shape[0]))
    plate, depth = deep_plate(path.stem, all_stems)

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
    out = TRACK_DIR / f"players_{path.stem}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    n = len(frames)
    near = sum(1 for r in rows if r["player"] == "near")
    far = sum(1 for r in rows if r["player"] == "far")
    print(f"  pass B {path.stem}: deep plate x{depth}  near {near}/{n}  far {far}/{n}")
    return near / n, far / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="*")
    parser.add_argument("--pass-a", action="store_true")
    parser.add_argument("--pass-b", action="store_true")
    args = parser.parse_args()

    paths = ([Path(p) for p in args.clips] if args.clips
             else sorted((ROOT / "clips/points_t4").glob("t4_point_*.mp4")))
    all_paths = sorted((ROOT / "clips/points_t4").glob("t4_point_*.mp4"))
    all_stems = [p.stem for p in all_paths]
    PLATE_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_DIR.mkdir(parents=True, exist_ok=True)

    do_a = args.pass_a or not args.pass_b
    do_b = args.pass_b or not args.pass_a

    if do_a:
        for p in paths:
            pass_a(p)
    if do_b:
        probe = cv2.VideoCapture(str(paths[0]))
        ok, fr0 = probe.read()
        region = court_region_mask(fr0.shape)
        covs = []
        for p in paths:
            covs.append(pass_b(p, region, all_stems))
        if len(covs) > 1:
            print(f"coverage: near median {np.median([c[0] for c in covs]):.0%}, "
                  f"far median {np.median([c[1] for c in covs]):.0%}")


if __name__ == "__main__":
    main()
