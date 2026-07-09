"""M3 experiment 1b — split the merged player mask back into two players.

What m3_track_players.py found: two box prompts in one video-rle call come
back as ONE merged mask per frame ('boxes' is their union), and the mask
also leaked onto the red-shirted line judge behind Gasquet — a figure that
was never prompted.

The salvage: the mask's connected components are spatially disjoint all
rally (baseline rally, nobody crosses anybody in image space). Split each
frame's mask into components, then:
  - kill STATIC pixels first (on in >85% of frames) — the line judge never
    moves, so he erases himself; the players never stand that still
  - near player  = largest surviving component with bottom edge in the
    lower half of the frame
  - far player   = largest surviving component with bottom edge in the
    upper half

Feet = bottom-center of the component's bbox — and feet are ON the ground
plane, so unlike the airborne ball the homography maps them exactly.

Usage:
    uv run experiments/m3_split_players.py
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"
W, H = 1280, 720
MIN_AREA = 250          # px; smaller components are mask crumbs
STATIC_FRAC = 0.85      # pixel on in >85% of frames = scenery, not player


def decode(rle_str):
    toks = list(map(int, rle_str.split()))
    flat = np.zeros(W * H, np.uint8)
    for i in range(0, len(toks), 2):
        start, run = toks[i], toks[i + 1]
        flat[start:start + run] = 1
    return flat.reshape(H, W)


def main():
    d = json.loads((OUT_DIR / "rle_players.json").read_text())
    rles = d["rle"]
    n_frames = len(rles)

    print(f"decoding {n_frames} masks...")
    masks = [decode(r) if r else np.zeros((H, W), np.uint8) for r in rles]

    on_frac = np.mean([m.astype(np.float32) for m in masks], axis=0)
    static = (on_frac > STATIC_FRAC).astype(np.uint8)
    print(f"static pixels (>{STATIC_FRAC:.0%} of frames): {static.sum()} px")
    ys, xs = np.nonzero(static)
    if len(xs):
        print(f"  static blob spans x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}"
              f"  <- expect the line judge")
    cv2.imwrite(str(OUT_DIR / "static_mask.png"), static * 255)

    rows = []
    misses = {"near": [], "far": []}
    for fi, m in enumerate(masks):
        m = m & ~static
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        halves = {"near": [], "far": []}
        for lab in range(1, n):
            x, y, w, h, area = stats[lab]
            if area < MIN_AREA:
                continue
            side = "near" if y + h > H / 2 else "far"
            halves[side].append((area, x, y, w, h))
        for side in ("near", "far"):
            if not halves[side]:
                misses[side].append(fi)
                continue
            area, x, y, w, h = max(halves[side])
            rows.append({
                "frame": fi, "player": side,
                "cx": (x + w / 2) / W, "cy": (y + h / 2) / H,
                "w": w / W, "h": h / H,
                "foot_x": x + w / 2, "foot_y": y + h,   # px, bbox bottom-center
                "area": area,
            })

    csv_path = OUT_DIR / "players_traj.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    for side in ("near", "far"):
        got = sum(1 for r in rows if r["player"] == side)
        gaps = misses[side]
        print(f"{side}: {got}/{n_frames} frames"
              + (f", missing {len(gaps)} (first few: {gaps[:8]})" if gaps else ""))
    print(f"saved {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for side, color in (("near", "tab:orange"), ("far", "tab:blue")):
        pts = [r for r in rows if r["player"] == side]
        axes[0].plot([p["frame"] for p in pts], [p["foot_x"] for p in pts],
                     ".", markersize=3, color=color, label=side)
        axes[1].plot([p["frame"] for p in pts], [p["foot_y"] for p in pts],
                     ".", markersize=3, color=color, label=side)
    axes[0].set_ylabel("foot x (px)")
    axes[1].set_ylabel("foot y (px)")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("frame (30 fps)")
    axes[0].legend()
    fig.suptitle("M3 — player feet from component-split SAM 3 mask (judge removed as static)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "players_traj.png", dpi=150)
    print(f"saved {OUT_DIR / 'players_traj.png'}")


if __name__ == "__main__":
    main()
