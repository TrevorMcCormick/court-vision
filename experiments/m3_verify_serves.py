"""M3 run 5 verification — server crops around each detected serve moment.

For sampled clips, a 5-frame strip centered on the detected serve frame,
cropped to the detected server, so eyes can judge both calls at once:
right player? toss/contact at the peak?

Usage:
    uv run experiments/m3_verify_serves.py [--every 5]
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "m3"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = OUT_BASE / "serve_checks"

PAD = 60
TILE_H = 380


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args()

    serves = [r for r in csv.DictReader(open(OUT_BASE / "serves.csv"))
              if r["server"] != "?"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for s in serves[::args.every]:
        stem = s["clip"]
        tracks = {}
        with open(OUT_BASE / "players" / f"players_{stem}.csv") as f:
            for r in csv.DictReader(f):
                tracks.setdefault(int(r["frame"]), {})[r["player"]] = r
        cap = cv2.VideoCapture(str(ROOT / "clips/points" / f"{stem}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)

        sf = int(s["serve_frame"])
        side = s["server"]
        p = tracks.get(sf, {}).get(side)
        if p is None:
            continue
        W, Hh = frames[0].shape[1], frames[0].shape[0]
        cx, cy = float(p["cx"]) * W, float(p["cy"]) * Hh
        w, h = float(p["w"]) * W, float(p["h"]) * Hh
        x1 = int(max(0, cx - w / 2 - PAD)); x2 = int(min(W, cx + w / 2 + PAD))
        y1 = int(max(0, cy - h / 2 - PAD)); y2 = int(min(Hh, cy + h / 2 + PAD))

        offs = [int(-0.6 * fps), int(-0.2 * fps), 0, int(0.2 * fps), int(0.6 * fps)]
        tiles = []
        for off in offs:
            fi = min(max(sf + off, 0), len(frames) - 1)
            crop = frames[fi][y1:y2, x1:x2]
            scale = TILE_H / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), TILE_H))
            cv2.putText(crop, "SERVE?" if off == 0 else f"{off:+d}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 255) if off == 0 else (255, 255, 255), 2)
            tiles.append(crop)
        strip = np.hstack(tiles)
        bar = np.zeros((40, strip.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{stem}  server={side} ({s['side']}, x={s['server_x_m']}m, "
                         f"margin {s['margin_m']}m)  serve@f{sf} ({s['serve_s']}s)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / f"serve_{stem}.png"), np.vstack([bar, strip]))
        print(f"-> serve_{stem}.png")


if __name__ == "__main__":
    main()
