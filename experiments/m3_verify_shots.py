"""M3 run 2 verification — zoomed strips of the striker at each hit.

m2_verify_frames.py strips show the whole frame; the far player is ~60 px
tall in them and a forehand/backhand call needs the swing. This pulls a
5-frame window (fi-6, fi-2, fi, fi+2, fi+6) cropped around the STRIKER's
box (padded), upscaled, ball circled when tracked, so eyes can call f/b
against the pipeline's letter.

Usage:
    uv run experiments/m3_verify_shots.py clips/rally.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"
ROOT = Path(__file__).resolve().parent.parent
W, H = 1280, 720
OFFSETS = (-6, -2, 0, 2, 6)
PAD = 55          # px around the striker box
TILE_H = 420      # upscale target height per tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    args = parser.parse_args()

    ball = {}
    with open(ROOT / "outputs/m0/trajectory_ballfix.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx"]) * W, float(row["cy"]) * H)
    players = {}
    with open(OUT_DIR / "players_traj.csv") as f:
        for row in csv.DictReader(f):
            players.setdefault(int(row["frame"]), {})[row["player"]] = row
    shots = list(csv.DictReader(open(OUT_DIR / "shot_types.csv")))

    cap = cv2.VideoCapture(args.clip)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)

    for s in shots:
        fi = int(s["frame"])
        side = s["striker"]
        # fixed crop across the strip, centered on the striker at contact
        p = players[fi][side]
        cx, cy = float(p["cx"]) * W, float(p["cy"]) * H
        w, h = float(p["w"]) * W, float(p["h"]) * H
        x1 = int(max(0, cx - w / 2 - PAD)); x2 = int(min(W, cx + w / 2 + PAD))
        y1 = int(max(0, cy - h / 2 - PAD)); y2 = int(min(H, cy + h / 2 + PAD))

        tiles = []
        for off in OFFSETS:
            f = min(max(fi + off, 0), len(frames) - 1)
            fr = frames[f].copy()
            if f in ball:
                bx, by = ball[f]
                cv2.circle(fr, (int(bx), int(by)), 14, (0, 255, 255), 2)
            crop = fr[y1:y2, x1:x2]
            scale = TILE_H / crop.shape[0]
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), TILE_H))
            cv2.putText(crop, f"{off:+d}" if off else "CONTACT", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255) if off == 0 else (255, 255, 255), 2)
            tiles.append(crop)

        strip = np.hstack(tiles)
        label = (f"shot {s['shot']}  f{fi}  {side.upper()}  ->  "
                 f"'{s['shot_type']}'  (dx {s['ball_dx_px']} px)")
        bar = np.zeros((44, strip.shape[1], 3), np.uint8)
        cv2.putText(bar, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        out = np.vstack([bar, strip])
        path = OUT_DIR / f"shot_{int(s['shot']):02d}_f{fi:03d}_{s['shot_type']}.png"
        cv2.imwrite(str(path), out)
        print(f"-> {path.name}")


if __name__ == "__main__":
    main()
