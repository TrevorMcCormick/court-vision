"""M3 experiment 3 — clip sourcing round 2, pass 1: per-frame view features.

The 24-minute R2 highlights reel is full of camera cuts (M0's first
lesson: compilations cut away from the broadcast angle constantly). To
slice it into complete points we first need to know, frame by frame,
whether we're looking at the broadcast court view or a cutaway.

Features per frame (computed at 320x180):
  blue_frac   — area of the LARGEST blue contour / frame area (the court
                is one big blue quad from the broadcast angle; M1's HSV
                range reused)
  blue_cx/cy  — that contour's centroid (normalized; broadcast angle puts
                the court mid-frame)
  green_frac  — green-pixel fraction anywhere (the apron surrounds the
                court from the broadcast angle; close-ups on court are
                blue-heavy but green-poor)

Classification happens in m3_segment_points.py — this pass just measures,
so thresholds can be iterated without re-decoding 36k frames.

Usage:
    uv run experiments/m3_court_view.py clips/match_r2_highlights.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"

BLUE_LO, BLUE_HI = (95, 60, 40), (135, 255, 255)
GREEN_LO, GREEN_HI = (35, 40, 40), (85, 255, 255)
DS_W, DS_H = 320, 180


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{args.video}: {n_total} frames @ {fps:.2f} fps")

    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (DS_W, DS_H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
        green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
        green_frac = float(np.count_nonzero(green)) / (DS_W * DS_H)

        blue_frac, bcx, bcy = 0.0, -1.0, -1.0
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            if area > 0:
                blue_frac = float(area) / (DS_W * DS_H)
                m = cv2.moments(biggest)
                bcx = m["m10"] / m["m00"] / DS_W
                bcy = m["m01"] / m["m00"] / DS_H

        rows.append({"frame": i, "blue_frac": round(blue_frac, 4),
                     "blue_cx": round(bcx, 3), "blue_cy": round(bcy, 3),
                     "green_frac": round(green_frac, 4)})
        i += 1
        if i % 5000 == 0:
            print(f"  {i}/{n_total}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "view_features.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {out} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
