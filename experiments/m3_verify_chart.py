"""M3 chart-v2 verification — one strip per charted shot, full context.

Unlike m3_verify_shots (striker-crop only), each strip shows the WHOLE
frame at (contact-4, contact, contact+4) with both player boxes, the ball
circled, and the chart's claim in the caption — so eyes can adjudicate
striker AND letter AND whether the 'hit' was a hit at all.

Usage:
    uv run experiments/m3_verify_chart.py point_16 point_53
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "m3"
CHART_DIR = OUT_BASE / "charts"
ROOT = Path(__file__).resolve().parent.parent
W, H = 1280, 720
OFFSETS = (-4, 0, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    args = parser.parse_args()

    for stem in args.clips:
        shots = list(csv.DictReader(open(CHART_DIR / f"chart2_{stem}.csv")))
        ball = {}
        with open(OUT_BASE / "ball" / f"ball_{stem}.csv") as f:
            for row in csv.DictReader(f):
                ball[int(row["frame"])] = (float(row["cx_raw"]) * W,
                                           float(row["cy_raw"]) * H)
        players = {}
        with open(OUT_BASE / "players" / f"players_{stem}.csv") as f:
            for row in csv.DictReader(f):
                players.setdefault(int(row["frame"]), {})[row["player"]] = row

        cap = cv2.VideoCapture(str(ROOT / "clips/points" / f"{stem}.mp4"))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)

        for s in shots:
            fi = int(s["contact_frame"])
            tiles = []
            for off in OFFSETS:
                f = min(max(fi + off, 0), len(frames) - 1)
                fr = frames[f].copy()
                for side, col in (("near", (80, 220, 80)), ("far", (80, 80, 230))):
                    p = players.get(f, {}).get(side)
                    if p:
                        x1 = int((float(p["cx"]) - float(p["w"]) / 2) * W)
                        y1 = int((float(p["cy"]) - float(p["h"]) / 2) * H)
                        x2 = int((float(p["cx"]) + float(p["w"]) / 2) * W)
                        y2 = int((float(p["cy"]) + float(p["h"]) / 2) * H)
                        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
                if f in ball:
                    bx, by = ball[f]
                    cv2.circle(fr, (int(bx), int(by)), 12, (0, 255, 255), 2)
                cv2.putText(fr, f"f{f}" + ("" if off else " CONTACT"), (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255) if off == 0 else (255, 255, 255), 2)
                tiles.append(cv2.resize(fr, (W // 2, H // 2)))
            strip = np.hstack(tiles)
            label = (f"{stem} shot {s['shot']}  event f{s['frame']} -> contact "
                     f"f{fi} (d={s['contact_dist_px']}px)  "
                     f"{s['striker'].upper()} '{s['letter']}'  zone {s['zone']}  "
                     f"vcy_after {s['vcy_after'][:6] if s['vcy_after'] else '--'}")
            bar = np.zeros((40, strip.shape[1], 3), np.uint8)
            cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2)
            out = np.vstack([bar, strip])
            path = CHART_DIR / f"verify2_{stem}_s{int(s['shot']):02d}.png"
            cv2.imwrite(str(path), out)
            print(f"-> {path.name}")


if __name__ == "__main__":
    main()
