"""M2 run 3: pull video frames at detected events so eyes can judge them.

For each event in events.csv, save a 3-frame strip (event-4, event,
event+4) with the ball box drawn. Also dump sparse frames over any extra
suspect windows (where a bounce SHOULD be but wasn't detected) passed as
--window start:end.

Usage:
    uv run experiments/m2_verify_frames.py clips/rally.mp4 \
        outputs/m0/trajectory_ballfix.csv outputs/m2/events.csv \
        --window 60:95 --window 155:185
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("traj_csv")
    parser.add_argument("events_csv")
    parser.add_argument("--window", action="append", default=[])
    args = parser.parse_args()

    boxes = {}
    with open(args.traj_csv) as f:
        for row in csv.DictReader(f):
            boxes[int(row["frame"])] = (float(row["cx"]), float(row["cy"]),
                                        float(row["w"]), float(row["h"]))
    events = list(csv.DictReader(open(args.events_csv)))

    cap = cv2.VideoCapture(args.clip)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_frames = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames[i] = frame
        i += 1

    def annotated(fi, label):
        frame = all_frames[fi].copy()
        if fi in boxes:
            cx, cy, w, h = boxes[fi]
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * Hh)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * Hh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.circle(frame, (int(cx * W), int(cy * Hh)), 24, (0, 255, 255), 2)
        cv2.putText(frame, f"f{fi} {label}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
        return frame

    for e in events:
        fi = int(e["frame"])
        strip = np.hstack([annotated(max(0, fi - 4), ""),
                           annotated(fi, e["kind"].upper()),
                           annotated(min(i - 1, fi + 4), "")])
        strip = cv2.resize(strip, (strip.shape[1] // 2, strip.shape[0] // 2))
        cv2.imwrite(str(OUT_DIR / f"event_{fi:03d}_{e['kind']}.png"), strip)

    for wspec in args.window:
        a, b = map(int, wspec.split(":"))
        fs = list(range(a, b + 1, 4))
        rows = []
        for chunk_start in range(0, len(fs), 3):
            chunk = fs[chunk_start:chunk_start + 3]
            row = np.hstack([annotated(f, "") for f in chunk] +
                            [np.zeros((Hh, W * (3 - len(chunk)), 3), np.uint8)])
            rows.append(row)
        grid = np.vstack(rows)
        grid = cv2.resize(grid, (grid.shape[1] // 3, grid.shape[0] // 3))
        cv2.imwrite(str(OUT_DIR / f"window_{a:03d}_{b:03d}.png"), grid)

    print(f"-> {len(events)} event strips + {len(args.window)} window grids in outputs/m2/")


if __name__ == "__main__":
    main()
