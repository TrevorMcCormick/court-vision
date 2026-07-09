"""Render tracked boxes onto the clip so we can see what SAM 3 actually followed.

Usage:
    uv run experiments/m0_render_boxes.py clips/rally.mp4 outputs/m0/trajectory_ball.csv --tag ball
"""

import argparse
import csv
from pathlib import Path

import cv2

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("traj_csv")
    parser.add_argument("--tag", default="ball")
    parser.add_argument("--sample-every", type=int, default=60,
                        help="also save a PNG every N frames")
    args = parser.parse_args()

    boxes = {}
    with open(args.traj_csv) as f:
        for row in csv.DictReader(f):
            i = int(row["frame"])
            boxes[i] = (float(row["cx"]), float(row["cy"]), float(row["w"]), float(row["h"]))

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = OUT_DIR / f"boxes_{args.tag}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    trail = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in boxes:
            cx, cy, w, h = boxes[i]
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            trail.append((int(cx * W), int(cy * H)))
        for j in range(1, len(trail)):
            cv2.line(frame, trail[j - 1], trail[j], (0, 200, 255), 1)
        cv2.putText(frame, f"frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        if i % args.sample_every == 0:
            cv2.imwrite(str(OUT_DIR / f"boxframe_{args.tag}_{i:04d}.png"), frame)
        i += 1

    writer.release()
    cap.release()
    print(f"saved {out_path} and sample frames ({i} frames)")


if __name__ == "__main__":
    main()
