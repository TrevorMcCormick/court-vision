"""M1 run 1: static-camera check via temporal median.

If the broadcast camera is static for the rally, the per-pixel median over
time is a crisp "clean plate": court lines stay sharp, players/ball vanish.
If the camera pans/zooms even slightly, lines smear. Also quantifies it:
edge overlap (dilated-Canny IoU) between frame 0 and frame 450.

Usage:
    uv run experiments/m1_clean_plate.py clips/rally.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m1"


def edges(gray):
    return cv2.Canny(gray, 50, 150)


def edge_iou(a, b, dilate_px=3):
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    da = cv2.dilate(a, kernel) > 0
    db = cv2.dilate(b, kernel) > 0
    inter = np.logical_and(da, db).sum()
    union = np.logical_or(da, db).sum()
    return inter / union if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--sample-every", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    frames = []
    first = last = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first is None:
            first = frame.copy()
        last = frame
        if i % args.sample_every == 0:
            frames.append(frame)
        i += 1
    cap.release()

    stack = np.stack(frames)  # (N, H, W, 3) uint8
    median = np.median(stack, axis=0).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "clean_plate.png"), median)
    print(f"median over {len(frames)} sampled frames ({i} total) -> outputs/m1/clean_plate.png")

    g_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    g_last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    g_median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

    iou_first_last = edge_iou(edges(g_first), edges(g_last))
    iou_first_median = edge_iou(edges(g_first), edges(g_median))
    print(f"edge IoU frame0 vs frame{i-1}: {iou_first_last:.3f}")
    print(f"edge IoU frame0 vs median:    {iou_first_median:.3f}")

    # sharpness of line detail in the median vs a single frame:
    # if the camera moved, median lines blur and Laplacian variance drops
    lap_first = cv2.Laplacian(g_first, cv2.CV_64F).var()
    lap_median = cv2.Laplacian(g_median, cv2.CV_64F).var()
    print(f"Laplacian variance frame0: {lap_first:.1f}  median: {lap_median:.1f}  "
          f"(ratio {lap_median / lap_first:.2f})")


if __name__ == "__main__":
    main()
