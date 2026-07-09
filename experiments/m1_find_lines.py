"""M1 run 2: find the court lines on the clean plate.

Strategy: the court surface is blue and the lines are white — so first find
the blue region (largest blue contour = the court), then look for white
pixels only inside (a slightly dilated hull of) it. That keeps broadcast
text overlays, the scoreboard, and crowd whites out of Hough.

Usage:
    uv run experiments/m1_find_lines.py outputs/m1/clean_plate.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plate")
    args = parser.parse_args()

    img = cv2.imread(args.plate)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # what hue is the court, actually? sample the middle of the frame
    patch = hsv[int(h * 0.55):int(h * 0.65), int(w * 0.45):int(w * 0.55)]
    print(f"center patch HSV median: {np.median(patch.reshape(-1, 3), axis=0)}")

    # blue court mask
    blue = cv2.inRange(hsv, (95, 60, 40), (135, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    court = max(contours, key=cv2.contourArea)
    print(f"court contour area: {cv2.contourArea(court):.0f} px "
          f"({cv2.contourArea(court) / (h * w) * 100:.1f}% of frame)")

    court_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(court_mask, [cv2.convexHull(court)], -1, 255, -1)
    court_mask = cv2.dilate(court_mask, np.ones((25, 25), np.uint8))
    cv2.imwrite(str(OUT_DIR / "court_mask.png"), court_mask)

    # white lines inside the court region
    white = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    white = cv2.bitwise_and(white, court_mask)
    cv2.imwrite(str(OUT_DIR / "white_mask.png"), white)

    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=80,
                           minLineLength=60, maxLineGap=12)
    segs = segs.reshape(-1, 4) if segs is not None else []
    print(f"HoughLinesP segments: {len(segs)}")

    overlay = img.copy()
    for x1, y1, x2, y2 in segs:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / "lines_overlay.png"), overlay)
    np.save(OUT_DIR / "segments.npy", np.array(segs))
    print("-> outputs/m1/{court_mask,white_mask,lines_overlay}.png, segments.npy")


if __name__ == "__main__":
    main()
