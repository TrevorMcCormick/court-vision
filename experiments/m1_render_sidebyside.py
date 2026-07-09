"""M1: render the demo video — broadcast + tracked box on the left, the
top-down court with the mapped shadow track drawing itself on the right.

Usage:
    uv run experiments/m1_render_sidebyside.py clips/rally.mp4 \
        outputs/m0/trajectory_ballfix.csv outputs/m1/H_img_to_court.npy
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m1"

W_COURT = 10.97
L_COURT = 23.77
SINGLES_INSET = 1.372
NET_Y = L_COURT / 2
SVC_FAR_Y = NET_Y - 6.40
SVC_NEAR_Y = NET_Y + 6.40
CENTER_X = W_COURT / 2

PPM = 22            # pixels per meter in the court panel
MARGIN_M = 2.5      # meters of apron around the court
PANEL_H = 720
LAST_FRAME = 290    # track dies ~300 (rally over)


def court_to_panel(x, y):
    return int((x + MARGIN_M) * PPM), int((y + MARGIN_M) * PPM)


def draw_court_panel():
    w = int((W_COURT + 2 * MARGIN_M) * PPM)
    h = int((L_COURT + 2 * MARGIN_M) * PPM)
    panel = np.full((PANEL_H, w, 3), (60, 60, 60), np.uint8)
    court = np.full((h, w, 3), (146, 91, 59), np.uint8)  # court blue (BGR)

    def line(x1, y1, x2, y2, color=(255, 255, 255), t=2):
        cv2.line(court, court_to_panel(x1, y1), court_to_panel(x2, y2), color, t)

    line(0, 0, W_COURT, 0)
    line(0, L_COURT, W_COURT, L_COURT)
    line(0, 0, 0, L_COURT)
    line(W_COURT, 0, W_COURT, L_COURT)
    line(SINGLES_INSET, 0, SINGLES_INSET, L_COURT)
    line(W_COURT - SINGLES_INSET, 0, W_COURT - SINGLES_INSET, L_COURT)
    line(SINGLES_INSET, SVC_FAR_Y, W_COURT - SINGLES_INSET, SVC_FAR_Y)
    line(SINGLES_INSET, SVC_NEAR_Y, W_COURT - SINGLES_INSET, SVC_NEAR_Y)
    line(CENTER_X, SVC_FAR_Y, CENTER_X, SVC_NEAR_Y)
    line(0, NET_Y, W_COURT, NET_Y, color=(200, 200, 200), t=3)

    # letterbox the court into the fixed-height panel
    scale = PANEL_H / h
    court = cv2.resize(court, (int(w * scale), PANEL_H))
    return court, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("traj_csv")
    parser.add_argument("h_npy")
    args = parser.parse_args()

    H = np.load(args.h_npy)

    boxes = {}
    with open(args.traj_csv) as f:
        for row in csv.DictReader(f):
            boxes[int(row["frame"])] = (float(row["cx"]), float(row["cy"]),
                                        float(row["w"]), float(row["h"]))

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_panel, scale = draw_court_panel()
    panel_w = base_panel.shape[1]

    out_path = OUT_DIR / "sidebyside.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W + panel_w, PANEL_H))

    def to_panel_px(court_xy):
        x, y = court_xy
        return (int((x + MARGIN_M) * PPM * scale), int((y + MARGIN_M) * PPM * scale))

    trail = []
    i = 0
    while i <= LAST_FRAME:
        ok, frame = cap.read()
        if not ok:
            break
        if i in boxes:
            cx, cy, w, h = boxes[i]
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * Hh)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * Hh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            pt = cv2.perspectiveTransform(
                np.float32([[cx * W, cy * Hh]]).reshape(-1, 1, 2), H).reshape(2)
            trail.append(to_panel_px(pt))

        panel = base_panel.copy()
        for j, p in enumerate(trail[:-1]):
            cv2.circle(panel, p, 2, (80, 200, 255), -1)
        if trail:
            cv2.circle(panel, trail[-1], 6, (0, 255, 255), 2)

        canvas = np.zeros((PANEL_H, W + panel_w, 3), np.uint8)
        canvas[:Hh, :W] = frame
        canvas[:, W:] = panel
        writer.write(canvas)
        if i % 72 == 0:
            cv2.imwrite(str(OUT_DIR / f"sbs_{i:04d}.png"), canvas)
        i += 1

    writer.release()
    print(f"-> {out_path} ({i} frames, {W + panel_w}x{PANEL_H})")


if __name__ == "__main__":
    main()
