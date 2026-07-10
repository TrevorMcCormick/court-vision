"""cv-12 hero: t2_point_01 (the 12-shot Federer rally) side-by-side —
the SAME clip twice, left with SAM-3's sparse ball track, right with
WASB's dense one, each drawn as a trailing comet of dots. SAM charted
this rally as 6 shots; WASB through the identical frozen loop got
12/12 shots and 7/7 committed letters.

Usage:
    uv run experiments/render_cv12_hero.py
"""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t2"

POINT = "t2_point_01"
TRAIL = 20  # frames of comet tail

PANEL_W, PANEL_H = 640, 360
BAR_H = 96

C_SAM = (0, 255, 255)   # yellow
C_WASB = (80, 255, 120)  # green


def load_track(path):
    track = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            track[int(row["frame"])] = (float(row["cx_raw"]), float(row["cy_raw"]))
    return track


def draw_panel(frame_small, track, i, color):
    panel = frame_small.copy()
    for d in range(TRAIL, -1, -1):
        j = i - d
        if j not in track:
            continue
        cx, cy = track[j]
        x, y = int(cx * PANEL_W), int(cy * PANEL_H)
        if d == 0:
            cv2.circle(panel, (x, y), 9, color, 2)
            cv2.circle(panel, (x, y), 3, color, -1)
        else:
            fade = 1.0 - d / (TRAIL + 1)
            r = max(1, int(1 + 3 * fade))
            c = tuple(int(ch * (0.35 + 0.65 * fade)) for ch in color)
            cv2.circle(panel, (x, y), r, c, -1)
    return panel


def main():
    sam = load_track(OUT_DIR / f"ball/ball_{POINT}.csv")
    wasb = load_track(OUT_DIR / f"ball_wasb/ball_{POINT}.csv")

    clip = ROOT / f"clips/points_t2/{POINT}.mp4"
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)

    raw = OUT_DIR / "cv12_hero_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (PANEL_W * 2, PANEL_H + BAR_H))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (PANEL_W, PANEL_H))

        left = draw_panel(small, sam, i, C_SAM)
        right = draw_panel(small, wasb, i, C_WASB)

        for panel, tag in ((left, "SAM-3"), (right, "WASB")):
            cv2.putText(panel, tag, (16, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(panel, tag, (16, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)

        bar = np.full((BAR_H, PANEL_W * 2, 3), (24, 24, 24), np.uint8)
        cv2.putText(bar, "SAM-3: 6 shots detected", (24, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, C_SAM, 2)
        cv2.putText(bar, "WASB: 12/12 shots, 7/7 letters", (PANEL_W + 24, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, C_WASB, 2)
        cv2.line(bar, (PANEL_W, 0), (PANEL_W, BAR_H), (60, 60, 60), 1)

        canvas = np.zeros((PANEL_H + BAR_H, PANEL_W * 2, 3), np.uint8)
        canvas[:PANEL_H, :PANEL_W] = left
        canvas[:PANEL_H, PANEL_W:] = right
        canvas[PANEL_H:] = bar
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), (24, 24, 24), 2)
        writer.write(canvas)
        i += 1

    writer.release()
    print(f"-> {raw} ({i} frames, sam {len(sam)} pts, wasb {len(wasb)} pts)")

    out = OUT_DIR / "cv12_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
        check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
