"""cv-09 hero: a charted point playing with its v2 string assembling.

Adapted from m3_render_demo.py, but driven by the chart2 CSVs (charting
loop v2): broadcast frame + ball box + player boxes, striker's box
flashes with the shot word at each contact, and the bottom bar types
the v2 pseudo-MCP string token by token — letter+zone at each
contact_frame, ending '?' at the end.

Default subject: point_36 (7 shots, s6f1f?b3b??3f2?).

Usage:
    uv run experiments/render_cv09_hero.py [point_36]
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "m3" / "charts"

BAR_H = 76
FLASH = 14

C_BALL = (0, 255, 255)
C_NEAR = (0, 165, 255)
C_FAR = (255, 100, 255)


def load(point):
    ball = {}
    with open(ROOT / f"outputs/m3/ball/ball_{point}.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx_raw"]), float(row["cy_raw"]),
                                       float(row["w"]), float(row["h"]))
    players = {}
    with open(ROOT / f"outputs/m3/players/players_{point}.csv") as f:
        for row in csv.DictReader(f):
            players.setdefault(int(row["frame"]), {})[row["player"]] = (
                float(row["cx"]), float(row["cy"]),
                float(row["w"]), float(row["h"]))
    chart = list(csv.DictReader(open(OUT_DIR / f"chart2_{point}.csv")))
    return ball, players, chart


def tokens(chart):
    """(frame, token, striker, word) per shot; token = letter+zone."""
    out = []
    for s in chart:
        frame = int(s["contact_frame"]) if s["contact_frame"] else int(s["frame"])
        letter = s["letter"] or "?"
        zone = s["zone"] or "?"
        if s["is_serve"] == "True":
            tok, word = f"s{zone}", "SERVE"
        else:
            tok = f"{letter}{zone}"
            word = {"f": "FOREHAND", "b": "BACKHAND"}.get(letter, "HIT")
        out.append((frame, tok, s["striker"], word))
    return sorted(out)


def main():
    point = sys.argv[1] if len(sys.argv) > 1 else "point_36"
    ball, players, chart = load(point)
    toks = tokens(chart)

    clip = ROOT / f"clips/points/{point}.mp4"
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw = OUT_DIR / f"cv09_hero_{point}_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, Hh + BAR_H))

    def box_px(cx, cy, w, h):
        return (int((cx - w / 2) * W), int((cy - h / 2) * Hh),
                int((cx + w / 2) * W), int((cy + h / 2) * Hh))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pl = players.get(i, {})
        for side, color in (("near", C_NEAR), ("far", C_FAR)):
            if side in pl:
                x1, y1, x2, y2 = box_px(*pl[side])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if i in ball:
            x1, y1, x2, y2 = box_px(*ball[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2)

        for f_ev, tok, striker, word in toks:
            d = i - f_ev
            if not (0 <= d < FLASH):
                continue
            color = C_NEAR if striker == "near" else C_FAR
            cv2.putText(frame, word, (W // 2 - 170, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9)
            cv2.putText(frame, word, (W // 2 - 170, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
            if striker in pl:
                x1, y1, x2, y2 = box_px(*pl[striker])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

        s = "".join(tok for f_ev, tok, _, _ in toks if f_ev <= i)
        if i >= n_frames - 10:
            s += "?"  # ending: not coded yet
        bar = np.full((BAR_H, W, 3), (24, 24, 24), np.uint8)
        cv2.putText(bar, f"{point}  chart v2: {s}", (24, 52),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        canvas = np.zeros((Hh + BAR_H, W, 3), np.uint8)
        canvas[:Hh] = frame
        canvas[Hh:] = bar
        writer.write(canvas)
        i += 1

    writer.release()
    final = "".join(t for _, t, _, _ in toks) + "?"
    print(f"-> {raw} ({i} frames, final string {final})")

    out = OUT_DIR / f"cv09_hero_{point}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
