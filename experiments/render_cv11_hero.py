"""cv-11 hero: t1_point_25 (match point, Shapovalov d. Nadal 2017)
playing with TWO strings drawn — ours assembling shot by shot from
chart2_t1_point_25.csv (as render_cv09_hero.py does), and below it the
human Match Charting Project string for the same point, static, for
comparison.

The bar is labeled "ours (frozen pipeline)" — the string as graded in
the T1 scorecard (LOG 2026-07-09): s???b2?1?2????b3???. The working-tree
chart CSVs have since been regenerated with the handedness fix (b -> f
on shots 3 and 8), so FROZEN_LETTERS pins the graded letters back; drop
it to render the post-fix chart instead.

Usage:
    uv run experiments/render_cv11_hero.py
"""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t1"

POINT = "t1_point_25"
HUMAN_MCP = "4b29f1b3f2b2f3f3f3f+1*"
FROZEN_LETTERS = {3: "b", 8: "b"}  # letters as graded pre-handedness-fix

BAR_H = 128
FLASH = 14

C_BALL = (0, 255, 255)
C_NEAR = (0, 165, 255)
C_FAR = (255, 100, 255)


def load():
    ball = {}
    with open(OUT_DIR / f"ball/ball_{POINT}.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx_raw"]), float(row["cy_raw"]),
                                       float(row["w"]), float(row["h"]))
    players = {}
    with open(OUT_DIR / f"players/players_{POINT}.csv") as f:
        for row in csv.DictReader(f):
            players.setdefault(int(row["frame"]), {})[row["player"]] = (
                float(row["cx"]), float(row["cy"]),
                float(row["w"]), float(row["h"]))
    chart = list(csv.DictReader(open(OUT_DIR / f"charts/chart2_{POINT}.csv")))
    return ball, players, chart


def tokens(chart):
    out = []
    for s in chart:
        frame = int(s["contact_frame"]) if s["contact_frame"] else int(s["frame"])
        letter = FROZEN_LETTERS.get(int(s["shot"]), s["letter"]) or "?"
        zone = s["zone"] or "?"
        if s["is_serve"] == "True":
            tok, word = f"s{zone}", "SERVE"
        else:
            tok = f"{letter}{zone}"
            word = {"f": "FOREHAND", "b": "BACKHAND"}.get(letter, "HIT")
        out.append((frame, tok, s["striker"], word))
    return sorted(out)


def main():
    ball, players, chart = load()
    toks = tokens(chart)

    clip = ROOT / f"clips/points_t1/{POINT}.mp4"
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw = OUT_DIR / "cv11_hero_raw.mp4"
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
            s += "?"
        bar = np.full((BAR_H, W, 3), (24, 24, 24), np.uint8)
        cv2.putText(bar, f"ours (frozen pipeline): {s}", (24, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(bar, f"human chart (MCP):      {HUMAN_MCP}", (24, 104),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (120, 220, 120), 2)
        canvas = np.zeros((Hh + BAR_H, W, 3), np.uint8)
        canvas[:Hh] = frame
        canvas[Hh:] = bar
        writer.write(canvas)
        i += 1

    writer.release()
    final = "".join(t for _, t, _, _ in toks) + "?"
    print(f"-> {raw} ({i} frames, ours: {final} vs MCP: {HUMAN_MCP})")

    out = OUT_DIR / "cv11_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
