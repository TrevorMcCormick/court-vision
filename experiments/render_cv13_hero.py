"""cv-13 hero: t3_point_11 — the 16-shot Djokovic-Ruud rally the court
probe shattered into four fragments (old segs 12-15), reassembled into
one clip by the score-bug point-ID pass and charted 12/16 through the
frozen loop.

Plays the merged clip with the WASB ball comet, player boxes, shot-word
flashes, and the chart string typing in the bottom bar (cv09/cv12
patterns) — plus a timeline strip where the four former fragments show
as lit spans and the old cut boundaries as tick marks, with a playhead.

Usage:
    uv run experiments/render_cv13_hero.py
"""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t3"

POINT = "t3_point_11"
TRAIL = 20
FLASH = 14
BAR_H = 168

# old segs 12-15 mapped into merged-clip frames (clip starts at reel 8303)
FRAGMENTS = [(0, 117), (144, 249), (319, 429), (472, 630)]
N_FRAMES_TL = 630

C_BALL = (80, 255, 120)   # WASB green
C_NEAR = (0, 165, 255)
C_FAR = (255, 100, 255)
C_FRAG = (96, 128, 196)   # clay-ish spans
C_TICK = (0, 220, 255)    # old cut boundaries


def load():
    ball = {}
    with open(OUT_DIR / f"ball_wasb/ball_{POINT}.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx_raw"]), float(row["cy_raw"]))
    players = {}
    with open(OUT_DIR / f"players/players_{POINT}.csv") as f:
        for row in csv.DictReader(f):
            players.setdefault(int(row["frame"]), {})[row["player"]] = (
                float(row["cx"]), float(row["cy"]),
                float(row["w"]), float(row["h"]))
    chart = list(csv.DictReader(open(OUT_DIR / f"charts_wasb/chart2_{POINT}.csv")))
    return ball, players, chart


def tokens(chart):
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


def draw_bar(W, i, string_now):
    bar = np.full((BAR_H, W, 3), (24, 24, 24), np.uint8)
    # timeline strip
    x0, x1, ty0, ty1 = 24, W - 24, 14, 44

    def tx(f):
        return x0 + int((x1 - x0) * f / N_FRAMES_TL)

    cv2.rectangle(bar, (x0, ty0), (x1, ty1), (44, 44, 44), -1)
    for a, b in FRAGMENTS:
        cv2.rectangle(bar, (tx(a), ty0), (tx(b), ty1), C_FRAG, -1)
    for k, (a, b) in enumerate(FRAGMENTS):
        for f_edge in ([a] if k else []) + ([b] if k < len(FRAGMENTS) - 1 else []):
            x = tx(f_edge)
            cv2.line(bar, (x, ty0 - 6), (x, ty1 + 6), C_TICK, 2)
    cv2.line(bar, (tx(min(i, N_FRAMES_TL)), ty0 - 4),
             (tx(min(i, N_FRAMES_TL)), ty1 + 4), (255, 255, 255), 2)
    cv2.putText(bar, "old court probe: 4 fragments, charted 1-3 shots each",
                (24, 84), cv2.FONT_HERSHEY_DUPLEX, 0.85, C_TICK, 1)
    cv2.putText(bar, f"score-bug pass: one 21 s point  {string_now}",
                (24, 138), cv2.FONT_HERSHEY_DUPLEX, 1.1, C_BALL, 2)
    return bar


def main():
    ball, players, chart = load()
    toks = tokens(chart)

    clip = ROOT / f"clips/points_t3/{POINT}.mp4"
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw = OUT_DIR / "cv13_hero_raw.mp4"
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

        # ball comet
        for d in range(TRAIL, -1, -1):
            j = i - d
            if j not in ball:
                continue
            cx, cy = ball[j]
            x, y = int(cx * W), int(cy * Hh)
            if d == 0:
                cv2.circle(frame, (x, y), 10, C_BALL, 2)
                cv2.circle(frame, (x, y), 4, C_BALL, -1)
            else:
                fade = 1.0 - d / (TRAIL + 1)
                r = max(1, int(1 + 4 * fade))
                c = tuple(int(ch * (0.35 + 0.65 * fade)) for ch in C_BALL)
                cv2.circle(frame, (x, y), r, c, -1)

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
        canvas = np.zeros((Hh + BAR_H, W, 3), np.uint8)
        canvas[:Hh] = frame
        canvas[Hh:] = draw_bar(W, i, s)
        writer.write(canvas)
        i += 1

    writer.release()
    final = "".join(t for _, t, _, _ in toks) + "?"
    print(f"-> {raw} ({i} frames, final string {final})")

    out = OUT_DIR / "cv13_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
