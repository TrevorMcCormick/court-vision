"""cv-14 hero: t3_point_45 — a clay point the v5 skeleton + direction
model charts one token edit from the human chart, i.e. ACCEPTED under
the north-star metric.

Plays the clip with the WASB ball comet while the machine string
assembles token-by-token above the human MCP string: tokens light
green on exact match; the one mismatched token (our '*' vs the
charter's 'n') lights amber, and the verdict stamp lands — distance 1,
accepted at <= 1 edit. End card: the acceptance effort curve, before
(v5) -> after (shot-direction v2), plus the honest footnote (t4 0/49).

Reads chart/ball CSVs + MCP map only; touches nothing in the pipeline.

Usage:
    uv run experiments/render_cv14_hero.py
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from mcp_accept import (mcp_point_tokens, chart_point_tokens,
                        token_levenshtein)  # noqa: E402

OUT_DIR = ROOT / "outputs" / "t3"
POINT = "t3_point_45"

TRAIL = 20
FLASH = 16
BAR_H = 216
OUT_FPS = 24          # 30 fps source -> gentle slow-mo
END_CARD_S = 6.5
HOLD_S = 1.2

C_BALL = (80, 255, 120)    # WASB green
C_OK = (80, 255, 120)
C_BAD = (0, 165, 255)      # amber
C_DIM = (130, 130, 130)
C_HUM = (210, 210, 210)
C_NEAR = (0, 165, 255)
C_FAR = (255, 100, 255)
BG = (24, 24, 24)


def load():
    ball = {}
    with open(OUT_DIR / f"ball_wasb/ball_{POINT}.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx_raw"]), float(row["cy_raw"]))
    shots = list(csv.DictReader(open(OUT_DIR / f"charts_wasb/chart2_{POINT}.csv")))
    match = {r["clip"]: r for r in
             csv.DictReader(open(OUT_DIR / "charts_wasb/match_chart_v2.csv"))}
    mapd = {r["clip"]: r for r in
            csv.DictReader(open(ROOT / "data/mcp/t3_mcp_map.csv"))}
    m = mapd[POINT]
    played = m["second"] if m["second"].strip() else m["first"]
    return ball, shots, match[POINT]["ending"], played


def token_schedule(shots, n_frames):
    """(reveal_frame, striker, word) per machine token, serve..ending."""
    sched = []
    for s in shots:
        f = int(s["contact_frame"]) if s["contact_frame"] else int(s["frame"])
        if s["is_serve"] == "True":
            word = "SERVE"
        else:
            word = {"f": "FOREHAND", "b": "BACKHAND"}.get(s["letter"], "HIT")
        sched.append((f, s["striker"], word))
    sched.append((min(sched[-1][0] + 32, n_frames - 12), "", "ENDING"))
    return sched


def draw_bar(W, i, ours, mcp, sched, dist):
    bar = np.full((BAR_H, W, 3), BG, np.uint8)
    x0, dx = 330, 128
    cv2.putText(bar, "machine", (24, 64), cv2.FONT_HERSHEY_DUPLEX,
                0.9, (255, 255, 255), 1)
    cv2.putText(bar, "human (MCP)", (24, 136), cv2.FONT_HERSHEY_DUPLEX,
                0.9, C_HUM, 1)
    n_shown = sum(1 for f, _, _ in sched if i >= f)
    for k, tok in enumerate(mcp):
        judged = k < n_shown and k < len(ours)
        ok = judged and ours[k] == tok
        # human row: always visible, dim until judged
        c_h = C_DIM if not judged else (C_OK if ok else C_HUM)
        cv2.putText(bar, tok, (x0 + k * dx, 136), cv2.FONT_HERSHEY_DUPLEX,
                    1.35, c_h, 2)
    for k, tok in enumerate(ours):
        if k >= n_shown:
            continue
        judged_ok = k < len(mcp) and tok == mcp[k]
        c_m = C_OK if judged_ok else C_BAD
        cv2.putText(bar, tok, (x0 + k * dx, 64), cv2.FONT_HERSHEY_DUPLEX,
                    1.35, c_m, 2)
        # fresh token gets an underline pulse
        f_rev = sched[k][0]
        if 0 <= i - f_rev < FLASH:
            cv2.line(bar, (x0 + k * dx - 6, 76),
                     (x0 + k * dx + 74, 76), c_m, 2)
    if n_shown >= len(ours):
        cv2.putText(bar, f"token distance {dist}  ->  ACCEPTED (<= 1 edit)",
                    (24, 196), cv2.FONT_HERSHEY_DUPLEX, 1.0, C_OK, 2)
    else:
        cv2.putText(bar, "accepted = within 1 token edit of the human chart",
                    (24, 196), cv2.FONT_HERSHEY_DUPLEX, 0.8, C_DIM, 1)
    return bar


def end_card(W, H):
    """Acceptance effort curve, v5 baseline -> after shot-direction v2."""
    card = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(card, "acceptance: the north star", (60, 90),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(card, "points within k token edits of the human chart (n=135)",
                (60, 140), cv2.FONT_HERSHEY_DUPLEX, 0.85, C_HUM, 1)
    rows = [("<= 1 edit (accepted)", 2.2, 5.2),
            ("<= 2 edits", 6.7, 11.1),
            ("<= 3 edits", 14.8, 23.7),
            ("<= 5 edits", 41.5, 57.0)]
    bx, bw, y = 480, 560, 220
    for label, before, after in rows:
        cv2.putText(card, label, (60, y + 30), cv2.FONT_HERSHEY_DUPLEX,
                    0.95, (255, 255, 255), 1)
        cv2.rectangle(card, (bx, y), (bx + int(bw * before / 60), y + 16),
                      C_DIM, -1)
        cv2.rectangle(card, (bx, y + 24), (bx + int(bw * after / 60), y + 40),
                      C_OK, -1)
        cv2.putText(card, f"{before}%", (bx + int(bw * before / 60) + 12, y + 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, C_DIM, 1)
        cv2.putText(card, f"{after}%", (bx + int(bw * after / 60) + 12, y + 39),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, C_OK, 1)
        y += 100
    cv2.putText(card, "grey: event detector v5   green: + shot-direction v2",
                (60, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, C_HUM, 1)
    cv2.putText(card, "7/135 accepted  |  mean token distance 7.18 -> 6.19  |  t4 still 0/49",
                (60, y + 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, C_BAD, 1)
    return card


def main():
    ball, shots, ending, played = load()
    ours = chart_point_tokens(shots, ending)
    mcp = mcp_point_tokens(played)
    dist = token_levenshtein(mcp, ours)
    print(f"machine {ours}  human {mcp}  distance {dist}")
    assert dist <= 1, "hero premise broken: point no longer accepted"

    clip = ROOT / f"clips/points_t3/{POINT}.mp4"
    cap = cv2.VideoCapture(str(clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sched = token_schedule(shots, n_frames)

    raw = OUT_DIR / "cv14_hero_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             OUT_FPS, (W, Hh + BAR_H))

    i = 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
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
        # shot-word flash
        for k, (f_ev, striker, word) in enumerate(sched[:-1]):
            if 0 <= i - f_ev < FLASH:
                color = C_NEAR if striker == "near" else C_FAR
                cv2.putText(frame, word, (W // 2 - 170, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9)
                cv2.putText(frame, word, (W // 2 - 170, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
        canvas = np.zeros((Hh + BAR_H, W, 3), np.uint8)
        canvas[:Hh] = frame
        canvas[Hh:] = draw_bar(W, i, ours, mcp, sched, dist)
        writer.write(canvas)
        last = canvas
        i += 1

    for _ in range(int(HOLD_S * OUT_FPS)):
        writer.write(last)
    card = end_card(W, Hh + BAR_H)
    for _ in range(int(END_CARD_S * OUT_FPS)):
        writer.write(card)
    writer.release()
    print(f"-> {raw} ({i} clip frames + hold + end card)")

    out = OUT_DIR / "cv14_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
