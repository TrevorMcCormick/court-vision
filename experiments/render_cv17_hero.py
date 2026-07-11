"""cv-17 hero: the review copy.

Phase 1: the deliverable itself — t6_mcp_draft.csv rendered as a
scrolling table, all 134 rows, HIGH rows lit green, with the split
pinned in a summary card. Phase 2: two HIGH points play with the WASB
ball comet and their draft row in the top bar — the charter's job
becoming "check it" instead of "type it". Phase 3: the honest-split
card (what the file admits). End card: the ask — 32 charters covered
a quarter of the 2025 tour; this file is an application to assist
all of them.

Reads only outputs/t6/export/, clips/points_t6/, outputs/t6/ball_wasb/.
Touches nothing in the pipeline.

Usage:
    uv run experiments/render_cv17_hero.py
"""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

W, H = 1280, 720
OUT_FPS = 24

FONT = cv2.FONT_HERSHEY_DUPLEX
BG = (24, 24, 24)
C_HIGH = (80, 255, 120)          # green
C_HDR = (210, 210, 210)
C_DIM = (120, 120, 120)
C_AMBER = (0, 165, 255)
C_WHITE = (255, 255, 255)
C_BALL = (80, 255, 120)

DRAFT = ROOT / "outputs/t6/export/t6_mcp_draft.csv"
EXCERPTS = ["t6_point_01", "t6_point_26"]
EXCERPT_MAX_FRAMES = 100
TRAIL = 20
END_CARD_S = 3.6

ROW_H = 26
TABLE_Y0, TABLE_Y1 = 148, 636    # scroll window on screen
# column x positions: Pt, Pts, Svr, 1st (draft), conf, p, jump
COLS = [(40, "Pt"), (110, "Pts"), (215, "Sv"), (280, "1st (draft)"),
        (640, "conf"), (760, "p"), (860, "jump to")]


def load_rows():
    return list(csv.DictReader(open(DRAFT)))


def build_strip(rows):
    """Pre-render every CSV row into one tall image strip."""
    strip = np.full((len(rows) * ROW_H + 8, W, 3), BG, np.uint8)
    for i, r in enumerate(rows):
        y = i * ROW_H + 20
        hi = r["confidence"] == "high"
        c_main = C_HIGH if hi else C_DIM
        vals = [r["Pt"] or "-", r["Pts"] or "-", r["Svr"] or "-",
                r["1st"], r["confidence"], r["conf_p"],
                f'{r["serve_s"]}s' if r["serve_s"] else "-"]
        for (x, _), v in zip(COLS, vals):
            cv2.putText(strip, v, (x, y), FONT, 0.52, c_main, 1)
        if hi:
            cv2.circle(strip, (24, y - 6), 4, C_HIGH, -1)
    return strip


def draw_table_frame(strip, off, n_high, n_low, n_blank):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "the review copy: t6_mcp_draft.csv",
                (32, 52), FONT, 0.95, C_WHITE, 2)
    cv2.putText(img, "Sabalenka-Pegula, US Open F 2024 -- MCP points schema + confidence",
                (32, 84), FONT, 0.58, C_HDR, 1)
    for x, name in COLS:
        cv2.putText(img, name, (x, 124), FONT, 0.5, C_HDR, 1)
    cv2.line(img, (32, 132), (W - 240, 132), (70, 70, 70), 1)
    win = TABLE_Y1 - TABLE_Y0
    off = int(np.clip(off, 0, strip.shape[0] - win))
    img[TABLE_Y0:TABLE_Y1, :] = strip[off:off + win, :]
    # pinned summary card, right edge
    x0, y0 = W - 220, 148
    cv2.rectangle(img, (x0, y0), (W - 24, y0 + 190), (70, 70, 70), 1)
    for dy, txt, c in [(40, "134 points", C_WHITE),
                       (80, f"{n_high} HIGH", C_HIGH),
                       (120, f"{n_low} low", C_DIM),
                       (160, f"{n_blank} unplaced", C_AMBER)]:
        cv2.putText(img, txt, (x0 + 16, y0 + dy), FONT, 0.62, c, 1)
    cv2.putText(img, "one row per point, in the Match Charting Project's own columns",
                (32, H - 36), FONT, 0.68, C_HDR, 1)
    return img


def table_phase(writer, rows):
    strip = build_strip(rows)
    n_high = sum(r["confidence"] == "high" for r in rows)
    n_low = sum(r["confidence"] == "low" for r in rows)
    n_blank = sum(not r["Pt"] for r in rows)
    win = TABLE_Y1 - TABLE_Y0
    total = strip.shape[0] - win
    hold = lambda img, s: [writer.write(img) for _ in range(int(s * OUT_FPS))]
    hold(draw_table_frame(strip, 0, n_high, n_low, n_blank), 1.6)
    n = int(6.5 * OUT_FPS)
    for i in range(n):
        t = (i + 1) / n
        writer.write(draw_table_frame(strip, total * t,
                                      n_high, n_low, n_blank))
    hold(draw_table_frame(strip, total, n_high, n_low, n_blank), 1.4)


def excerpt_phase(writer, rows, clip, caption):
    r = next(row for row in rows if row["clip"] == clip)
    ball = {}
    with open(ROOT / f"outputs/t6/ball_wasb/ball_{clip}.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx_raw"]),
                                       float(row["cy_raw"]))
    cap = cv2.VideoCapture(str(ROOT / f"clips/points_t6/{clip}.mp4"))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    serve_f = int(float(r["serve_s"]) * 30) if r["serve_s"] else 0
    start = max(0, serve_f - 15)
    idxs = list(range(start, len(frames)))
    if len(idxs) > EXCERPT_MAX_FRAMES:
        step = len(idxs) / EXCERPT_MAX_FRAMES
        idxs = [idxs[int(k * step)] for k in range(EXCERPT_MAX_FRAMES)]
    for i in idxs:
        fr = frames[i].copy()
        for d in range(TRAIL, -1, -1):
            j = i - d
            if j not in ball:
                continue
            cx, cy = ball[j]
            x, y = int(cx * W), int(cy * H)
            if d == 0:
                cv2.circle(fr, (x, y), 10, C_BALL, 2)
                cv2.circle(fr, (x, y), 4, C_BALL, -1)
            else:
                fade = 1.0 - d / (TRAIL + 1)
                rad = max(1, int(1 + 4 * fade))
                c = tuple(int(ch * (0.35 + 0.65 * fade)) for ch in C_BALL)
                cv2.circle(fr, (x, y), rad, c, -1)
        cv2.rectangle(fr, (0, 0), (W, 96), BG, -1)
        cv2.putText(fr, "HIGH", (40, 40), FONT, 0.9, C_HIGH, 2)
        cv2.putText(fr, f"conf {r['conf_p']}", (150, 40), FONT, 0.7,
                    C_HIGH, 1)
        cv2.putText(fr, r["1st"], (40, 82), FONT, 1.1, C_WHITE, 2)
        cv2.putText(fr, f"Pt {r['Pt']}   {r['Pts']}   jump to {r['serve_s']}s",
                    (W - 560, 40), FONT, 0.65, C_HDR, 1)
        cv2.putText(fr, caption, (W - 560, 82), FONT, 0.65, C_HIGH, 1)
        writer.write(fr)


def split_card(writer):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "what the file admits", (60, 100), FONT, 1.2,
                C_WHITE, 2)
    lines = [
        ("39 rows HIGH -- start from the draft", C_HIGH),
        ("   (94% within 5 token edits, held out)", C_HIGH),
        ("95 rows low -- chart those from scratch", C_DIM),
        ("6 rows the score bug couldn't place", C_AMBER),
        ("every fault invisible -- drafts assume first serve", C_AMBER),
    ]
    y = 210
    for text, c in lines:
        cv2.putText(img, text, (60, y), FONT, 0.9, c, 2)
        y += 78
    cv2.putText(img, "a draft that hides its weaknesses wastes the charter's hour",
                (60, 640), FONT, 0.7, C_HDR, 1)
    for _ in range(int(4.0 * OUT_FPS)):
        writer.write(img)


def end_card(writer):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "the 33rd charter", (60, 100), FONT, 1.4, C_WHITE, 2)
    lines = [
        ("32 humans charted a quarter of the 2025 tour", C_HDR),
        ("this file is an application to assist all of them", C_HIGH),
        ("", C_HDR),
        ("if you chart for the MCP: tear it apart", C_AMBER),
        ("github.com/TrevorMcCormick/court-vision", C_WHITE),
    ]
    y = 210
    for text, c in lines:
        if text:
            cv2.putText(img, text, (60, y), FONT, 0.95, c, 2)
        y += 66
    cv2.putText(img, "trmccormick.com  |  the draft is the deliverable", (60, 620),
                FONT, 0.8, C_HDR, 1)
    for _ in range(int(END_CARD_S * OUT_FPS)):
        writer.write(img)


def main():
    rows = load_rows()
    raw = ROOT / "outputs" / "cv17_hero_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             OUT_FPS, (W, H))
    table_phase(writer, rows)
    excerpt_phase(writer, rows, EXCERPTS[0],
                  "the charter's job becomes: check it")
    excerpt_phase(writer, rows, EXCERPTS[1],
                  "five shots pre-filled; the fault stays human")
    split_card(writer)
    end_card(writer)
    writer.release()
    print(f"-> {raw}")

    out = ROOT / "outputs" / "cv17_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
