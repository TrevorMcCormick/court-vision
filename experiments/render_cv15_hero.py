"""cv-15 hero: the draft, as the charter sees it.

Phase 1: the actual t3 export CSV (outputs/t3/export/t3_mcp_draft.csv)
scrolls into view row by row — high-confidence rows light green with
their machine strings, low rows stay dim — ending on the tally
(59 points, 16 flagged high). Phase 2: three high-flagged points play
as short excerpts with the WASB ball comet and their draft row
overlaid — the string, the flag, the jump-to timestamp a volunteer
charter would use. End card: the pooled LOMO numbers, said honestly.

Reads only outputs/t3/export/, outputs/t3/ball_wasb/, and
clips/points_t3/. Touches nothing in the pipeline.

Usage:
    uv run experiments/render_cv15_hero.py
"""

import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t3"
DRAFT_CSV = OUT_DIR / "export" / "t3_mcp_draft.csv"

W, H = 1280, 720
OUT_FPS = 24
FRAMES_PER_ROW = 2.4
TALLY_HOLD_S = 1.6
EXCERPTS = ["t3_point_28", "t3_point_07", "t3_point_02"]
EXCERPT_MAX_FRAMES = 80          # per excerpt, after sampling
TRAIL = 20
END_CARD_S = 3.0

C_HIGH = (80, 255, 120)          # green
C_LOW = (110, 110, 110)          # dim gray
C_HDR = (210, 210, 210)
C_AMBER = (0, 165, 255)
C_BALL = (80, 255, 120)
BG = (24, 24, 24)
ROW_BG_HIGH = (34, 52, 34)

FONT = cv2.FONT_HERSHEY_DUPLEX


def load_draft():
    rows = list(csv.DictReader(open(DRAFT_CSV)))
    assert rows, "empty draft csv"
    return rows


def draw_table_frame(rows, n_shown, flash_row=None):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "courtvision draft t3  ->  t3_mcp_draft.csv", (40, 52),
                FONT, 1.0, (255, 255, 255), 2)
    cv2.putText(img, "the draft, as the charter sees it (one row per point)",
                (40, 90), FONT, 0.65, C_HDR, 1)
    # header
    y0, row_h = 140, 28
    cols = [(40, "Pt"), (110, "score"), (240, "Svr"),
            (310, "1st (machine draft)"), (960, "conf"), (1100, "flag")]
    for x, name in cols:
        cv2.putText(img, name, (x, y0), FONT, 0.6, C_HDR, 1)
    cv2.line(img, (40, y0 + 8), (W - 40, y0 + 8), (70, 70, 70), 1)
    # viewport follows the newest row
    max_vis = 18
    first = max(0, n_shown - max_vis)
    for k in range(first, n_shown):
        r = rows[k]
        y = y0 + 40 + (k - first) * row_h
        hi = r["confidence"] == "high"
        if hi:
            cv2.rectangle(img, (34, y - 20), (W - 34, y + 6), ROW_BG_HIGH, -1)
        c = C_HIGH if hi else C_LOW
        cv2.putText(img, r["Pt"], (40, y), FONT, 0.6, c, 1)
        cv2.putText(img, r["Pts"], (110, y), FONT, 0.6, c, 1)
        cv2.putText(img, r["Svr"], (255, y), FONT, 0.6, c, 1)
        cv2.putText(img, r["1st"], (310, y), FONT, 0.6, c,
                    2 if hi else 1)
        cv2.putText(img, r["conf_p"], (960, y), FONT, 0.6, c, 1)
        cv2.putText(img, r["confidence"].upper() if hi else "low",
                    (1100, y), FONT, 0.6, c, 2 if hi else 1)
        if flash_row == k:
            cv2.rectangle(img, (34, y - 20), (W - 34, y + 6),
                          C_HIGH if hi else (90, 90, 90), 1)
    return img


def table_phase(writer, rows):
    n_total = len(rows)
    n_frames = int(n_total * FRAMES_PER_ROW)
    for i in range(n_frames):
        n_shown = min(n_total, int(i / FRAMES_PER_ROW) + 1)
        img = draw_table_frame(rows, n_shown, flash_row=n_shown - 1)
        writer.write(img)
    # tally hold
    n_high = sum(1 for r in rows if r["confidence"] == "high")
    img = draw_table_frame(rows, n_total)
    cv2.rectangle(img, (34, H - 90), (W - 34, H - 30), BG, -1)
    cv2.putText(img,
                f"{n_total} draft points  |  {n_high} flagged HIGH  ->  start from the draft",
                (40, H - 50), FONT, 0.9, C_HIGH, 2)
    for _ in range(int(TALLY_HOLD_S * OUT_FPS)):
        writer.write(img)


def load_ball(point):
    ball = {}
    with open(OUT_DIR / f"ball_wasb/ball_{point}.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx_raw"]),
                                       float(row["cy_raw"]))
    return ball


def excerpt_phase(writer, rows):
    by_clip = {r["clip"]: r for r in rows}
    for point in EXCERPTS:
        r = by_clip[point]
        ball = load_ball(point)
        cap = cv2.VideoCapture(str(ROOT / f"clips/points_t3/{point}.mp4"))
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
            # ball comet
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
            # draft-row banner, top
            cv2.rectangle(fr, (0, 0), (W, 96), BG, -1)
            cv2.putText(fr, "HIGH", (40, 40), FONT, 0.9, C_HIGH, 2)
            cv2.putText(fr, f"conf {r['conf_p']}", (150, 40), FONT, 0.7,
                        C_HIGH, 1)
            cv2.putText(fr, r["1st"], (40, 82), FONT, 1.1, (255, 255, 255), 2)
            cv2.putText(fr, f"{point}   jump to {r['serve_s']}s   {r['Pts']}",
                        (W - 520, 40), FONT, 0.65, C_HDR, 1)
            cv2.putText(fr, "start from the draft", (W - 520, 82), FONT,
                        0.65, C_HIGH, 1)
            writer.write(fr)


def end_card(writer):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "the confidence layer, held out (LOMO)", (60, 90),
                FONT, 1.3, (255, 255, 255), 2)
    lines = [
        ("HIGH = start from the draft:", (255, 255, 255)),
        ("41/44 (93%) within 5 token edits, at 32.6% coverage", C_HIGH),
        ("138 draft points across four matches, 46 flagged high", C_HDR),
        ("", C_HDR),
        ("not a sign-off: 27% of HIGH are within 2 edits,", C_AMBER),
        ("and the within-2 tier died in leave-one-match-out", C_AMBER),
    ]
    y = 190
    for text, c in lines:
        if text:
            cv2.putText(img, text, (60, y), FONT, 0.95, c, 2)
        y += 62
    cv2.putText(img, "courtvision draft <match>  |  $0 per match", (60, 620),
                FONT, 0.8, C_HDR, 1)
    for _ in range(int(END_CARD_S * OUT_FPS)):
        writer.write(img)


def main():
    rows = load_draft()
    raw = OUT_DIR / "cv15_hero_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             OUT_FPS, (W, H))
    table_phase(writer, rows)
    excerpt_phase(writer, rows)
    end_card(writer)
    writer.release()
    print(f"-> {raw}")

    out = OUT_DIR / "cv15_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
