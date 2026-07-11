"""cv-16 hero: the benchmark at scale.

Phase 1: a wall of the seven benchmark matches — one representative
frame each, labeled with the match and its aligned human-charted point
count — landing one by one while the total rolls 135 -> 491 (the three
new matches land last, highlighted). Phase 2: the calibration numbers
correcting themselves: 93% @ 32.6% (4 folds) -> 88% @ 21.2% (7 folds)
-> 94% @ 19.6% (t4 whole-point gates). Phase 3: one high-flagged point
from a NEW match (t6, US Open F 2024) plays with the WASB ball comet
and its draft string. End card: the honest claim.

Reads only clips/points_t*/, outputs/t*/export/, outputs/t6/ball_wasb/.
Touches nothing in the pipeline.

Usage:
    uv run experiments/render_cv16_hero.py
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

# (match id, rep clip, label, points scored, is_new)
MATCHES = [
    ("t1", "t1_point_03", "t1 Nadal-Shapovalov Canada 2017", 22, False),
    ("t2", "t2_point_01", "t2 Federer-Haase Canada 2017", 5, False),
    ("t3", "t3_point_07", "t3 Djokovic-Ruud RG F 2023", 59, False),
    ("t4", "t4_point_10", "t4 Krejcikova-Paolini Wimbledon F 2024", 49, False),
    ("t5", "t5_point_10", "t5 Sinner-Zverev AO F 2025", 71, True),
    ("t6", "t6_point_44", "t6 Sabalenka-Pegula US Open F 2024", 128, True),
    ("t7", "t7_point_20", "t7 Djokovic-Sinner ATP Finals 2023", 157, True),
]

TILE_W, TILE_H = 290, 163
GRID_X0, GAP_X = 32, 18
ROW_Y = [112, 344]

CLOSER = "t6_point_44"           # HIGH 0.902, s5b2f1f2f2x@
CLOSER_MATCH = "t6"
EXCERPT_MAX_FRAMES = 110
TRAIL = 20
END_CARD_S = 3.2


def tile_xy(k):
    row, col = divmod(k, 4)
    return GRID_X0 + col * (TILE_W + GAP_X), ROW_Y[row]


def load_tiles():
    tiles = []
    for _, clip, _, _, _ in MATCHES:
        t = clip.split("_")[0]
        cap = cv2.VideoCapture(str(ROOT / f"clips/points_{t}/{clip}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.45))
        ok, fr = cap.read()
        cap.release()
        assert ok, clip
        tiles.append(cv2.resize(fr, (TILE_W, TILE_H)))
    return tiles


def draw_grid_frame(tiles, n_shown, counter, caption, caption_c):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "the benchmark at scale: seven matches, five feeds",
                (32, 52), FONT, 0.95, C_WHITE, 2)
    cv2.putText(img, "every point aligned to a human MCP chart via the score bug",
                (32, 84), FONT, 0.58, C_HDR, 1)
    for k in range(min(n_shown, len(MATCHES))):
        _, _, label, pts, is_new = MATCHES[k]
        x, y = tile_xy(k)
        img[y:y + TILE_H, x:x + TILE_W] = tiles[k]
        border = C_HIGH if is_new else (90, 90, 90)
        cv2.rectangle(img, (x - 1, y - 1), (x + TILE_W, y + TILE_H), border,
                      2 if is_new else 1)
        cv2.putText(img, label, (x, y + TILE_H + 22), FONT, 0.42,
                    C_HIGH if is_new else C_HDR, 1)
        cv2.putText(img, f"{pts} aligned points", (x, y + TILE_H + 44),
                    FONT, 0.48, C_WHITE if is_new else C_DIM, 1)
    # counter cell (8th slot)
    x, y = tile_xy(7)
    cv2.rectangle(img, (x - 1, y - 1), (x + TILE_W, y + TILE_H),
                  (70, 70, 70), 1)
    txt = str(int(round(counter)))
    (tw, _), _ = cv2.getTextSize(txt, FONT, 1.9, 3)
    cv2.putText(img, txt, (x + (TILE_W - tw) // 2, y + 95), FONT, 1.9,
                C_HIGH, 3)
    cv2.putText(img, "human-charted points", (x + 18, y + 135), FONT, 0.5,
                C_HDR, 1)
    if caption:
        cv2.putText(img, caption, (32, H - 36), FONT, 0.75, caption_c, 2)
    return img


def grid_phase(writer, tiles):
    """Old four land, hold at 135; new three land, hold at 491."""
    cum = np.cumsum([m[3] for m in MATCHES]).tolist()
    counter = 0.0

    def land(k, seconds, caption, caption_c):
        nonlocal counter
        n = int(seconds * OUT_FPS)
        target = cum[k]
        start = counter
        for i in range(n):
            counter = start + (target - start) * (i + 1) / n
            writer.write(draw_grid_frame(tiles, k + 1, counter,
                                         caption, caption_c))

    def hold(n_shown, seconds, caption, caption_c):
        img = draw_grid_frame(tiles, n_shown, counter, caption, caption_c)
        for _ in range(int(seconds * OUT_FPS)):
            writer.write(img)

    for k in range(4):
        land(k, 0.6, "", C_HDR)
    hold(4, 1.4, "the 4-match benchmark cv-15 shipped on: 135 points",
         C_HDR)
    for k in range(4, 7):
        land(k, 1.0, "three new matches in one pass, zero new scripts",
             C_HIGH)
    hold(7, 1.8, "3.6x the ground truth, $0 marginal cost", C_HIGH)


def roll_text(a, b, t):
    return a + (b - a) * t


def draw_calib_frame(prec, cov, stage_lines):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "the recalibration (leave-one-match-out, held out)",
                (60, 70), FONT, 1.0, C_WHITE, 2)
    cv2.putText(img, "HIGH flag = 'start from the draft' (within 5 token edits)",
                (60, 108), FONT, 0.6, C_HDR, 1)
    big = f"{prec:.0f}%  @  {cov:.1f}% coverage"
    cv2.putText(img, big, (110, 250), FONT, 2.2, C_HIGH, 4)
    y = 360
    for text, note, c in stage_lines:
        cv2.putText(img, text, (90, y), FONT, 0.8, c, 2)
        if note:
            cv2.putText(img, note, (640, y), FONT, 0.7, C_AMBER, 1)
        y += 70
    return img


def calib_phase(writer):
    stages = [
        (93.0, 32.6, "4 matches, 44 flags       93% @ 32.6%",
         "<- small-n flattery"),
        (88.0, 21.2, "7 matches, 104 flags      88% @ 21.2%",
         "<- more data, honest number"),
        (94.0, 19.6, "+ t4 whole-point gates    94% @ 19.6%",
         "<- disasters halved, 12 -> 6"),
    ]
    lines = []

    def hold(prec, cov, seconds):
        img = draw_calib_frame(prec, cov, lines)
        for _ in range(int(seconds * OUT_FPS)):
            writer.write(img)

    def roll(a, b, seconds):
        n = int(seconds * OUT_FPS)
        for i in range(n):
            t = (i + 1) / n
            writer.write(draw_calib_frame(roll_text(a[0], b[0], t),
                                          roll_text(a[1], b[1], t), lines))

    prev = None
    for prec, cov, text, note in stages:
        if prev is not None:
            roll(prev, (prec, cov), 0.9)
        lines.append((text, note, C_WHITE))
        hold(prec, cov, 1.6 if prev is not None else 1.3)
        prev = (prec, cov)
    hold(*prev, 0.8)


def excerpt_phase(writer):
    draft = ROOT / f"outputs/{CLOSER_MATCH}/export/{CLOSER_MATCH}_mcp_draft.csv"
    r = next(row for row in csv.DictReader(open(draft))
             if row["clip"] == CLOSER)
    ball = {}
    with open(ROOT / f"outputs/{CLOSER_MATCH}/ball_wasb/ball_{CLOSER}.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx_raw"]),
                                       float(row["cy_raw"]))
    cap = cv2.VideoCapture(str(ROOT / f"clips/points_{CLOSER_MATCH}/{CLOSER}.mp4"))
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
        cv2.putText(fr, f"{CLOSER}   jump to {r['serve_s']}s   {r['Pts']}",
                    (W - 560, 40), FONT, 0.65, C_HDR, 1)
        cv2.putText(fr, "a feed the held-out model had never seen",
                    (W - 560, 82), FONT, 0.65, C_HIGH, 1)
        writer.write(fr)


def end_card(writer):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.putText(img, "the benchmark grows up", (60, 100), FONT, 1.4,
                C_WHITE, 2)
    lines = [
        ("seven matches, five feeds, 491 human-charted points", C_HDR),
        ("HIGH = start from the draft: 90/96 (94%) held out", C_HIGH),
        ("at 19.6% coverage; high-tier disasters 12 -> 6", C_HIGH),
        ("", C_HDR),
        ("more data made the number honest, not better", C_AMBER),
    ]
    y = 210
    for text, c in lines:
        if text:
            cv2.putText(img, text, (60, y), FONT, 0.95, c, 2)
        y += 66
    cv2.putText(img, "courtvision draft <match>  |  $0 per match", (60, 620),
                FONT, 0.8, C_HDR, 1)
    for _ in range(int(END_CARD_S * OUT_FPS)):
        writer.write(img)


def main():
    tiles = load_tiles()
    raw = ROOT / "outputs" / "cv16_hero_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             OUT_FPS, (W, H))
    grid_phase(writer, tiles)
    calib_phase(writer)
    excerpt_phase(writer)
    end_card(writer)
    writer.release()
    print(f"-> {raw}")

    out = ROOT / "outputs" / "cv16_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
