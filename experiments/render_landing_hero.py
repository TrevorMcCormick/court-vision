"""cv-18 hero: the boundary race, drawn on real broadcast frames.

Two US Open points (t6) whose true ending the racer got right:
  t6_point_110 — truth WIDE, called by extrapolation (t_side ~11 frames
                 past track death, baseline never crossed)
  t6_point_117 — truth DEEP, already beyond the baseline at track death

Per point: play the final seconds with the tracked ball trailed, freeze
at track death, grow the extrapolated flight line toward the two finish
lines (sideline = wide, baseline = deep), flash the winner, stamp the
call vs the human charter. Muted, ~18 s total, 720p30.

Run:  PYTHONPATH=.:experiments uv run python experiments/render_landing_hero.py
Out:  outputs/cv18_hero.mp4
"""

import csv
import subprocess

import cv2
import numpy as np

from courtvision import config
from courtvision.config import ROOT
from courtvision.court import L_C, SINGLES_MARGIN, W_C
from landing_spot import (FPS, TAIL_K, final_flight_segment, infer_target_far,
                          _court_track)

POINTS = [
    ("t6", "t6_point_110", "w", "WIDE", "sideline first"),
    ("t6", "t6_point_117", "d", "DEEP", "baseline first"),
]
W, H = 1280, 720
TRAIL = 18                       # trailing ball dots kept on screen
PRE_S = 2.0                      # seconds of play shown before track death
FREEZE_S = 3.2                   # freeze-frame race duration
HOLD_S = 1.6                     # verdict hold
GROW_FRAMES = 45                 # frames the extrapolation takes to grow
COL_TRAIL = (60, 220, 255)       # ball trail (warm yellow)
COL_EXTRAP = (255, 255, 255)     # extrapolated flight
COL_SIDE = (0, 165, 255)         # sideline finish line (orange)
COL_BASE = (255, 200, 0)         # baseline finish line (cyan-ish)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def court_to_img(Hc2i, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hc2i).reshape(-1, 2)


def put_label(img, text, org, scale=0.9, color=(255, 255, 255), thick=2):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 8, y - th - 10), (x + tw + 8, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def title_card(lines, seconds):
    frames = []
    img = np.zeros((H, W, 3), np.uint8)
    y = H // 2 - 20 * len(lines)
    for i, (text, scale) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(img, text, ((W - tw) // 2, y + i * 60), FONT, scale,
                    (235, 235, 235), 2, cv2.LINE_AA)
    frames.extend([img] * int(seconds * FPS))
    return frames


def render_point(mid, clip, truth, verdict, why):
    cfg = config.load(mid)
    Hm = np.load(cfg.homography)
    Hc2i = np.load(cfg.homography.parent / "H_court_to_img.npy")
    offsets = cfg.load_offsets()
    track = _court_track(cfg, clip, Hm, offsets)
    chart = list(csv.DictReader(open(cfg.charts_dir / f"chart2_{clip}.csv")))
    last = chart[-1]
    contact = float(last["contact_frame"] or last["frame"])
    seg_f, seg_x, seg_y = final_flight_segment(track, contact)

    # the racer's fit, recomputed exactly as race_boundaries does
    k = min(TAIL_K, len(seg_f))
    t, xs, ys = seg_f[-k:], seg_x[-k:], seg_y[-k:]
    bx = np.polyfit(t, xs, 1)[0]
    by = np.polyfit(t, ys, 1)[0]
    x0, y0 = xs[-1], ys[-1]

    target_far = infer_target_far(y0, by)
    side_x = (W_C - SINGLES_MARGIN) if bx > 0 else SINGLES_MARGIN

    # finish lines in image space
    side_line = court_to_img(Hc2i, [(side_x, -3.0), (side_x, L_C + 3.0)])
    by0 = 0.0 if target_far else L_C
    base_line = court_to_img(Hc2i, [(-2.0, by0), (W_C + 2.0, by0)])

    # extrapolated path in image space (grown over GROW_FRAMES)
    horizon = np.linspace(0, 28.0, GROW_FRAMES)          # frames of flight
    extrap = court_to_img(Hc2i, [(x0 + bx * h, y0 + by * h) for h in horizon])

    # image positions of the observed track (stabilised coords)
    ball = list(csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv")))
    img_xy = {int(r["frame"]): (float(r["x_stab"]), float(r["y_stab"]))
              for r in ball}
    death = int(seg_f[-1])

    cap = cv2.VideoCapture(str(ROOT / "clips" / f"points_{mid}" / f"{clip}.mp4"))
    n_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, death - int(PRE_S * FPS))
    frames_out = []

    def draw_trail(img, upto):
        pts = [img_xy[f] for f in range(max(0, upto - TRAIL), upto + 1)
               if f in img_xy]
        for i, (px, py) in enumerate(pts):
            r = 3 + (4 * i) // max(1, len(pts))
            cv2.circle(img, (int(px), int(py)), r, COL_TRAIL, -1, cv2.LINE_AA)

    # phase A — live play up to track death
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(start, min(death + 1, n_clip)):
        ok, img = cap.read()
        if not ok:
            break
        draw_trail(img, f)
        put_label(img, "the tracker is watching the ball...", (40, H - 40), 0.8)
        frames_out.append(img)

    # phase B — freeze at death, race the boundaries
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(death, n_clip - 1))
    ok, base_img = cap.read()
    if not ok:
        raise RuntimeError(f"no freeze frame for {clip}")
    draw_trail(base_img, death)
    lx, ly = img_xy[death]
    n_freeze = int(FREEZE_S * FPS)
    for i in range(n_freeze):
        img = base_img.copy()
        cv2.polylines(img, [side_line.astype(int)], False, COL_SIDE, 3, cv2.LINE_AA)
        cv2.polylines(img, [base_line.astype(int)], False, COL_BASE, 3, cv2.LINE_AA)
        put_label(img, "sideline = wide", (int(side_line[0][0]) - 60, 90),
                  0.7, COL_SIDE)
        put_label(img, "baseline = deep", (60, int(base_line[0][1]) - 14),
                  0.7, COL_BASE)
        cv2.circle(img, (int(lx), int(ly)), 9, (0, 0, 255), 2, cv2.LINE_AA)
        put_label(img, "...and loses it. mid-air.", (40, H - 40), 0.8)
        grown = min(GROW_FRAMES, max(2, int(i / (n_freeze * 0.75) * GROW_FRAMES)))
        cv2.polylines(img, [extrap[:grown].astype(int)], False, COL_EXTRAP,
                      2, cv2.LINE_AA)
        if grown >= GROW_FRAMES:
            col = COL_SIDE if truth == "w" else COL_BASE
            put_label(img, f"call: {verdict} — {why}", (40, 80), 1.1, col, 3)
        frames_out.append(img)

    # phase C — verdict hold
    img = frames_out[-1].copy()
    put_label(img, f"human charter: {verdict} — correct",
              (40, 140), 0.9, (120, 255, 120))
    frames_out.extend([img] * int(HOLD_S * FPS))
    cap.release()
    return frames_out


def main():
    frames = title_card(
        [("Nobody sees the ball land.", 1.5),
         ("Not even Hawk-Eye. The landing is calculated from the flight.", 0.7)],
        2.6)
    for spec in POINTS:
        frames += render_point(*spec)
    frames += title_card(
        [("Wide-or-deep, from flight physics alone:", 0.9),
         ("0% -> half named right (83 of 169 out-balls, 8 matches)", 0.9),
         ("courtvision devlog #18", 0.7)],
        3.0)

    out = ROOT / "outputs" / "cv18_hero_raw.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    final = ROOT / "outputs" / "cv18_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-c:v", "libx264", "-preset", "medium",
         "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", str(final)], check=True, capture_output=True)
    print(f"[saved] {final}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
