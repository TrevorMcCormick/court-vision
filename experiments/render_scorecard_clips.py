"""Purpose-made scorecard card loops (serve / direction / serve-zone).

Three ~3 s, 480 px, muted loops for the scorecard grid, drawn from t6
(US Open) data the pipeline already computed. The other cards' loops are
currently cut from the devlog hero videos; these three had nothing to
cut from.

Run:  PYTHONPATH=.:experiments uv run python experiments/render_scorecard_clips.py
Out:  docs/clips/serve.mp4, direction.mp4, zone.mp4
"""

import csv
import subprocess

import cv2
import numpy as np

from courtvision import config
from courtvision.config import ROOT
from courtvision.court import W_C, L_C, NET_Y, SINGLES_MARGIN

MID = "t6"
FPS = 30
OUT = ROOT / "docs" / "clips"
COL = (60, 220, 255)


def encode(frames, name, w, h):
    raw = OUT / f"_{name}_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-vf", "scale=480:-2",
         "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(OUT / f"{name}.mp4")],
        check=True)
    raw.unlink()
    print(f"[saved] docs/clips/{name}.mp4 ({len(frames)/FPS:.1f}s)")


def clip_frames(clip, f0, f1):
    cap = cv2.VideoCapture(str(ROOT / "clips" / f"points_{MID}" / f"{clip}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0))
    out = []
    for _ in range(f1 - f0):
        ok, img = cap.read()
        if not ok:
            break
        out.append(img)
    cap.release()
    return out


def main():
    OUT.mkdir(exist_ok=True)
    cfg = config.load(MID)
    Hc2i = np.load(cfg.out_dir / "H_court_to_img.npy")
    serves = cfg.load_serves()

    def court_line(img, a, b, color, th=2):
        p = cv2.perspectiveTransform(
            np.float32([a, b]).reshape(-1, 1, 2), Hc2i).reshape(-1, 2)
        cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)),
                 color, th, cv2.LINE_AA)

    # ---- serve.mp4: the serve moment, server ringed ----------------------
    clip = "t6_point_05"
    srow = serves[clip]
    sf = int(float(srow["serve_s"]) * FPS)
    players = {(r["player"], int(r["frame"])): r for r in csv.DictReader(
        open(cfg.out_dir / "players" / f"players_{clip}.csv"))}
    frames = []
    for i, img in enumerate(clip_frames(clip, sf - 25, sf + 55)):
        f = sf - 25 + i
        r = players.get((srow["server"], f))
        if r:
            cx, cy = float(r["cx"]) * 1280, float(r["cy"]) * 720
            rad = 34 + int(6 * np.sin(i / 3.5))
            cv2.circle(img, (int(cx), int(cy)), rad, COL, 3, cv2.LINE_AA)
        frames.append(img)
    encode(frames, "serve", 1280, 720)

    # ---- direction.mp4: one shot's flight with an arrow ------------------
    clip = "t6_point_08"
    chart = list(csv.DictReader(open(cfg.charts_dir / f"chart2_{clip}.csv")))
    ball = {int(r["frame"]): (float(r["x_stab"]), float(r["y_stab"]))
            for r in csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv"))}
    c1 = int(float(chart[2]["contact_frame"]))
    c2 = int(float(chart[3]["contact_frame"]))
    frames = []
    for i, img in enumerate(clip_frames(clip, c1 - 6, c2 + 10)):
        f = c1 - 6 + i
        pts = [ball[k] for k in range(c1, min(f, c2) + 1) if k in ball]
        for p in pts:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, COL, -1, cv2.LINE_AA)
        if len(pts) >= 2 and f >= c2 - 2:
            a, b = np.array(pts[0]), np.array(pts[-1])
            cv2.arrowedLine(img, tuple(a.astype(int)), tuple(b.astype(int)),
                            (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.06)
        frames.append(img)
    encode(frames, "direction", 1280, 720)

    # ---- zone.mp4: service box split into thirds + landing dot -----------
    clip = "t6_point_05"
    srow = serves[clip]
    srv_shot = next(r for r in csv.DictReader(
        open(cfg.charts_dir / f"chart2_{clip}.csv")) if r["is_serve"] == "True")
    sf = int(float(srow["serve_s"]) * FPS)
    lx, ly = float(srv_shot["landing_x"]), float(srv_shot["landing_y"])
    # receiver's service box: the half the serve lands in
    far_box = ly < NET_Y
    y0, y1 = (NET_Y - 6.40, NET_Y) if far_box else (NET_Y, NET_Y + 6.40)
    xs = np.linspace(SINGLES_MARGIN, W_C - SINGLES_MARGIN, 4)
    land_px = cv2.perspectiveTransform(
        np.float32([[lx, ly]]).reshape(-1, 1, 2), Hc2i).reshape(2)
    frames = []
    for i, img in enumerate(clip_frames(clip, sf - 10, sf + 75)):
        for x in xs:
            court_line(img, (x, y0), (x, y1), COL, 2)
        court_line(img, (xs[0], y0), (xs[-1], y0), COL, 2)
        court_line(img, (xs[0], y1), (xs[-1], y1), COL, 2)
        if i > 30:
            cv2.circle(img, tuple(land_px.astype(int)), 9, (0, 0, 255), -1,
                       cv2.LINE_AA)
        frames.append(img)
    encode(frames, "zone", 1280, 720)


if __name__ == "__main__":
    main()
