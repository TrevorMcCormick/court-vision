"""cv-10 hero: homography A/B on the t1 night plate.

Projects the court model onto outputs/t1/plate_fit.png with two
homographies:
  (a) the dev-reel fit, outputs/m1/H_court_to_img.npy  — red,
      "dev homography (transferred)"
  (b) the final t1 fit, outputs/t1/H_court_to_img.npy  — green,
      "manual 4-corner fit"

Alternates a/b every ~1.2 s for ~7 s, then holds a final segment with
both overlaid. Court model lines drawn exactly as t1_fit_homography.py
validates them (baselines, doubles + singles sidelines, service lines,
center service line).

Usage:
    uv run experiments/render_cv10_hero.py
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "t1"

W_COURT = 10.97
L_COURT = 23.77
SINGLES_INSET = 1.372
NET_Y = L_COURT / 2
SVC_FAR_Y = NET_Y - 6.40
SVC_NEAR_Y = NET_Y + 6.40
CENTER_X = W_COURT / 2

MODEL_LINES = [
    ((0, 0), (W_COURT, 0)),                                # far baseline
    ((0, L_COURT), (W_COURT, L_COURT)),                    # near baseline
    ((0, 0), (0, L_COURT)),                                # doubles sidelines
    ((W_COURT, 0), (W_COURT, L_COURT)),
    ((SINGLES_INSET, 0), (SINGLES_INSET, L_COURT)),        # singles sidelines
    ((W_COURT - SINGLES_INSET, 0), (W_COURT - SINGLES_INSET, L_COURT)),
    ((SINGLES_INSET, SVC_FAR_Y), (W_COURT - SINGLES_INSET, SVC_FAR_Y)),
    ((SINGLES_INSET, SVC_NEAR_Y), (W_COURT - SINGLES_INSET, SVC_NEAR_Y)),
    ((CENTER_X, SVC_FAR_Y), (CENTER_X, SVC_NEAR_Y)),       # center service
]

FPS = 30
SEG_S = 1.2
N_ALT = 6          # a,b,a,b,a,b -> 7.2 s
BOTH_S = 2.8

C_DEV = (60, 60, 235)    # red (BGR)
C_T1 = (80, 220, 80)     # green


def project_lines(img, H, color, thick=3):
    for (x1, y1), (x2, y2) in MODEL_LINES:
        # densify so perspective is exact along the segment
        ts = np.linspace(0, 1, 32)
        pts = np.float32([(x1 + t * (x2 - x1), y1 + t * (y2 - y1)) for t in ts])
        proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        for a, b in zip(proj[:-1], proj[1:]):
            cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), color, thick,
                     cv2.LINE_AA)


def label(img, text, color, y=54):
    cv2.putText(img, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(img, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                color, 3, cv2.LINE_AA)


def main():
    plate = cv2.imread(str(OUT_DIR / "plate_fit.png"))
    H_dev = np.load(ROOT / "outputs/m1/H_court_to_img.npy")
    H_t1 = np.load(OUT_DIR / "H_court_to_img.npy")

    frame_a = plate.copy()
    project_lines(frame_a, H_dev, C_DEV)
    label(frame_a, "dev homography (transferred)", C_DEV)

    frame_b = plate.copy()
    project_lines(frame_b, H_t1, C_T1)
    label(frame_b, "manual 4-corner fit", C_T1)

    frame_both = plate.copy()
    project_lines(frame_both, H_dev, C_DEV, thick=2)
    project_lines(frame_both, H_t1, C_T1, thick=2)
    label(frame_both, "dev homography (transferred)", C_DEV)
    label(frame_both, "manual 4-corner fit", C_T1, y=100)

    h, w = plate.shape[:2]
    raw = OUT_DIR / "cv10_hero_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (w, h))
    seg_frames = int(SEG_S * FPS)
    for k in range(N_ALT):
        frame = frame_a if k % 2 == 0 else frame_b
        for _ in range(seg_frames):
            writer.write(frame)
    for _ in range(int(BOTH_S * FPS)):
        writer.write(frame_both)
    writer.release()
    total = N_ALT * seg_frames + int(BOTH_S * FPS)
    print(f"-> {raw} ({total} frames, {total / FPS:.1f} s)")

    out = OUT_DIR / "cv10_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(raw),
         "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(out)], check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
