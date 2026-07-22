"""cv-20 hero: the court that fits itself — overlay montage.

Cards + the auto-fit overlay stills (green = the human's hand fit,
orange = the pretrained net's fit), holding on the g1 clay reveal, the
t1 night-feed abstention, and the t3 twist. ~28 s, 720p30, silent.

Run:  PYTHONPATH=.:experiments uv run python experiments/render_court_hero.py
Out:  outputs/cv20_hero.mp4
"""

import subprocess

import cv2
import numpy as np

from courtvision.config import ROOT

W, H = 1280, 720
FPS = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
DIAG = ROOT / "outputs" / "diag"

SHOTS = [                       # (image, caption, hold_s)
    ("autofit_t6.png", "US Open. green = the human's 4 clicks. orange = the machine, no clicks.", 2.6),
    ("autofit_t5.png", "Melbourne. 3 pixels apart.", 1.8),
    ("autofit_t7.png", "Turin, indoors. 5 pixels.", 1.8),
    ("autofit_t2.png", "Montreal. 7 pixels.", 1.8),
    ("autofit_t4.png", "Wimbledon. 8 pixels.", 1.8),
    ("autofit_g1.png", "Roland Garros — the plate the old auto-fitter missed by 74 pixels. Fit, first try.", 4.0),
    ("autofit_t3.png", "And a twist: here the MACHINE hugs the paint. The 'error' may be the human's.", 3.6),
]


def put_label(img, text, scale=0.78):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x, y = (W - tw) // 2, H - 34
    cv2.rectangle(img, (x - 10, y - th - 12), (x + tw + 10, y + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)


def card(lines, seconds):
    img = np.zeros((H, W, 3), np.uint8)
    y = H // 2 - 30 * len(lines)
    for i, (text, scale) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(img, text, ((W - tw) // 2, y + i * 62), FONT, scale,
                    (235, 235, 235), 2, cv2.LINE_AA)
    return [img] * int(seconds * FPS)


def main():
    frames = card(
        [("The last manual knob:", 1.1),
         ("a human clicks 4 corners, every match.", 0.85),
         ("Today a free pretrained net tries to take the job.", 0.75)], 3.0)

    for name, caption, hold in SHOTS:
        img = cv2.imread(str(DIAG / name))
        if img is None:
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        img = img.copy()
        put_label(img, caption)
        frames += [img] * int(hold * FPS)

    night = cv2.imread(str(ROOT / "outputs" / "t1" / "plate_fit.png"))
    if night is not None:
        night = cv2.resize(night, (W, H))
        dark = (night * 0.85).astype(np.uint8)
        put_label(dark, "The night match: 2 of 14 landmarks found. The machine ABSTAINS and asks a human.")
        frames += [dark] * int(3.2 * FPS)

    frames += card(
        [("7 of 8 courts fit themselves, 2-16 pixels", 0.9),
         ("from the human's hand. The 8th asked for help", 0.9),
         ("instead of guessing. That's the whole design.", 0.9),
         ("courtvision devlog #20", 0.65)], 3.4)

    out = ROOT / "outputs" / "cv20_hero_raw.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    final = ROOT / "outputs" / "cv20_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-c:v", "libx264", "-preset", "medium",
         "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
         str(final)], check=True, capture_output=True)
    print(f"[saved] {final}  ({len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
