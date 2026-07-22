"""cv-21 hero: stick figures beat smudges — audit-still montage.

Cards + pose audit stills: a near-player skeleton, the far-player
skeleton at ~90 px, and the kept blooper (v1 grading a ball kid).

Run:  PYTHONPATH=.:experiments uv run python experiments/render_pose_hero.py
Out:  outputs/cv21_hero.mp4
"""

import subprocess

import cv2
import numpy as np

from courtvision.config import ROOT

W, H = 1280, 720
FPS = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
DIAG = ROOT / "outputs" / "diag"

SHOTS = [
    ("pose_t3_point_04_s2_near.png",
     "The near player, as the machine now sees him: wrists, elbows, hips, feet.", 3.0),
    ("pose_t3_point_19_s3_far.png",
     "The far player — 90 pixels tall, and still a readable swing.", 3.4),
    ("pose_blooper_ballkid.png",
     "Kept on the record: version 1 matched by blob box... and graded a ball kid.", 3.6),
    ("pose_t3_point_21_s3_far.png",
     "Version 1.1: candidates must STAND IN THE STRIKER'S HALF of the court.", 3.0),
]


def put_label(img, text, scale=0.74):
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
        [("For two weeks, forehand-vs-backhand has been stuck", 0.85),
         ("reading which side of a SMUDGE the ball arrives on.", 0.85),
         ("Today the smudges get skeletons.", 0.95)], 3.2)

    for name, caption, hold in SHOTS:
        img = cv2.imread(str(DIAG / name))
        if img is None:
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        img = img.copy()
        put_label(img, caption)
        frames += [img] * int(hold * FPS)

    frames += card(
        [("Same shots, same human answer key:", 0.9),
         ("clay: skeletons 84%, smudges 82%", 0.9),
         ("grass (the disaster surface): 66% vs 53%", 0.9),
         ("courtvision devlog #21", 0.65)], 3.4)

    out = ROOT / "outputs" / "cv21_hero_raw.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    final = ROOT / "outputs" / "cv21_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-c:v", "libx264", "-preset", "medium",
         "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
         str(final)], check=True, capture_output=True)
    print(f"[saved] {final}  ({len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
