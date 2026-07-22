"""cv-19 hero: the machine grows ears — g1_point_05, a 15-shot clay rally.

Layout: the broadcast frame on top; a sound strip along the bottom —
the waveform in gray, YELLOW spikes where the onset detector heard an
impact, GREEN ticks where the video pipeline detected a hit, RED shading
over the seconds where the ball track has holes (the eyes' blind time).
A playhead sweeps; "heard" flashes when an onset passes inside a red
zone. Intro and outro cards carry the claim and the numbers.

Run:  PYTHONPATH=.:experiments uv run python experiments/render_ears_hero.py
Out:  outputs/cv19_hero.mp4
"""

import subprocess

import cv2
import numpy as np

import audio_hits as ah
from courtvision import config
from courtvision.config import ROOT

CLIP = "g1_point_05"
MID = "g1"
OFFSET_S = 0.310                 # the calibrated g1 A/V offset
W, H = 1280, 720
STRIP_H = 130
FPS = 30.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_WAVE = (150, 150, 150)
COL_ONSET = (60, 220, 255)       # yellow — heard
COL_HIT = (120, 255, 120)        # green — seen
COL_BLIND = (60, 60, 200)        # red shade — eyes blind
PAD_END_S = 1.5


def put_label(img, text, org, scale=0.85, color=(255, 255, 255), thick=2):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 8, y - th - 10), (x + tw + 8, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def card(lines, seconds):
    img = np.zeros((H, W, 3), np.uint8)
    y = H // 2 - 30 * len(lines)
    for i, (text, scale) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(img, text, ((W - tw) // 2, y + i * 62), FONT, scale,
                    (235, 235, 235), 2, cv2.LINE_AA)
    return [img] * int(seconds * FPS)


def main():
    wav = ah.load_wav(ROOT / "clips" / "audio" / f"{MID}.wav")
    clips = ah.match_data(MID)
    e = clips[CLIP]
    a = int(e["start_s"] * ah.SR)
    b = int((e["start_s"] + e["dur_s"]) * ah.SR)
    seg = wav[a:b]
    onsets, _ = ah.spectral_flux_onsets(seg)
    onsets_v = onsets - OFFSET_S             # onset times on the video clock

    end_s = max(e["hits"]) + PAD_END_S       # trim the applause tail
    n_show = int(end_s * FPS)

    # waveform envelope, one column per strip pixel
    x = seg[: int(end_s * ah.SR)].astype(np.float32) / 32768.0
    cols = np.array_split(np.abs(x), W)
    env = np.array([c.max() if len(c) else 0 for c in cols])
    env = (env / (env.max() + 1e-9)) ** 0.6  # perceptual-ish scaling

    # blind zones (track holes) between serve and last hit
    tf, sv = e["track_frames"], e["serve_s"]
    lo, hi = sv * FPS, max(e["hits"]) * FPS + FPS
    w = tf[(tf >= lo) & (tf <= hi)]
    holes = [(p / FPS, f / FPS) for p, f in zip(w[:-1], w[1:]) if f - p > 3]

    def strip(img, t_now):
        y0 = H - STRIP_H
        cv2.rectangle(img, (0, y0), (W, H), (18, 18, 18), -1)
        for (ha, hb) in holes:                       # blind shading
            xa, xb = int(ha / end_s * W), int(hb / end_s * W)
            cv2.rectangle(img, (xa, y0), (xb, H), COL_BLIND, -1)
        mid = y0 + STRIP_H // 2
        for px in range(W):                          # waveform
            h2 = int(env[px] * (STRIP_H // 2 - 22))
            cv2.line(img, (px, mid - h2), (px, mid + h2), COL_WAVE, 1)
        for h in e["hits"]:                          # green: eyes saw a hit
            px = int(h / end_s * W)
            cv2.line(img, (px, y0 + 4), (px, y0 + 26), COL_HIT, 3)
        for t in onsets_v:                           # yellow: ears heard
            if t > end_s:
                continue
            px = int(t / end_s * W)
            cv2.line(img, (px, H - 30), (px, H - 6), COL_ONSET, 3)
        px = int(t_now / end_s * W)                  # playhead
        cv2.line(img, (px, y0), (px, H), (255, 255, 255), 2)
        cv2.putText(img, "seen (video)", (12, y0 + 22), FONT, 0.55,
                    COL_HIT, 1, cv2.LINE_AA)
        cv2.putText(img, "heard (audio)", (12, H - 12), FONT, 0.55,
                    COL_ONSET, 1, cv2.LINE_AA)
        # flash when an onset fires inside a blind zone
        for t in onsets_v:
            if abs(t - t_now) < 0.12 and any(ha <= t <= hb for ha, hb in holes):
                put_label(img, "HEARD — eyes blind here",
                          (W // 2 - 190, y0 - 16), 0.9, COL_ONSET)

    frames = card(
        [("The machine grows ears.", 1.5),
         ("Every video on disk was silent. The soundtrack came back", 0.7),
         ("with receipts: a 15-shot rally at Roland Garros.", 0.7)], 2.8)

    cap = cv2.VideoCapture(str(ROOT / "clips" / f"points_{MID}" / f"{CLIP}.mp4"))
    f = 0
    while f < n_show:
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        strip(img, f / FPS)
        put_label(img, "yellow = the mic. green = the eyes. red = eyes blind.",
                  (40, 44), 0.7)
        frames.append(img)
        f += 1
    cap.release()

    frames += card(
        [("Half of everything the eyes saw, the mic confirmed", 0.8),
         ("to within 2-3 frames. And in 78 seconds of clay-court", 0.8),
         ("blindness, the mic logged 196 impacts the eyes missed.", 0.8),
         ("courtvision devlog #19", 0.65)], 3.2)

    out = ROOT / "outputs" / "cv19_hero_raw.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (W, H))
    for img in frames:
        vw.write(img)
    vw.release()
    final = ROOT / "outputs" / "cv19_hero.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-c:v", "libx264", "-preset",
         "medium", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags",
         "+faststart", "-an", str(final)], check=True, capture_output=True)
    print(f"[saved] {final}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
