"""Auto-pick a static full-court-view window for fitcourt.

Scan the reel; a frame is a MAIN COURT VIEW when the clay hull band fills a
large fraction of the central image AND the score bug is present. Among
court-view frames find contiguous runs, then inside the longest runs pick
the ~4 s sub-window with the lowest inter-frame camera motion (ECC needs
low drift for a crisp median plate). Prints the best (lo,hi).
"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import read_bug as rb

VIDEO = "clips/g1_swiatek_paolini_30fps.mp4"
HULL_LO = (0, 60, 120); HULL_HI = (15, 255, 255)
STEP = 15
WIN_S = 4.0


def scan():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []          # (frame, court_bool, small_gray)
    prev = None
    i = 0
    while i < n:
        if not cap.grab():
            break
        if i % STEP == 0:
            ok, fr = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                central = hsv[220:600, 300:980]
                clay = cv2.inRange(central, HULL_LO, HULL_HI).mean() / 255.0
                pres = rb._present(fr) > 0.08
                court = clay > 0.45 and pres
                g = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY)
                samples.append((i, court, g))
        i += 1
    cap.release()
    return samples, fps


def best_window(samples, fps):
    # contiguous court-view runs
    runs, s = [], None
    for idx, (f, court, g) in enumerate(samples):
        if court and s is None:
            s = idx
        elif not court and s is not None:
            runs.append((s, idx - 1)); s = None
    if s is not None:
        runs.append((s, len(samples) - 1))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    win = int(WIN_S * fps / STEP)
    best = None
    for (a, b) in runs[:6]:
        for start in range(a, b - win + 1):
            drift = 0.0
            for k in range(start + 1, start + win):
                d = np.abs(samples[k][2].astype(np.int16) - samples[k-1][2].astype(np.int16)).mean()
                drift += d
            drift /= win
            f0 = samples[start][0]; f1 = samples[start + win][0]
            if best is None or drift < best[0]:
                best = (drift, f0, f1, samples[a][0], samples[b][0])
    return best, runs


if __name__ == "__main__":
    samples, fps = scan()
    nc = sum(1 for s in samples if s[1])
    print(f"samples={len(samples)} court-view={nc} ({100*nc/len(samples):.0f}%)")
    best, runs = best_window(samples, fps)
    print("top court-view runs (frames):",
          [(samples[a][0], samples[b][0]) for a, b in runs[:5]])
    if best:
        print(f"BEST fit window: fit_lo={best[1]} fit_hi={best[2]} "
              f"(mean drift {best[0]:.2f}px/small-frame) within run {best[3]}-{best[4]}")
