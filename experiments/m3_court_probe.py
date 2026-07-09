"""M3 experiment 3, take 2 — court-view detection by geometry, not color mood.

Take 1 (m3_court_view.py) classified on blue/green fractions and broke on
daytime frames: the shaded stadium seats merge with the court into one
oversized blue blob, and the sunlit apron washes out below the green
threshold. Evening-tuned color rules break at noon.

Take 2 uses what M1 already knows: the broadcast framing is FIXED (the M1
court quad lands pixel-perfect on reel frames from both ends of the match
— framing_check.png). So probe the geometry:

  interior probes — a grid of court-model points projected into the image
                    must read court-blue (players/net cover a few; allow
                    slack)
  apron probes    — points just OUTSIDE the doubles lines and behind the
                    near baseline must NOT read blue (green apron day or
                    night; blue there means a close-up filled the frame)

court view  <=>  most interior probes blue AND most apron probes not.

Then segment: median-smooth the boolean, keep runs >= MIN_RUN_S, merging
gaps <= GAP_S.

Usage:
    uv run experiments/m3_court_probe.py clips/match_r2_highlights.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"
ROOT = Path(__file__).resolve().parent.parent

BLUE_LO, BLUE_HI = (95, 50, 40), (135, 255, 255)
W_C, L_C = 10.97, 23.77
DS = 2  # probe at half res

COURT_HIT_MIN = 0.80
APRON_HIT_MAX = 0.35
MIN_RUN_S = 3.0
GAP_S = 0.6
SMOOTH = 9


def probe_points():
    interior = [(x, y)
                for x in np.linspace(1.5, W_C - 1.5, 5)
                for y in np.linspace(2.0, L_C - 2.0, 10)]
    apron = ([(-1.6, y) for y in np.linspace(3, L_C - 3, 6)] +
             [(W_C + 1.6, y) for y in np.linspace(3, L_C - 3, 6)] +
             [(x, L_C + 2.5) for x in np.linspace(1, W_C - 1, 5)])
    return np.float32(interior), np.float32(apron)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    Hc2i = np.load(ROOT / "outputs/m1/H_court_to_img.npy")
    interior, apron = probe_points()

    def project(pts):
        img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hc2i).reshape(-1, 2)
        return (img / DS).astype(int)

    p_int = project(interior)
    p_apr = project(apron)

    cap = cv2.VideoCapture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // DS
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // DS
    ok_int = (p_int[:, 0] >= 0) & (p_int[:, 0] < w) & (p_int[:, 1] >= 0) & (p_int[:, 1] < h)
    ok_apr = (p_apr[:, 0] >= 0) & (p_apr[:, 0] < w) & (p_apr[:, 1] >= 0) & (p_apr[:, 1] < h)
    p_int, p_apr = p_int[ok_int], p_apr[ok_apr]
    print(f"{n_total} frames @ {fps:.2f} fps; probes: {len(p_int)} interior, {len(p_apr)} apron")

    court_hits, apron_hits = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        blue = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), BLUE_LO, BLUE_HI)
        court_hits.append(float(np.mean(blue[p_int[:, 1], p_int[:, 0]] > 0)))
        apron_hits.append(float(np.mean(blue[p_apr[:, 1], p_apr[:, 0]] > 0)))
        i += 1
        if i % 5000 == 0:
            print(f"  {i}/{n_total}")

    court_hits = np.array(court_hits)
    apron_hits = np.array(apron_hits)
    raw = (court_hits > COURT_HIT_MIN) & (apron_hits < APRON_HIT_MAX)

    # median smooth
    k = SMOOTH
    pad = np.pad(raw.astype(np.uint8), k // 2, mode="edge")
    smooth = np.array([np.median(pad[j:j + k]) for j in range(len(raw))]).astype(bool)

    # runs -> segments with gap merging
    segs = []
    start = None
    for j, v in enumerate(smooth):
        if v and start is None:
            start = j
        elif not v and start is not None:
            segs.append([start, j - 1])
            start = None
    if start is not None:
        segs.append([start, len(smooth) - 1])
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) / fps <= GAP_S:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    keep = [s for s in merged if (s[1] - s[0] + 1) / fps >= MIN_RUN_S]

    with open(OUT_DIR / "view_probe.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "court_hit", "apron_hit", "court_view"])
        for j in range(len(raw)):
            wr.writerow([j, round(court_hits[j], 3), round(apron_hits[j], 3),
                         int(smooth[j])])
    with open(OUT_DIR / "segments.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seg", "start_frame", "end_frame", "start_s", "dur_s"])
        for k2, (a, b) in enumerate(keep, 1):
            wr.writerow([k2, a, b, round(a / fps, 2), round((b - a + 1) / fps, 2)])

    total_s = sum((b - a + 1) for a, b in keep) / fps
    print(f"court-view frames (smoothed): {int(smooth.sum())} ({smooth.mean():.1%})")
    print(f"segments >= {MIN_RUN_S}s: {len(keep)}, total {total_s:.0f}s")
    for k2, (a, b) in enumerate(keep, 1):
        print(f"  seg {k2:>2}: f{a}-f{b}  {a/fps:7.1f}s  dur {(b-a+1)/fps:5.1f}s")
    print(f"-> {OUT_DIR / 'view_probe.csv'}, {OUT_DIR / 'segments.csv'}")


if __name__ == "__main__":
    main()
