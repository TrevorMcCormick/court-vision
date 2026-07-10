"""T2 — extract candidate point clips: m3_extract_points with t2 paths.

From segments.csv (geometry-probe court-view runs):
  clips/points/point_NN.mp4  — one re-encoded (frame-accurate) clip per
                               segment; next sessions chart these
  outputs/m3/point_timeline.png — the 24-minute reel as a strip: green
                               court-view segments, the zoomed early block
                               hatched, cutaways gray
  outputs/m3/points_montage.mp4 — first 2 s of every segment back to back
                               (the reel's point-starts, wall to wall)

Usage:
    uv run experiments/t2_extract_points.py clips/t2_federer_haase_30fps.mp4
"""

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "t2"
ROOT = Path(__file__).resolve().parent.parent
POINTS_DIR = ROOT / "clips" / "points_t2"

ZOOM_BLOCK = (0.0, 0.0)  # t1: no zoomed block identified yet   # early set-1 tighter framing; probes reject it
MONTAGE_S = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--skip-clips", action="store_true")
    args = parser.parse_args()

    segs = list(csv.DictReader(open(OUT_DIR / "segments.csv")))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- per-point clips ----
    if not args.skip_clips:
        POINTS_DIR.mkdir(parents=True, exist_ok=True)
        for s in segs:
            k = int(s["seg"])
            start = int(s["start_frame"]) / fps
            dur = float(s["dur_s"])
            out = POINTS_DIR / f"t2_point_{k:02d}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
                 "-i", args.video, "-t", f"{dur:.3f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-an", str(out)], check=True)
        print(f"-> {len(segs)} clips in {POINTS_DIR}")

    # ---- timeline strip ----
    fig, ax = plt.subplots(figsize=(14, 2.6))
    total_s = n_total / fps
    ax.axhspan(0, 1, color="#d9d9d9")
    ax.axvspan(*ZOOM_BLOCK, color="#f5a623", alpha=0.45, hatch="///",
               label="court view, zoomed framing (missed)")
    for s in segs:
        a = float(s["start_s"])
        ax.axvspan(a, a + float(s["dur_s"]), color="#2e9e4f")
    ax.axvspan(-10, -9, color="#2e9e4f", label="court view (matched framing)")
    ax.axvspan(-10, -9, color="#d9d9d9", label="cutaways / replays / close-ups")
    ax.set_xlim(0, total_s)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("reel time (s)")
    ax.set_title(f"24-minute highlights reel -> {len(segs)} court-view segments "
                 f"({sum(float(s['dur_s']) for s in segs):.0f}s of chartable play)")
    ax.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "point_timeline.png", dpi=150)
    print(f"-> {OUT_DIR / 'point_timeline.png'}")

    # ---- montage of point starts ----
    n_show = int(MONTAGE_S * fps)
    writer = cv2.VideoWriter(str(OUT_DIR / "points_montage.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + 70))
    for s in segs:
        k = int(s["seg"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s["start_frame"]))
        for j in range(n_show):
            ok, frame = cap.read()
            if not ok:
                break
            bar = np.full((70, W, 3), (24, 24, 24), np.uint8)
            cv2.putText(bar, f"point candidate {k}/{len(segs)}   "
                             f"reel @{float(s['start_s']):.0f}s   "
                             f"dur {s['dur_s']}s",
                        (18, 46), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (255, 255, 255), 2)
            canvas = np.vstack([frame, bar])
            writer.write(canvas)
    writer.release()
    print(f"-> {OUT_DIR / 'points_montage.mp4'} "
          f"({len(segs)} x {MONTAGE_S:.0f}s)")


if __name__ == "__main__":
    main()
