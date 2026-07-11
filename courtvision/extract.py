"""Extract per-point clips + clip camera offsets — t3/t4 recipe, config-driven.

Prefers the score-bug point boundaries (segments_v2.csv from
courtvision.boundaries) over the probe's raw segments — segment != point
(merges, splits, no-bug drops; see LOG). Each clip also gets a camera
offset vs the fit camera (median of the probe's winning shifts over the
segment): downstream stages stabilize to clip frame 0 and must subtract
this to land in fit-camera coordinates.

Also writes the transcription receipts the alignment pass needs: a
contact sheet per clip (bug crop at several timestamps) so the score
bug can be read BY EYE into data/mcp/<id>_clip_alignment.csv.

Usage:
    uv run python -m courtvision extract t5 [--skip-clips]
"""

import csv
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def extract_match(cfg, skip_clips=False):
    out_dir = cfg.out_dir
    points_dir = cfg.clips_dir

    seg_csv = out_dir / "segments_v2.csv"
    if not seg_csv.exists():
        seg_csv = out_dir / "segments.csv"
    segs = list(csv.DictReader(open(seg_csv)))
    print(f"segments from {seg_csv.name}")
    cap = cv2.VideoCapture(str(cfg.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- per-clip camera offset vs the fit camera ----
    probe = list(csv.DictReader(open(out_dir / "view_probe.csv")))
    with open(out_dir / "clip_offsets.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "dx", "dy"])
        for s in segs:
            k = int(s["seg"])
            a, b = int(s["start_frame"]), int(s["end_frame"])
            dxs = [int(r["shift_dx"]) for r in probe[a:b + 1]]
            dys = [int(r["shift_dy"]) for r in probe[a:b + 1]]
            wr.writerow([f"{cfg.id}_point_{k:02d}",
                         int(np.median(dxs)), int(np.median(dys))])
    print(f"-> {out_dir / 'clip_offsets.csv'}")

    # ---- per-point clips ----
    if not skip_clips:
        points_dir.mkdir(parents=True, exist_ok=True)
        for s in segs:
            k = int(s["seg"])
            start = int(s["start_frame"]) / fps
            dur = float(s["dur_s"])
            out = points_dir / f"{cfg.id}_point_{k:02d}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
                 "-i", str(cfg.video), "-t", f"{dur:.3f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-an", str(out)], check=True)
        print(f"-> {len(segs)} clips in {points_dir}")

    # ---- timeline strip ----
    fig, ax = plt.subplots(figsize=(14, 2.6))
    total_s = n_total / fps
    ax.axhspan(0, 1, color="#d9d9d9")
    for s in segs:
        a = float(s["start_s"])
        ax.axvspan(a, a + float(s["dur_s"]), color="#2e9e4f")
    ax.set_xlim(0, total_s)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("reel time (s)")
    ax.set_title(f"{total_s/60:.0f}-minute reel -> {len(segs)} point clips "
                 f"({sum(float(s['dur_s']) for s in segs):.0f}s of chartable play)")
    fig.tight_layout()
    fig.savefig(out_dir / "point_timeline.png", dpi=150)
    print(f"-> {out_dir / 'point_timeline.png'}")


def bug_sheets(cfg, crop, per_row=4):
    """Transcription receipts: for each extracted clip, tile the bug
    region crop at several timestamps (start/25%/50%/75%/end) into
    outputs/<id>/bug_sheets/<clip>.png. crop = (y0, y1, x0, x1) full-res,
    generous enough to hold every era of a growing bug."""
    sheets_dir = cfg.out_dir / "bug_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1 = crop
    for path in sorted(cfg.clips_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles = []
        for t in (0.02, 0.25, 0.5, 0.75, 0.97):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * (n - 1)))
            ok, fr = cap.read()
            if not ok:
                continue
            c = cv2.resize(fr[y0:y1, x0:x1], None, fx=3, fy=3,
                           interpolation=cv2.INTER_NEAREST)
            bar = np.full((22, c.shape[1], 3), 40, np.uint8)
            cv2.putText(bar, f"{path.stem} @{t:.0%} f{int(t * (n - 1))}",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            tiles.append(np.vstack([bar, c]))
        cap.release()
        if tiles:
            cv2.imwrite(str(sheets_dir / f"{path.stem}.png"), np.vstack(tiles))
    print(f"-> {sheets_dir}")
