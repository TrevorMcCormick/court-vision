"""M2 run 4: detector v2, corrected by the frame-verified ground truth.

What v1 got wrong (verified by eyeballing frames in run 3):
  - Near-side bounces (f44, f139) classified as hits: the sign-flip test
    was fooled by post-bounce shadow noise of ~-2..-4 m/s. Fix: classify
    by OUTGOING speed. A real return leaves at >5 m/s of shadow speed; a
    bounce's outgoing shadow speed collapses to ~0.
  - Far-side bounces (~f85, ~f179) missed entirely: perspective compresses
    the bounce cusp to a couple of pixels, far below the near-side swing
    threshold. Fix: scale the swing threshold by where the ball is in the
    frame (far half: much smaller cusps count).

Ground truth (run 3): hits f2, f57, f99, f153, f185, f243;
bounces f44, ~f85, f139, ~f179, f232.

Usage:
    uv run experiments/m2_detect_events_v2.py outputs/m2/kinematics.npz
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m2"

SMOOTH = 3
WIN = 6
MIN_GAP = 8
SWING_NEAR = 6.0     # px/frame, ball low in frame (near side)
SWING_FAR = 1.2      # px/frame, ball high in frame (far side, compressed)
FAR_Y_PX = 250       # image y above this = "near", below = "far"
HIT_SPEED = 4.5      # m/s outgoing shadow speed that separates hit/bounce


def moving_average(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kin_npz")
    args = parser.parse_args()

    kin = np.load(args.kin_npz)
    frames, iy, cy = kin["frames"], kin["iy"], kin["cy"]

    iy_s = moving_average(iy, SMOOTH)
    viy = np.gradient(iy_s, frames.astype(float))
    vcy = np.gradient(cy, frames.astype(float)) * 30.0

    cands = []
    for i in range(2, len(frames) - 2):
        if viy[i - 1] > 0 and viy[i + 1] < 0:
            swing = viy[i - 1] - viy[i + 1]
            need = SWING_NEAR if iy_s[i] >= FAR_Y_PX else SWING_FAR
            if swing >= need:
                cands.append((i, swing))

    merged = []
    for i, swing in sorted(cands, key=lambda c: -c[1]):
        if all(abs(int(frames[i]) - int(frames[j])) >= MIN_GAP for j, _ in merged):
            merged.append((i, swing))
    merged.sort()

    events = []
    for i, swing in merged:
        after = vcy[i + 1:i + 1 + WIN]
        ma = np.median(after)
        mb = np.median(vcy[max(0, i - WIN):i])
        kind = "hit" if abs(ma) > HIT_SPEED else "bounce"
        events.append({"frame": int(frames[i]), "kind": kind,
                       "swing_px_per_frame": round(float(swing), 1),
                       "vcy_before_ms": round(float(mb), 1),
                       "vcy_after_ms": round(float(ma), 1),
                       "court_y_m": round(float(cy[i]), 2)})
        print(f"frame {frames[i]:3d}  {kind.upper():6s}  swing {swing:5.1f}  "
              f"vcy {mb:6.1f} -> {ma:6.1f} m/s   court-y {cy[i]:5.1f} m")

    with open(OUT_DIR / "events_v2.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        wr.writeheader()
        wr.writerows(events)

    truth = {2: "hit", 44: "bounce", 57: "hit", 85: "bounce", 99: "hit",
             139: "bounce", 153: "hit", 179: "bounce", 185: "hit",
             232: "bounce", 243: "hit"}
    matched = {}
    for e in events:
        for tf, tk in truth.items():
            if abs(e["frame"] - tf) <= 5 and tf not in matched:
                matched[tf] = (e["frame"], e["kind"], tk)
                break
    correct = sum(1 for _, (f, k, tk) in matched.items() if k == tk)
    extra = len(events) - len(matched)
    print(f"\nvs ground truth: {len(matched)}/{len(truth)} events matched, "
          f"{correct}/{len(matched)} correctly classified, {extra} extra detections")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frames, iy, ".-", ms=3, lw=0.7, color="#cfe3ff", zorder=1)
    for e in events:
        i = np.searchsorted(frames, e["frame"])
        color = "#ff5555" if e["kind"] == "hit" else "#ffffff"
        ax.scatter(e["frame"], iy[i], s=130, zorder=2, color=color,
                   edgecolors="black", linewidths=1.5)
        ax.annotate(e["kind"], (e["frame"], iy[i]), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=8, color="white")
    ax.invert_yaxis()
    ax.set_facecolor("#3b5b92")
    ax.set_xlabel("frame (30 fps)")
    ax.set_ylabel("image y (px)")
    ax.set_title(f"M2 v2 — {sum(e['kind'] == 'hit' for e in events)} hits, "
                 f"{sum(e['kind'] == 'bounce' for e in events)} bounces")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "events_v2.png", dpi=150)
    print("-> outputs/m2/events_v2.csv, events_v2.png")


if __name__ == "__main__":
    main()
