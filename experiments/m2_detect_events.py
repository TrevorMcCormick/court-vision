"""M2 run 2: detect trajectory events and classify hit vs bounce.

Candidates: sign flips in the (lightly smoothed) image-y velocity — the
ball stops falling or stops rising. Each candidate is then classified by
what the ball's along-court direction of travel does across the event,
using windowed medians so the airborne projection spikes and small track
gaps don't lie to us:

    direction flips  -> HIT   (somebody sent it back)
    direction holds  -> BOUNCE (the court doesn't return serve)

Only downward-to-upward flips (ball was descending, then ascends in image
terms) are considered — apexes (up->down) are just gravity doing its job.

Usage:
    uv run experiments/m2_detect_events.py outputs/m2/kinematics.npz
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m2"

SMOOTH = 3          # frames, moving average on image y
WIN = 6             # frames, median window each side for classification
MIN_GAP = 8         # frames, merge events closer than this
MIN_SWING = 6.0     # px/frame, |velocity change| across the event


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
    vcy = np.gradient(cy, frames.astype(float)) * 30.0  # m/s

    # candidates: descending (viy > 0, image y growing = falling) -> ascending
    cands = []
    for i in range(2, len(frames) - 2):
        if viy[i - 1] > 0 and viy[i + 1] < 0:
            swing = viy[i - 1] - viy[i + 1]
            if swing >= MIN_SWING:
                cands.append((i, swing))

    # merge close candidates, keep the biggest swing
    merged = []
    for i, swing in sorted(cands, key=lambda c: -c[1]):
        if all(abs(int(frames[i]) - int(frames[j])) >= MIN_GAP for j, _ in merged):
            merged.append((i, swing))
    merged.sort()

    events = []
    for i, swing in merged:
        before = vcy[max(0, i - WIN):i]
        after = vcy[i + 1:i + 1 + WIN]
        mb, ma = np.median(before), np.median(after)
        flipped = np.sign(mb) != np.sign(ma) and abs(mb) > 1 and abs(ma) > 1
        kind = "hit" if flipped else "bounce"
        events.append({"frame": int(frames[i]), "kind": kind,
                       "swing_px_per_frame": round(float(swing), 1),
                       "vcy_before_ms": round(float(mb), 1),
                       "vcy_after_ms": round(float(ma), 1),
                       "court_x_m": None, "court_y_m": round(float(cy[i]), 2)})
        print(f"frame {frames[i]:3d}  {kind.upper():6s}  swing {swing:5.1f}  "
              f"vcy {mb:6.1f} -> {ma:6.1f} m/s   court-y {cy[i]:5.1f} m")

    with open(OUT_DIR / "events.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        wr.writeheader()
        wr.writerows(events)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frames, iy, ".-", ms=3, lw=0.7, color="tab:blue", zorder=1)
    for e in events:
        i = np.searchsorted(frames, e["frame"])
        color = "red" if e["kind"] == "hit" else "white"
        ax.scatter(e["frame"], iy[i], s=120, zorder=2, color=color,
                   edgecolors="black", linewidths=1.5)
        ax.annotate(e["kind"], (e["frame"], iy[i]), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_facecolor("#3b5b92")
    ax.set_xlabel("frame (30 fps)")
    ax.set_ylabel("image y (px)")
    ax.set_title(f"M2 — detected events: "
                 f"{sum(e['kind'] == 'hit' for e in events)} hits, "
                 f"{sum(e['kind'] == 'bounce' for e in events)} bounces")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "events_on_trajectory.png", dpi=150)
    print(f"-> outputs/m2/events.csv, events_on_trajectory.png")


if __name__ == "__main__":
    main()
