"""M2 run 5: detector v3 = v2 + a second pass for far-side bounces.

Far bounces have no image-y cusp (rising and receding push image-y the
same way), so they get their own signal: shadow-speed collapse. Airborne,
the ground shadow races at projection-amplified speed; grounded, it moves
at the true ball speed. A sharp collapse of |court-y velocity| without a
direction flip, in the far half of the court, is a bounce.

Honesty note: thresholds here are tuned against the ground truth of THIS
one rally. The claim M2 makes is that the signals exist and survive frame
verification — not that these numbers generalize. That's later-milestone
work.

Usage:
    uv run experiments/m2_detect_events_v3.py outputs/m2/kinematics.npz
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
SWING_NEAR = 6.0
SWING_FAR = 1.2
FAR_Y_PX = 250
HIT_SPEED = 4.5      # m/s outgoing shadow speed separating hit from bounce
COLLAPSE = 3.0       # |vcy| before/after ratio marking a far bounce
FAR_HALF_M = 11.885  # net line; far bounces have court-y below this


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

    # pass 1: image-y cusps (v2 logic)
    cands = []
    for i in range(2, len(frames) - 2):
        if viy[i - 1] > 0 and viy[i + 1] < 0:
            swing = viy[i - 1] - viy[i + 1]
            need = SWING_NEAR if iy_s[i] >= FAR_Y_PX else SWING_FAR
            if swing >= need:
                cands.append((i, swing, "cusp"))

    # pass 2: far-side shadow-speed collapse (no direction flip)
    for i in range(WIN, len(frames) - WIN):
        if cy[i] > FAR_HALF_M:
            continue
        mb = np.median(vcy[i - WIN:i])
        ma = np.median(vcy[i + 1:i + 1 + WIN])
        if np.sign(mb) == np.sign(ma) and abs(ma) > 0.3 and abs(mb) / abs(ma) >= COLLAPSE:
            cands.append((i, abs(mb) / abs(ma), "collapse"))

    # merge, preferring cusp candidates (they localize better)
    merged = []
    for i, score, how in sorted(cands, key=lambda c: (c[2] != "cusp", -c[1])):
        if all(abs(int(frames[i]) - int(frames[j])) >= MIN_GAP for j, _, _ in merged):
            merged.append((i, score, how))
    merged.sort()

    events = []
    for i, score, how in merged:
        mb = np.median(vcy[max(0, i - WIN):i])
        ma = np.median(vcy[i + 1:i + 1 + WIN])
        kind = "bounce" if how == "collapse" else ("hit" if abs(ma) > HIT_SPEED else "bounce")
        events.append({"frame": int(frames[i]), "kind": kind, "signal": how,
                       "vcy_before_ms": round(float(mb), 1),
                       "vcy_after_ms": round(float(ma), 1),
                       "court_y_m": round(float(cy[i]), 2)})
        print(f"frame {frames[i]:3d}  {kind.upper():6s}  via {how:8s}  "
              f"vcy {mb:6.1f} -> {ma:6.1f} m/s   court-y {cy[i]:5.1f} m")

    with open(OUT_DIR / "events_v3.csv", "w", newline="") as f:
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
    print(f"\nvs ground truth: {len(matched)}/{len(truth)} matched, "
          f"{correct}/{len(matched)} correct, {extra} extra")

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
    ax.set_title(f"M2 v3 — {sum(e['kind'] == 'hit' for e in events)} hits, "
                 f"{sum(e['kind'] == 'bounce' for e in events)} bounces "
                 f"(white = bounce, red = hit)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "events_v3.png", dpi=150)
    print("-> outputs/m2/events_v3.csv, events_v3.png")


if __name__ == "__main__":
    main()
