"""M2 run 6: detector v4 — final for this milestone.

v3's false positive (f71, ball crossing the net) taught the real physics:
the shadow-speed collapse is not an instant at ground contact, it unwinds
GRADUALLY through the descent as the ball loses height. So early collapse
candidates are mid-descent artifacts; the bounce is the LAST collapse
candidate in each descent. v4 keeps, per segment between cusp events, only
the final collapse candidate.

Position estimates for collapse bounces also stabilize here: instead of
the instantaneous shadow at the detected frame (meters of error per frame
of timing slop at the far baseline), use the median shadow over the
post-bounce window, when the ball is low and slow.

Frame-verified ground truth for the full rally (runs 3+5):
  hits    f2, f57, f99, f153, f185, f243, ~f280
  bounces f44, ~f85, f139, ~f175, f232, ~f257

Usage:
    uv run experiments/m2_detect_events_v4.py outputs/m2/kinematics.npz
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
HIT_SPEED = 4.5
COLLAPSE = 3.0
FAR_HALF_M = 11.885

W_COURT = 10.97
L_COURT = 23.77
SINGLES_INSET = 1.372
NET_Y = L_COURT / 2
SVC_FAR_Y = NET_Y - 6.40
SVC_NEAR_Y = NET_Y + 6.40
CENTER_X = W_COURT / 2


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
    cx = kin["cy"] * 0  # placeholder if court-x absent in npz
    if "cx_court" in kin:
        cx = kin["cx_court"]

    iy_s = moving_average(iy, SMOOTH)
    viy = np.gradient(iy_s, frames.astype(float))
    vcy = np.gradient(cy, frames.astype(float)) * 30.0

    # pass 1: image-y cusps
    cusps = []
    for i in range(2, len(frames) - 2):
        if viy[i - 1] > 0 and viy[i + 1] < 0:
            swing = viy[i - 1] - viy[i + 1]
            need = SWING_NEAR if iy_s[i] >= FAR_Y_PX else SWING_FAR
            if swing >= need:
                cusps.append(i)
    merged_cusps = []
    for i in sorted(cusps, key=lambda i: -(viy[i - 1] - viy[i + 1])):
        if all(abs(int(frames[i]) - int(frames[j])) >= MIN_GAP for j in merged_cusps):
            merged_cusps.append(i)
    merged_cusps.sort()

    # pass 2: collapse candidates, keep only the LAST per inter-cusp segment
    collapse = []
    for i in range(WIN, len(frames) - WIN):
        if cy[i] > FAR_HALF_M:
            continue
        mb = np.median(vcy[i - WIN:i])
        ma = np.median(vcy[i + 1:i + 1 + WIN])
        if np.sign(mb) == np.sign(ma) and abs(ma) > 0.3 and abs(mb) / abs(ma) >= COLLAPSE:
            collapse.append(i)
    bounds = [-1] + [int(frames[i]) for i in merged_cusps] + [10 ** 9]
    last_per_segment = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = [i for i in collapse if a + MIN_GAP <= frames[i] <= b - MIN_GAP]
        if seg:
            last_per_segment.append(seg[-1])

    events = []
    for i in sorted(merged_cusps + last_per_segment):
        how = "cusp" if i in merged_cusps else "collapse"
        mb = np.median(vcy[max(0, i - WIN):i])
        ma = np.median(vcy[i + 1:i + 1 + WIN])
        kind = "bounce" if how == "collapse" else ("hit" if abs(ma) > HIT_SPEED else "bounce")
        # position: post-window median for collapse bounces, instantaneous otherwise
        if how == "collapse":
            pos_y = float(np.median(cy[i + 1:i + 1 + WIN]))
        else:
            pos_y = float(cy[i])
        events.append({"frame": int(frames[i]), "kind": kind, "signal": how,
                       "vcy_before_ms": round(float(mb), 1),
                       "vcy_after_ms": round(float(ma), 1),
                       "court_y_m": round(pos_y, 2)})
        print(f"frame {frames[i]:3d}  {kind.upper():6s}  via {how:8s}  "
              f"vcy {mb:6.1f} -> {ma:6.1f}   court-y {pos_y:5.1f} m")

    with open(OUT_DIR / "events_v4.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        wr.writeheader()
        wr.writerows(events)

    truth = {2: "hit", 44: "bounce", 57: "hit", 85: "bounce", 99: "hit",
             139: "bounce", 153: "hit", 175: "bounce", 185: "hit",
             232: "bounce", 243: "hit", 257: "bounce", 280: "hit"}
    matched = {}
    for e in events:
        best = None
        for tf, tk in truth.items():
            if tf not in matched and abs(e["frame"] - tf) <= 8:
                if best is None or abs(e["frame"] - tf) < abs(e["frame"] - best):
                    best = tf
        if best is not None:
            matched[best] = (e["frame"], e["kind"], truth[best])
    correct = sum(1 for _, (f, k, tk) in matched.items() if k == tk)
    extra = len(events) - len(matched)
    print(f"\nvs ground truth (±8 frames): {len(matched)}/{len(truth)} matched, "
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
    ax.set_title(f"M2 v4 — {sum(e['kind'] == 'hit' for e in events)} hits, "
                 f"{sum(e['kind'] == 'bounce' for e in events)} bounces "
                 f"(red = hit, white = bounce)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "events_v4.png", dpi=150)
    print("-> outputs/m2/events_v4.csv, events_v4.png")


if __name__ == "__main__":
    main()
