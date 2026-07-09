"""M0 iteration 4 — reparse the box-prompt response with the right box format.

Discovery: for box-prompted runs the response `boxes` are [cx, cy, w, h]
(center + size, normalized), NOT [x1, y1, x2, y2]. Frame 240's box decodes to
exactly where the game ball was prompted, ball-sized (~24x21 px). Earlier
parsing produced negative widths, which is what gave it away.

Reparses rle_ballbox.json, drops lost-track frames (box area way above
ball size), writes trajectory_ballfix.csv + plot.
"""

import csv
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m0"

MAX_BALL_W = 0.05  # normalized; game ball is ~0.01-0.02 wide at 720p

d = json.loads((OUT_DIR / "rle_ballbox.json").read_text())
rows, dropped = [], 0
for i, box in enumerate(d["boxes"]):
    if not box or len(box) != 4:
        dropped += 1
        continue
    cx, cy, w, h = box
    if w > MAX_BALL_W:  # lost track -> giant stuck box
        dropped += 1
        continue
    rows.append({"frame": i, "cx": cx, "cy": cy, "w": w, "h": h, "score": None})

csv_path = OUT_DIR / "trajectory_ballfix.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["frame", "cx", "cy", "w", "h", "score"])
    w.writeheader()
    w.writerows(rows)
print(f"kept {len(rows)} ball-sized boxes, dropped {dropped} (empty or lost-track)")
print(f"saved {csv_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

frames = [r["frame"] for r in rows]
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(frames, [r["cx"] for r in rows], ".", markersize=4)
axes[0].set_ylabel("ball x (normalized)")
axes[1].plot(frames, [r["cy"] for r in rows], ".", markersize=4, color="tab:orange")
axes[1].set_ylabel("ball y (normalized)")
axes[1].invert_yaxis()
axes[1].set_xlabel("frame (30 fps)")
fig.suptitle("M0 — game ball centroid, SAM 3 box prompt @ frame 240 (propagated both directions)")
fig.tight_layout()
fig.savefig(OUT_DIR / "trajectory_ballfix.png", dpi=150)
print(f"saved {OUT_DIR / 'trajectory_ballfix.png'}")
