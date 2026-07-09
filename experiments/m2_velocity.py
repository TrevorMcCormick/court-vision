"""M2 run 1: look at the velocity structure of the M0 ball track.

Before detecting anything, see what the data actually looks like: image-y
velocity (height-ish), court-y velocity (along-court travel from the M1
shadow track), and where the track has gaps. Hits should show as sign
flips in court-y velocity; bounces as cusps in image-y velocity without a
travel-direction flip.

Usage:
    uv run experiments/m2_velocity.py outputs/m0/trajectory_ballfix.csv outputs/m1/track_court.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m2"

IMG_H = 720


def load_csv(path, cols):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[int(row["frame"])] = [float(row[c]) for c in cols]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_csv")
    parser.add_argument("court_csv")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = load_csv(args.traj_csv, ["cx", "cy"])
    court = load_csv(args.court_csv, ["court_x_m", "court_y_m"])
    frames = np.array(sorted(set(img) & set(court)))

    gaps = np.where(np.diff(frames) > 1)[0]
    print(f"{len(frames)} frames, {len(gaps)} gaps: "
          f"{[(int(frames[g]), int(frames[g+1])) for g in gaps]}")

    iy = np.array([img[f][1] * IMG_H for f in frames])      # image y, px
    cy = np.array([court[f][1] for f in frames])            # court y, m

    # central differences over actual frame spacing (30 fps)
    dt = np.gradient(frames) / 30.0
    viy = np.gradient(iy) / dt / 30.0   # px/frame-ish scale for readability
    vcy = np.gradient(cy) / (np.gradient(frames) / 30.0)    # m/s along court

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(frames, iy, ".-", ms=3, lw=0.7)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("image y (px)\n(up = higher ball)")
    axes[1].plot(frames, viy, ".-", ms=3, lw=0.7, color="tab:orange")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("d(image y)/dt")
    axes[2].plot(frames, vcy, ".-", ms=3, lw=0.7, color="tab:green")
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("court-y velocity (m/s)\n(sign = direction of travel)")
    axes[2].set_xlabel("frame (30 fps)")
    axes[0].set_title("M2 — velocity structure of the M0 track")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "velocity_structure.png", dpi=150)
    print("-> outputs/m2/velocity_structure.png")

    np.savez(OUT_DIR / "kinematics.npz", frames=frames, iy=iy, cy=cy,
             viy=viy, vcy=vcy)


if __name__ == "__main__":
    main()
