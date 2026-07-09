"""M1 run 4: map the M0 ball track into real court coordinates.

Applies the img->court homography to the M0 trajectory and draws it
top-down on a to-scale court diagram. Honest caveat, stated up front: a
homography maps the GROUND PLANE, and the ball is airborne most of the
time — so what this plots is the ball's ground shadow (vertical
projection), not its 3-D position. The shadow is exactly right at bounces
and near-right at low contact points; that's what M2 (bounce detection)
will exploit.

Usage:
    uv run experiments/m1_map_track.py outputs/m0/trajectory_ballfix.csv outputs/m1/H_img_to_court.npy
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m1"

W_COURT = 10.97
L_COURT = 23.77
SINGLES_INSET = 1.372
NET_Y = L_COURT / 2
SVC_FAR_Y = NET_Y - 6.40
SVC_NEAR_Y = NET_Y + 6.40
CENTER_X = W_COURT / 2

IMG_W, IMG_H = 1280, 720


def draw_court(ax):
    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="white", lw=1.5, zorder=1)

    ax.add_patch(plt.Rectangle((-2, -3), W_COURT + 4, L_COURT + 6,
                               color="#3b5b92", zorder=0))
    line(0, 0, W_COURT, 0)
    line(0, L_COURT, W_COURT, L_COURT)
    line(0, 0, 0, L_COURT)
    line(W_COURT, 0, W_COURT, L_COURT)
    line(SINGLES_INSET, 0, SINGLES_INSET, L_COURT)
    line(W_COURT - SINGLES_INSET, 0, W_COURT - SINGLES_INSET, L_COURT)
    line(SINGLES_INSET, SVC_FAR_Y, W_COURT - SINGLES_INSET, SVC_FAR_Y)
    line(SINGLES_INSET, SVC_NEAR_Y, W_COURT - SINGLES_INSET, SVC_NEAR_Y)
    line(CENTER_X, SVC_FAR_Y, CENTER_X, SVC_NEAR_Y)
    ax.plot([0, W_COURT], [NET_Y, NET_Y], color="#dddddd", lw=2.5,
            linestyle=(0, (4, 2)), zorder=1)  # net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_csv")
    parser.add_argument("h_npy")
    args = parser.parse_args()

    H = np.load(args.h_npy)  # img px -> court meters

    frames, px = [], []
    with open(args.traj_csv) as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            px.append((float(row["cx"]) * IMG_W, float(row["cy"]) * IMG_H))
    frames = np.array(frames)
    px = np.float32(px)

    court = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)

    # write the mapped track
    out_csv = OUT_DIR / "track_court.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "court_x_m", "court_y_m"])
        for fr, (x, y) in zip(frames, court):
            wr.writerow([fr, f"{x:.3f}", f"{y:.3f}"])

    # how far outside the court does the shadow stray?
    margin = 2.0
    inside = ((court[:, 0] > -margin) & (court[:, 0] < W_COURT + margin)
              & (court[:, 1] > -margin) & (court[:, 1] < L_COURT + margin))
    print(f"{len(court)} points; {inside.sum()} within {margin} m of the court")
    print(f"court-x range: {court[:, 0].min():.1f} .. {court[:, 0].max():.1f} m")
    print(f"court-y range: {court[:, 1].min():.1f} .. {court[:, 1].max():.1f} m")

    fig, ax = plt.subplots(figsize=(6, 10))
    draw_court(ax)
    sc = ax.scatter(court[:, 0], court[:, 1], c=frames, cmap="plasma",
                    s=14, zorder=2)
    fig.colorbar(sc, ax=ax, label="frame (30 fps)", shrink=0.7)
    ax.set_xlim(-2, W_COURT + 2)
    ax.set_ylim(L_COURT + 3, -3)  # image-like: far baseline at top
    ax.set_aspect("equal")
    ax.set_title("M1 — M0 ball track mapped to court coordinates\n"
                 "(ground shadow: exact at bounces, projected while airborne)")
    ax.set_xlabel("court x (m)")
    ax.set_ylabel("court y (m)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "track_on_court.png", dpi=150)
    print("-> outputs/m1/track_on_court.png, track_court.csv")


if __name__ == "__main__":
    main()
