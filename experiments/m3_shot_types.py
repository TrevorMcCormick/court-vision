"""M3 experiment 2 — shot types: striker + contact side -> f/b letters.

The proto-chart's biggest '?' column. With per-player tracks we can answer,
per M2 hit event:

  striker      = player whose box is nearest the ball at the hit frame
                 (validates the old y>NET_Y heuristic AND the emergent
                 far/near alternation — three independent answers agree
                 or something is wrong)
  contact side = ball x vs striker's box center x, mirrored by end:
                 near player has his back to us (image-right = his right),
                 far player faces us (image-LEFT = his right)
  f/b          = right-side contact -> forehand for a right-hander.
                 Both Zverev and Gasquet are right-handed. Handedness is
                 ASSUMED, not detected — on the record.

Also maps striker feet through the M1 homography at each hit — feet are on
the ground plane, so unlike the airborne ball these positions are exact.

Usage:
    uv run experiments/m3_shot_types.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"
ROOT = Path(__file__).resolve().parent.parent
W, H = 1280, 720

W_COURT = 10.97
L_COURT = 23.77
NET_Y = L_COURT / 2


def load_ball():
    boxes = {}
    with open(ROOT / "outputs/m0/trajectory_ballfix.csv") as f:
        for row in csv.DictReader(f):
            boxes[int(row["frame"])] = (float(row["cx"]) * W, float(row["cy"]) * H)
    return boxes


def load_players():
    traj = {}
    with open(OUT_DIR / "players_traj.csv") as f:
        for row in csv.DictReader(f):
            traj.setdefault(int(row["frame"]), {})[row["player"]] = {
                "cx": float(row["cx"]) * W, "cy": float(row["cy"]) * H,
                "w": float(row["w"]) * W, "h": float(row["h"]) * H,
                "foot_x": float(row["foot_x"]), "foot_y": float(row["foot_y"]),
            }
    return traj


def box_dist(px, py, p):
    """Distance from point to nearest edge of player box (0 if inside)."""
    x1, y1 = p["cx"] - p["w"] / 2, p["cy"] - p["h"] / 2
    x2, y2 = p["cx"] + p["w"] / 2, p["cy"] + p["h"] / 2
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def main():
    ball = load_ball()
    players = load_players()
    Hm = np.load(ROOT / "outputs/m1/H_img_to_court.npy")

    events = list(csv.DictReader(open(ROOT / "outputs/m2/events_v4.csv")))
    hits = [e for e in events if e["kind"] == "hit"]

    def ball_at(fi, tol=3):
        for d in range(tol + 1):
            for f in (fi - d, fi + d):
                if f in ball:
                    return ball[f], f
        return None, None

    rows = []
    print(f"{'#':>2} {'frame':>5} {'striker':7} {'heur':5} {'side':>6} "
          f"{'dx_px':>6} {'type':4} {'feet@court':>12}")
    for k, e in enumerate(hits):
        fi = int(e["frame"])
        (bx, by), bframe = ball_at(fi)
        p = players[bframe]
        d_near = box_dist(bx, by, p["near"])
        d_far = box_dist(bx, by, p["far"])
        striker = "near" if d_near < d_far else "far"
        heur = "near" if float(e["court_y_m"]) > NET_Y else "far"

        s = p[striker]
        dx = bx - s["cx"]                       # +: ball on image-right
        right_side = dx > 0 if striker == "near" else dx < 0
        shot_type = "f" if right_side else "b"  # right-handed assumption

        foot = np.float32([[s["foot_x"], s["foot_y"]]]).reshape(-1, 1, 2)
        fx, fy = cv2.perspectiveTransform(foot, Hm).reshape(2)

        rows.append({
            "shot": k + 1, "frame": fi, "striker": striker,
            "striker_heuristic": heur, "agree": striker == heur,
            "ball_dx_px": round(dx, 1), "shot_type": shot_type,
            "striker_court_x": round(float(fx), 2),
            "striker_court_y": round(float(fy), 2),
        })
        print(f"{k+1:>2} {fi:>5} {striker:7} {heur:5} "
              f"{'right' if right_side else 'left':>6} {dx:>6.1f} {shot_type:4} "
              f"({fx:5.2f},{fy:6.2f})")

    agree = sum(r["agree"] for r in rows)
    print(f"\nstriker: proximity vs court-y heuristic agree {agree}/{len(rows)}")
    alt = all(rows[i]["striker"] != rows[i + 1]["striker"] for i in range(len(rows) - 1))
    print(f"striker alternation far/near intact: {alt}")

    with open(OUT_DIR / "shot_types.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {OUT_DIR / 'shot_types.csv'}")


if __name__ == "__main__":
    main()
