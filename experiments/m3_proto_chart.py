"""M3 scoping: hand-roll a proto-chart from the M2 events and let every
field we CAN'T fill define what M3 actually needs.

Target format is the Match Charting Project shot code string (see
projects/tennis data dictionary): e.g. `5b2f3n@` = serve to 5, backhand to
zone 2, forehand to zone 3, netted, unforced error. Per shot: type letter
(f/b/...), direction digit (1/2/3), optional depth (7/8/9), ending code.

What the pipeline has per shot: who struck it (near/far end from hit
position), where it landed (next bounce, court coords), rally length.
What it doesn't have gets a '?'.

Usage:
    uv run experiments/m3_proto_chart.py outputs/m2/events_v4.csv \
        outputs/m0/trajectory_ballfix.csv outputs/m1/H_img_to_court.npy
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"

W_COURT = 10.97
L_COURT = 23.77
NET_Y = L_COURT / 2
SVC_FAR_Y = NET_Y - 6.40
SVC_NEAR_Y = NET_Y + 6.40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events_csv")
    parser.add_argument("traj_csv")
    parser.add_argument("h_npy")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    H = np.load(args.h_npy)

    boxes = {}
    with open(args.traj_csv) as f:
        for row in csv.DictReader(f):
            boxes[int(row["frame"])] = (float(row["cx"]) * 1280, float(row["cy"]) * 720)

    def court_pos(fi, win=4):
        pts = [boxes[f] for f in range(fi - win, fi + win + 1) if f in boxes]
        proj = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)
        return np.median(proj, axis=0)

    events = []
    with open(args.events_csv) as f:
        for row in csv.DictReader(f):
            fi = int(row["frame"])
            x, y = court_pos(fi)
            events.append({"frame": fi, "kind": row["kind"], "x": x, "y": y})

    hits = [e for e in events if e["kind"] == "hit"]
    bounces = [e for e in events if e["kind"] == "bounce"]

    # MCP direction zones: thirds of the singles court width, but MCP
    # defines them relative to the RECEIVER's handedness — unknown. Use
    # absolute thirds and flag it.
    def zone(x):
        third = W_COURT / 3
        return 1 if x < third else (2 if x < 2 * third else 3)

    def depth(y):
        # returns/groundstrokes: 7 = shallow, 8 = moderate, 9 = deep (MCP
        # uses these for RETURN depth only; we compute for every landing)
        if y > NET_Y:  # near half
            d = L_COURT - y
        else:
            d = y
        return 9 if d < 4.0 else (8 if d < 7.5 else 7)

    shots = []
    for k, h in enumerate(hits):
        striker = "near" if h["y"] > NET_Y else "far"
        nxt_hit_frame = hits[k + 1]["frame"] if k + 1 < len(hits) else 10 ** 9
        landing = next((b for b in bounces
                        if h["frame"] < b["frame"] < nxt_hit_frame), None)
        far_landing = landing is not None and landing["y"] < NET_Y
        shots.append({
            "shot": k + 1,
            "striker": striker,
            "struck_at_frame": h["frame"],
            "shot_type": "?",                     # f/b: needs player + contact side
            "direction_zone": zone(landing["x"]) if landing else "?",
            "landing_y_m": round(float(landing["y"]), 1) if landing else "?",
            "depth_code": depth(landing["y"]) if landing else "?",
            "landing_trust": ("low (far-side bounce, ±m)" if far_landing
                              else "good") if landing else "-",
            "ending": "?" if k + 1 == len(hits) else "",
        })

    with open(OUT_DIR / "proto_chart.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(shots[0].keys()))
        wr.writeheader()
        wr.writerows(shots)

    print("PROTO-CHART — one rally, shot by shot")
    print(f"{'#':>2} {'striker':7} {'type':4} {'dir':3} {'depth':5} {'landing':>9}  trust")
    for s in shots:
        print(f"{s['shot']:>2} {s['striker']:7} {s['shot_type']:4} "
              f"{str(s['direction_zone']):3} {str(s['depth_code']):5} "
              f"{str(s['landing_y_m']):>7}m  {s['landing_trust']}")

    mcp = "?"  # serve: not in clip
    for s in shots:
        mcp += f"{s['shot_type']}{s['direction_zone']}"
    mcp += "?"  # ending
    print(f"\npseudo-MCP string: {mcp}")
    print(f"rallyCount: {len(shots)} (+ unseen serve)")

    print("\nWHAT '?' MEANS — the M3 requirements list:")
    print(" 1. shot_type f/b        -> needs PLAYER positions (M0 unfinished business)")
    print(" 2. serve codes          -> clip starts mid-rally; need full points from serve")
    print(" 3. ending code (*/@/#)  -> need to see the point end + call in/out")
    print(" 4. direction 1/2/3      -> derivable but MCP defines zones vs receiver")
    print("                            handedness; needs player identity")
    print(" 5. point/game context   -> needs rally segmentation across a full match")
    print("-> outputs/m3/proto_chart.csv")


if __name__ == "__main__":
    main()
