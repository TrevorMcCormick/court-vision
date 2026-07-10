"""Box-condition audit for the letter sink — pure analysis, nothing in
the pipeline changed.

Letters (f/b side) are the largest substitution sink after direction v2
(1.71 edits/pt across their two bins; 55% positional accuracy on
length-matched points). The long-suspected root cause is the $0
background-subtraction player boxes going rogue at exactly the wrong
moments — spectators, net-tape ghosts, shadows, player+shadow merges —
because the letter is read as ball-x minus box-center-x at the refined
contact frame, gated on the ball actually reaching the box. A rogue box
either blocks the letter ('?') or poisons the side read.

This script quantifies that story before anything is built on it: on
length-matched points across all 4 matches, every aligned rally letter
(same positional compare as the evals) is binned by the CONDITION of
the striker's box at the contact frame that fed the letter:

  sane        box present, height within [MIN_H_RATIO, MAX_H_RATIO] of
              the match+side median, width within MAX_W_RATIO, and the
              projected foot point inside the striker's half of the
              court (serve-detector stance geometry: foot_x/foot_y
              minus the clip offset, through H_img_to_court)
  implausible box present but failing a sanity check (net-tape sliver,
              spectator, shadow merge)
  absent      no box for the striker's side at the contact frame
              (refusals additionally split: no box+ball frame anywhere
              in the window vs gate-fail at a real box)

Output: outcome (right / wrong / refused) x box condition per match,
plus the box-quality ceiling — letters if every box were sane, at the
sane-box accuracy rate.

Usage:
    uv run experiments/box_letter_audit.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"

MATCHES = ["t1", "t2", "t3", "t4"]
LABELS = {"t1": "t1 night", "t2": "t2 ctrl", "t3": "t3 clay",
          "t4": "t4 grass"}

FH_SIDE = set("frvoul")
BH_SIDE = set("bszpym")
SHOT_CHARS = FH_SIDE | BH_SIDE | set("hijkt")

W_C, L_C = 10.97, 23.77
NET_Y = L_C / 2

# sanity gates, deliberately loose — the audit wants only the frank
# impostors (a 6-px net-tape sliver, a spectator box 3x the player)
MIN_H_RATIO = 0.55
MAX_H_RATIO = 1.9
MAX_W_RATIO = 2.4
# feet may legitimately cross the net line on approach; the margin is
# generous so only boxes clearly in the WRONG half or off-court flag
HALF_SLACK_M = 2.0
CX_SLACK_M = 7.0


def parse_mcp(first, second):
    s = second if second.strip() else first
    strokes = [c for c in s[1:] if c in SHOT_CHARS]
    return strokes


def load_players(path):
    players = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            players.setdefault(int(row["frame"]), {})[row["player"]] = row
    return players


def foot_court(row, Hm, odx, ody):
    pt = np.float32([[float(row["foot_x"]) - odx,
                      float(row["foot_y"]) - ody]]).reshape(-1, 1, 2)
    cx, cy = cv2.perspectiveTransform(pt, Hm).reshape(2)
    return float(cx), float(cy)


def box_condition(row, side, med, Hm, odx, ody):
    """Classify one players-CSV box row: 'sane' or the failing check."""
    h = float(row["h"])
    w = float(row["w"])
    mh, mw = med[side]
    if h < MIN_H_RATIO * mh or h > MAX_H_RATIO * mh:
        return "bad_height"
    if w > MAX_W_RATIO * mw:
        return "bad_width"
    cx, cy = foot_court(row, Hm, odx, ody)
    if not (-CX_SLACK_M <= cx <= W_C + CX_SLACK_M):
        return "off_court_x"
    if side == "near" and cy < NET_Y - HALF_SLACK_M:
        return "wrong_half"
    if side == "far" and cy > NET_Y + HALF_SLACK_M:
        return "wrong_half"
    if side == "near" and cy > L_C + 8:
        return "behind_court"
    if side == "far" and cy < -8:
        return "behind_court"
    return "sane"


def main():
    grand = {}
    print(f"{'match':10}{'outcome':10}{'sane':>7}{'implaus':>9}{'absent':>8}{'total':>7}")
    for t in MATCHES:
        mapd = {r["clip"]: r for r in csv.DictReader(open(DATA / f"{t}_mcp_map.csv"))}
        chart = ROOT / "outputs" / t / "charts_wasb"
        match = {r["clip"]: r for r in csv.DictReader(open(chart / "match_chart_v2.csv"))}
        Hm = np.load(ROOT / "outputs" / t / "H_img_to_court.npy")
        offs = {}
        off_csv = ROOT / "outputs" / t / "clip_offsets.csv"
        if off_csv.exists():
            for r in csv.DictReader(open(off_csv)):
                offs[r["clip"]] = (float(r["dx"]), float(r["dy"]))

        # match+side box scale: median over ALL boxes in the match
        heights = {"near": [], "far": []}
        widths = {"near": [], "far": []}
        pdir = ROOT / "outputs" / t / "players"
        for p in sorted(pdir.glob("players_*.csv")):
            for row in csv.DictReader(open(p)):
                heights[row["player"]].append(float(row["h"]))
                widths[row["player"]].append(float(row["w"]))
        med = {s: (float(np.median(heights[s])), float(np.median(widths[s])))
               for s in ("near", "far")}

        # outcome x condition tallies
        tab = {}          # (outcome, cond) -> n
        reasons = {}      # failing-check name -> n (implausible boxes only)

        def bump(outcome, cond):
            tab[(outcome, cond)] = tab.get((outcome, cond), 0) + 1

        for clip, mc in match.items():
            m = mapd[clip]
            if m["status"] != "matched":
                continue
            strokes = parse_mcp(m["first"], m["second"])
            shots = list(csv.DictReader(open(chart / f"chart2_{clip}.csv")))
            if len(shots) != 1 + len(strokes):
                continue                      # aligned = length-matched only
            players = load_players(pdir / f"players_{clip}.csv")
            odx, ody = offs.get(clip, (0.0, 0.0))
            for k, sh in enumerate(shots):
                if sh["is_serve"] == "True":
                    continue
                mcp_idx = k - 1
                if mcp_idx < 0 or mcp_idx >= len(strokes):
                    continue
                mcp_side = ("f" if strokes[mcp_idx] in FH_SIDE
                            else "b" if strokes[mcp_idx] in BH_SIDE else "?")
                if mcp_side == "?":
                    continue
                side = sh["striker"]
                cf = int(sh["contact_frame"]) if sh["contact_frame"] else None
                row = players.get(cf, {}).get(side) if cf is not None else None
                if row is None:
                    cond = "absent"
                else:
                    c = box_condition(row, side, med, Hm, odx, ody)
                    cond = "sane" if c == "sane" else "implaus"
                    if cond == "implaus":
                        reasons[f"{side}:{c}"] = reasons.get(f"{side}:{c}", 0) + 1
                if sh["letter"] in ("?", ""):
                    outcome = ("refused_nobox"
                               if not sh["contact_dist_px"] else "refused_gate")
                elif sh["letter"] == mcp_side:
                    outcome = "right"
                else:
                    outcome = "wrong"
                bump(outcome, cond)
                grand[(outcome, cond)] = grand.get((outcome, cond), 0) + 1

        for outcome in ("right", "wrong", "refused_gate", "refused_nobox"):
            s = tab.get((outcome, "sane"), 0)
            i = tab.get((outcome, "implaus"), 0)
            a = tab.get((outcome, "absent"), 0)
            if s + i + a == 0:
                continue
            print(f"{LABELS[t]:10}{outcome:10}{s:>7}{i:>9}{a:>8}{s + i + a:>7}")
        print(f"{'':10}medians h/w  near {med['near'][0]:.3f}/{med['near'][1]:.3f}"
              f"  far {med['far'][0]:.3f}/{med['far'][1]:.3f}"
              f"   implaus reasons: "
              + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
        print()

    print("=== overall (all matches, aligned rally letters) ===")
    conds = ("sane", "implaus", "absent")
    tot = {c: sum(grand.get((o, c), 0) for o in
                  ("right", "wrong", "refused_gate", "refused_nobox"))
           for c in conds}
    for outcome in ("right", "wrong", "refused_gate", "refused_nobox"):
        row = f"{outcome:14}"
        for c in conds:
            row += f"{grand.get((outcome, c), 0):>8}"
        print(row + f"{sum(grand.get((outcome, c), 0) for c in conds):>8}")
    print(f"{'total':14}" + "".join(f"{tot[c]:>8}" for c in conds)
          + f"{sum(tot.values()):>8}")

    n_all = sum(tot.values())
    right_sane = grand.get(("right", "sane"), 0)
    acc_sane = right_sane / tot["sane"] if tot["sane"] else 0.0
    right_all = sum(grand.get(("right", c), 0) for c in conds)
    bad = tot["implaus"] + tot["absent"]
    bad_lost = bad - sum(grand.get(("right", c), 0) for c in ("implaus", "absent"))
    print(f"\nletters (aligned, strict): {right_all}/{n_all} "
          f"({100 * right_all / n_all:.0f}%)")
    print(f"sane-box accuracy: {right_sane}/{tot['sane']} "
          f"({100 * acc_sane:.0f}%)")
    print(f"letters on a bad box (implausible or absent): {bad} "
          f"({100 * bad / n_all:.0f}% of aligned letters), "
          f"{bad_lost} of them not right")
    print(f"box-quality ceiling (every box sane, at sane-box accuracy): "
          f"{acc_sane * n_all:.0f}/{n_all} ({100 * acc_sane:.0f}%) — "
          f"vs {right_all} today (+{acc_sane * n_all - right_all:.0f})")


if __name__ == "__main__":
    main()
