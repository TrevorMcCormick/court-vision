"""Player-box hygiene for the $0 background-subtraction tracks.

The box_letter_audit receipt (2026-07-10): 45% of aligned rally letters
are read off a BAD box — net-tape slivers, spectators, shadows,
player+shadow merges, or no box at all — and those letters run 39%
against 71% on sane boxes. The boxes were always the letter sink's
root cause (LOG, many entries); this module is the cheap fix, shared
by all four t*w chart twins:

  1. PLAUSIBILITY, in court coordinates. Feet must be on the box's own
     half of the court (serve-detector stance geometry: foot minus
     clip offset, through H_img_to_court), with generous slack — only
     frank impostors flag (spectators, ballkids, crowd).
  2. IDENTITY CONTINUITY. A player does not teleport: each surviving
     box's center-x must sit within JUMP_M court-meters of the median
     center-x of its temporal neighbors (radius converted to pixels
     at the box's own depth, so one constant serves both ends).
  3. INTERPOLATION through short dropouts. Frames the two gates
     emptied (or the tracker missed) are filled by linear
     interpolation between surviving neighbors when the gap is short
     — the contact search may then use ANY frame in its window
     instead of falling silent or grabbing a ghost.

A height-vs-depth gate was built, measured, and removed (see the
constants block): box SIZE is bimodal on both ends (full-body vs
legs-only partials) and unusable as an impostor signal; LOCATION
carries the cleaning.

Constants tuned on t3 ONLY (the tuning tree; t1/t2/t4 held out), and
deliberately loose: the module removes impostors, it does not sculpt.

Lifted verbatim from experiments/player_boxes.py (2026-07-10) for the
courtvision package; the experiment script stays frozen as history.

Usage (from courtvision.chart):

    from courtvision import boxes
    players = boxes.load(path, Hm, offset)

Returns the same {frame: {side: row}} mapping the twins built from the
raw CSV; interpolated rows carry synthetic cx/cy/w/h/foot values.
"""

import csv

import cv2
import numpy as np

W_C, L_C = 10.97, 23.77
NET_Y = L_C / 2

# The height-vs-depth gate (k = h_px * mpp(foot) vs a clip reference)
# was BUILT, MEASURED, and REMOVED: the bgsub boxes are bimodal
# (full-body vs legs-only partials) on BOTH ends, so every clip-level
# reference median lands between the modes and the gate kills real
# boxes in full-body clips (t3_point_04: real near AND far boxes read
# ratio 2.4 of their clip's k_ref and their d=0 contacts went '?';
# t3 letters 65 -> 62 strict). A legs-only partial still centers on
# the player — size is NOT a usable impostor signal here. Location is:
# the court-half gate and the teleport gate below carry the cleaning.
# court-half gates (feet, meters) — approach volleys legitimately
# cross toward the net, so the slack is generous
HALF_SLACK_M = 2.0
CX_SLACK_M = 7.0
BEHIND_M = 8.0
# identity continuity: max distance from the temporal-neighbor median
# box-center X, in court meters converted at the box's own depth.
# X-ONLY by measurement: partial blobs (legs vs torso) flicker foot_y
# by 20-40 px, which projects to fake multi-meter court-y teleports at
# the far end — a y term rejected real far boxes wholesale (t3 receipts
# in the LOG). Center-x is the coordinate the letter reads, and it is
# stable on partials.
MED_WIN = 5                   # frames each side
JUMP_M = 2.5
MIN_NEIGHBORS = 3             # window occupancy below this: no verdict
# dropout interpolation cap (frames): ~0.5 s at 25 fps
INTERP_MAX_GAP = 12


def _mpp(Hm, x, y):
    """Meters per vertical pixel at image point (x, y) on the ground
    plane — converts meter radii to pixels at a box's own depth."""
    pts = np.float32([[x, y], [x, y - 1]]).reshape(-1, 1, 2)
    c = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    return float(np.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]))


def load(path, Hm, offset=(0.0, 0.0)):
    """Read a players_*.csv and return {frame: {side: row}} after the
    plausibility gate, the continuity gate, and dropout interpolation.
    Row values are strings (same contract as the raw CSV reader);
    interpolated rows are marked with row['interp'] = '1'."""
    odx, ody = offset
    raw = {"near": {}, "far": {}}
    with open(path) as f:
        for row in csv.DictReader(f):
            raw[row["player"]][int(row["frame"])] = row

    kept = {"near": {}, "far": {}}
    for side in ("near", "far"):
        for fi, row in raw[side].items():
            fx = float(row["foot_x"]) - odx
            fy = float(row["foot_y"]) - ody
            pt = np.float32([[fx, fy]]).reshape(-1, 1, 2)
            cx, cy = cv2.perspectiveTransform(pt, Hm).reshape(2)
            if not (-CX_SLACK_M <= cx <= W_C + CX_SLACK_M):
                continue
            if side == "near" and not (NET_Y - HALF_SLACK_M <= cy
                                       <= L_C + BEHIND_M):
                continue
            if side == "far" and not (-BEHIND_M <= cy
                                      <= NET_Y + HALF_SLACK_M):
                continue
            kept[side][fi] = row

    # identity continuity: distance to the neighbor-median center-x,
    # radius in meters converted at the box's own depth (x-only — see
    # constants block)
    final = {"near": {}, "far": {}}
    for side in ("near", "far"):
        fis = sorted(kept[side])
        cxs = {fi: float(kept[side][fi]["cx"]) * 1280 for fi in fis}
        for fi in fis:
            nb = [cxs[fj] for fj in fis
                  if fj != fi and abs(fj - fi) <= MED_WIN]
            if len(nb) < MIN_NEIGHBORS:
                final[side][fi] = kept[side][fi]     # too thin to judge
                continue
            mx = float(np.median(nb))
            row = kept[side][fi]
            r_px = JUMP_M / max(_mpp(Hm, float(row["foot_x"]) - odx,
                                     float(row["foot_y"]) - ody), 1e-6)
            if abs(cxs[fi] - mx) <= r_px:
                final[side][fi] = row

    # interpolate short dropouts between surviving neighbors
    numeric = ("cx", "cy", "w", "h", "foot_x", "foot_y")
    for side in ("near", "far"):
        fis = sorted(final[side])
        for a, b in zip(fis, fis[1:]):
            if not 1 < b - a <= INTERP_MAX_GAP:
                continue
            ra, rb = final[side][a], final[side][b]
            for fi in range(a + 1, b):
                t = (fi - a) / (b - a)
                row = {"frame": str(fi), "player": side, "interp": "1"}
                for k2 in numeric:
                    row[k2] = str(float(ra[k2])
                                  + t * (float(rb[k2]) - float(ra[k2])))
                final[side][fi] = row

    players = {}
    for side in ("near", "far"):
        for fi, row in final[side].items():
            players.setdefault(fi, {})[side] = row
    return players
