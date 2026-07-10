"""Shot-direction estimator — MCP digits 1/2/3 for BOTH halves.

Two findings drive this module (calibration receipts in
dir_calibrate.py and the LOG):

  SEMANTICS  MCP's direction digit is RECEIVER-END geometric, not
             absolute and not handedness-flipped: "1 = a right-hander's
             forehand side (a lefty's backhand)" names a fixed side of
             the receiving half. In our court frame (x=0 image-left)
             that is ascending thirds toward a FAR receiver and
             descending toward a NEAR one. Tested on all 150 committed,
             aligned, length-matched landings across the 4 matches:
             abs-asc (the old zone()) 57/150, receiver-mirror 79/150,
             handedness-flipped variant 73/150 (the both-lefty t1 match
             votes 11/24 vs 5/24 AGAINST the flip).

  SPARSITY   the far-half-only collapse landing reaches 70% of shots at
             48% accuracy. Every shot in the dense-WASB era carries two
             more where-did-it-go signals: the RECEIVER'S CONTACT point
             for the next shot (where the ball was received IS where
             this shot went, modulo receiver reach), and the ball's x
             at its NET CROSSING plus the flight's dx/dy slope
             extrapolated into the receiving half (crossings are the
             v5 rally skeleton). Signal quality measured on t3 ONLY
             (dir_signals_dev.py); t1/t2/t4 held out.

Precedence, measured not assumed (t3, n=113 aligned pairs):
  near-half landing 86% > receiver contact 77% > crossing+slope 67%
  > far-half landing 47%. The top-priority available signal COMMITS;
  '?' only when no signal exists. The disagreement veto was measured
  and rejected: refusing when the runner-up disagrees kept precision
  at 78% vs 77% but cut net-right tokens 85 -> 62 — under acceptance,
  refusal and error cost the same edit, so a 77% guess strictly beats
  a refusal. The refusal path stays for the signal-less case only.

Shared by all four t*w chart twins:

    import shot_direction
    shot_direction.annotate(shots, frames, cyc, cxc, fps)
"""

import numpy as np

W_C, L_C = 10.97, 23.77
NET_Y = L_C / 2

# extrapolation depth for the crossing signal: evaluate the flight line
# this many meters past the net into the receiving half. Swept 0-12 m
# on t3 ONLY: accuracy rises monotonically 51% -> 67% and plateaus at
# the receiver's BASELINE (net-to-baseline = 11.885) — where the
# receiver actually meets a rally ball
CROSS_DEPTH_M = NET_Y
# slope fit uses track samples within this many meters of the net line
CROSS_FIT_M = 5.0


def direction_digit(x, receiver_end):
    """Court-x -> MCP direction digit for a ball arriving at
    receiver_end ('near'/'far'). Geometric receiver-end mirror —
    empirically calibrated, NOT handedness-flipped (see module doc)."""
    third = W_C / 3
    z = 0 if x < third else (1 if x < 2 * third else 2)
    if receiver_end == "near":
        z = 2 - z
    return str(z + 1)


def landing_signal(sh):
    """(x, receiver_end) from the shot's own landing, if present.
    Receiver end is the half the landing was MEASURED in (geo)."""
    lx, ly = sh.get("landing_x"), sh.get("landing_y")
    if lx is None or ly is None:
        return None
    return float(lx), ("far" if float(ly) < NET_Y else "near")


def contact_signal(sh, nxt, frames, cxc):
    """(x, receiver_end) from the NEXT shot's refined contact point —
    where the ball was received is where this shot went. Uses the ball's
    projected court-x at the receiver's contact frame; the projection's
    y is garbage for an airborne ball but x degrades gently."""
    if nxt is None or nxt.get("synth"):
        return None
    fc = nxt.get("contact_frame")
    if fc is None:
        return None
    i = int(np.searchsorted(frames, fc))
    if i >= len(frames) or frames[i] != fc:
        return None                      # contact frame not on the track
    return float(cxc[i]), nxt.get("striker")


def crossing_signal(sh, nxt, frames, cyc, cxc, fps):
    """(x, receiver_end) from the ball's net crossing between this shot
    and the next: x at the net line plus the flight's dx/dy slope,
    extrapolated CROSS_DEPTH_M into the receiving half."""
    f0 = sh["frame"]
    f1 = nxt["frame"] if nxt is not None else int(frames[-1]) + 1
    i0 = int(np.searchsorted(frames, f0 + 1))
    i1 = int(np.searchsorted(frames, f1))
    s = np.sign(cyc[i0:i1] - NET_Y)
    flips = np.where(s[:-1] * s[1:] < 0)[0]
    if len(flips) == 0:
        return None
    i = i0 + int(flips[0])               # first net pass after the shot
    receiver = "near" if cyc[i + 1] > cyc[i] else "far"
    # slope fit on samples near the net line, this flight only
    lo = i
    while lo - 1 >= i0 and abs(cyc[lo - 1] - NET_Y) <= CROSS_FIT_M:
        lo -= 1
    hi = i + 1
    while hi + 1 < i1 and abs(cyc[hi + 1] - NET_Y) <= CROSS_FIT_M:
        hi += 1
    ys, xs = cyc[lo:hi + 1], cxc[lo:hi + 1]
    if len(ys) < 3 or abs(ys[-1] - ys[0]) < 1e-6:
        return None
    slope = np.polyfit(ys, xs, 1)[0]
    # interpolated x at the net line
    y_a, y_b = cyc[i], cyc[i + 1]
    t = (NET_Y - y_a) / (y_b - y_a)
    x_net = float(cxc[i] + t * (cxc[i + 1] - cxc[i]))
    depth = CROSS_DEPTH_M if receiver == "near" else -CROSS_DEPTH_M
    return x_net + slope * depth, receiver


def estimate(sh, nxt, frames, cyc, cxc, fps):
    """Direction digit for one rally shot, or '?'.

    Priority order is the measured t3 quality ladder: near-half landing
    (86%), receiver contact (77%), crossing+slope (67%), far-half
    landing (47%). The best available signal commits; '?' only when
    nothing measured this shot's flight (see module doc for the
    measured commit-vs-veto tradeoff)."""
    land = landing_signal(sh)
    cont = contact_signal(sh, nxt, frames, cxc)
    cross = crossing_signal(sh, nxt, frames, cyc, cxc, fps)

    ranked = []
    if land is not None and land[1] == "near":
        ranked.append(("land_near", direction_digit(*land)))
    if cont is not None and cont[1] in ("near", "far"):
        ranked.append(("contact", direction_digit(*cont)))
    if cross is not None:
        ranked.append(("cross", direction_digit(*cross)))
    if land is not None and land[1] == "far":
        ranked.append(("land_far", direction_digit(*land)))

    if not ranked:
        return "?", ""
    name, top = ranked[0]
    return top, name


def annotate(shots, frames, cyc, cxc, fps):
    """Set sh['zone'] for every non-serve shot from the estimator;
    serves keep their serve_zone. Returns per-signal commit counts."""
    stats = {}
    for k, sh in enumerate(shots):
        if sh.get("is_serve"):
            continue
        nxt = shots[k + 1] if k + 1 < len(shots) else None
        d, why = estimate(sh, nxt, frames, cyc, cxc, fps)
        sh["zone"] = d
        key = why if d != "?" else (f"refuse:{why}" if why else "refuse:none")
        stats[key] = stats.get(key, 0) + 1
    return stats
