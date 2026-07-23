"""Landing-spot extrapolation — the boundary race (blueprint roadmap #1, wired).

Promotes the validated core of experiments/landing_spot.py: infer where an
out-ball would have landed from its SEEN flight, the way a line judge reads
an arc before it lands (nobody photographs the ball touching the ground).
Fit lines to the final flight segment in court coordinates and race two
crossings — sideline first -> wide, baseline first -> deep, both -> x.

Only the classifier is promoted, not an out-DETECTOR: race_final_shot names
the flavor of an out-ball, so endings.infer calls it only where its own
evidence already ran out (ending == '?'). Constants are the experiment's,
chosen a priori from tennis flight time; staged reels are 30fps (ingest).

Stated physics bias (measured, not tuned around): the court projection maps
an airborne ball to its ground shadow, exaggerating DEPTH more than WIDTH,
so the baseline crossing fires early (bias toward 'd'); the sideline reading
is the trustworthy one.
"""

import numpy as np

from .court import W_C, L_C, NET_Y, SINGLES_MARGIN

FPS = 30.0                 # every staged reel is normalised to 30fps (ingest)
OUT_MARGIN = 0.25          # meters of slack before 'out' — endings.py value
FLIGHT_CAP_S = 2.0         # max segment length after final contact
EXTRAP_CAP_S = 0.7         # how far past track death the fit may be trusted
TAIL_K = 8                 # samples in the line fit
HOLE_FRAMES = 6            # a gap this big ends the contiguous segment
MAX_M_PER_FRAME = 3.0      # court m/frame beyond any real ball -> a teleport
                           # (tracker latched onto something else) ends it
BOTH_WIN_FR = 2.0          # crossings this close together -> 'x'

SIDE_L = SINGLES_MARGIN            # singles sidelines in court-x
SIDE_R = W_C - SINGLES_MARGIN


def final_flight_segment(track, contact_frame):
    """Contiguous track run after the final contact, flight-time capped."""
    f, cx, cy = track["frames"], track["cx"], track["cy"]
    m = (f > contact_frame) & (f <= contact_frame + FLIGHT_CAP_S * FPS)
    if not m.any():
        return None
    f, cx, cy = f[m], cx[m], cy[m]
    # cut at the first hole: a fit across a hole is a fit across a fiction
    gaps = np.where(np.diff(f) > HOLE_FRAMES)[0]
    end = gaps[0] + 1 if len(gaps) else len(f)
    f, cx, cy = f[:end].astype(float), cx[:end], cy[:end]
    # cut at the first teleport: balls don't move 3 m per frame — a jump
    # like that is the tracker latching onto something else
    if len(f) > 1:
        step = np.hypot(np.diff(cx), np.diff(cy)) / np.diff(f)
        tele = np.where(step > MAX_M_PER_FRAME)[0]
        if len(tele):
            f, cx, cy = f[:tele[0] + 1], cx[:tele[0] + 1], cy[:tele[0] + 1]
    if len(f) == 0:
        return None
    return f, cx, cy


def infer_target_far(y0, by):
    """Which half is the final shot heading for? Read the flight itself —
    already-out position wins; else the y-slope; else which half it's in."""
    if y0 < 0:
        return True
    if y0 > L_C:
        return False
    if abs(by) > 1e-3:
        return by < 0
    return y0 < NET_Y


def race_boundaries(f, cx, cy):
    """Race the flight line to the sideline vs the baseline.

    Returns (call, t_side, t_base): call in {'w','d','x','?'} with times in
    frames past the last observed sample (0 = already out at track death).
    """
    if len(f) < 4:
        return "?", None, None
    k = min(TAIL_K, len(f))
    t, x, y = f[-k:], cx[-k:], cy[-k:]
    bx = np.polyfit(t, x, 1)[0]            # court m / frame
    by = np.polyfit(t, y, 1)[0]
    x0, y0 = x[-1], y[-1]
    cap = EXTRAP_CAP_S * FPS

    target_far = infer_target_far(y0, by)
    base_y = -OUT_MARGIN if target_far else L_C + OUT_MARGIN
    to_base = base_y - y0
    if target_far:
        t_base = 0.0 if y0 < base_y else (to_base / by if by < 0 else np.inf)
    else:
        t_base = 0.0 if y0 > base_y else (to_base / by if by > 0 else np.inf)

    side_x = (SIDE_R + OUT_MARGIN) if bx > 0 else (SIDE_L - OUT_MARGIN)
    already_wide = not (SIDE_L - OUT_MARGIN <= x0 <= SIDE_R + OUT_MARGIN)
    if already_wide:
        t_side = 0.0
    elif abs(bx) < 1e-9:
        t_side = np.inf
    else:
        t_side = (side_x - x0) / bx
        if t_side < 0:
            t_side = np.inf

    s_in = t_side <= cap
    b_in = t_base <= cap
    if s_in and b_in and abs(t_side - t_base) <= BOTH_WIN_FR:
        return "x", t_side, t_base
    if s_in and t_side <= t_base:
        return "w", t_side, t_base
    if b_in:
        return "d", t_side, t_base
    if s_in:
        return "w", t_side, t_base
    return "?", t_side, t_base


def race_final_shot(last_frame, frames, cxc, cyc):
    """The out-flavor of the final shot's flight ('w'/'d'/'x'), or None when
    the flight can't name a boundary. cxc/cyc are the ball's court-space
    track; last_frame is the final shot's contact frame."""
    track = {"frames": np.asarray(frames),
             "cx": np.asarray(cxc), "cy": np.asarray(cyc)}
    seg = final_flight_segment(track, last_frame)
    if seg is None:
        return None
    call, _, _ = race_boundaries(*seg)
    return call if call in ("w", "d", "x") else None
