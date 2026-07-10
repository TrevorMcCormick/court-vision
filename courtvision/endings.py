"""Ending inference v1 — observable evidence only.

The last shot's own landing (far-half only, by construction) codes
out-deep/-wide; a ball track that dies at the net right after the last
hit codes a net error. Winner vs forced vs unforced is HUMAN judgment
the pipeline does not attempt: '@' means "error, attribution not
judged" and eval compares the ending TYPE only.

Near-half ending fill (the near_fill flag; t3/t4 staging) — near-half
landings are invisible to the collapse detector BY CONSTRUCTION (it
skips cy > NET_Y), so a final shot struck by the FAR player never got
an ending. The dense WASB track recovers the bounce as an image-y
V-cusp below the net line, and the position AT the cusp is trustworthy
exactly there: the ball is ON the ground plane at the bounce instant.
The trap is the SECOND bounce / the dead ball at the collector's feet
(freeze #3's own boundary: real near hits <= 25.5, ball-at-feet
>= 26.4) — winners' late cusps read cy 25.6-30 and miscode d/x. The
true first bounce arrives within flight time, so the search stops at
NEAR_BOUNCE_WIN_S: measured on t3+t4, the 2.0-s window commits 6 fills
with 3 misses (all late cusps, all >= 25.6), the 1.2-s window commits
3 with 0. Fill-only: it never overrides a far-half landing or a net
death.

Logic lifted verbatim from the t*w chart twins (2026-07-10).
"""

import numpy as np

from .court import W_C, L_C, NET_Y, SINGLES_MARGIN, moving_average

SMOOTH = 3
SWING_NEAR_30 = 6.0
OUT_MARGIN = 0.25             # meters of slack before calling a ball out
NET_ZONE_M = 1.5              # track dying this close to the net = net error
NEAR_BOUNCE_WIN_S = 1.2       # s after the final shot; flight time only
NEAR_DEEP_M = 0.5             # m behind the near baseline before 'deep'
NEAR_CY_CEIL = 8.0            # m; cusps beyond L_C+this are not bounces


def infer(shots, frames, ys, cyc, cxc, fps, near_fill=False):
    """Ending code for the point: '*', 'n@', 'w@', 'd@', 'x@', or '?'."""
    ending = "?"
    last = shots[-1]
    if last.get("landing_y") is not None:
        ly, lx = last["landing_y"], last.get("landing_x")
        deep = ly < -OUT_MARGIN
        wide = lx is not None and not (
            SINGLES_MARGIN - OUT_MARGIN <= lx <= W_C - SINGLES_MARGIN + OUT_MARGIN)
        ending = ("x@" if deep and wide else "d@" if deep
                  else "w@" if wide else "*")   # in, and nothing came back
    else:
        li = int(np.searchsorted(frames, last["frame"]))
        tail_y, tail_f = cyc[li:], frames[li:]
        if (len(tail_y) >= 3
                and abs(float(tail_y[-1]) - NET_Y) < NET_ZONE_M
                and tail_f[-1] - last["frame"] <= 1.2 * fps):
            ending = "n@"
    if ending == "?" and near_fill:
        # near-half ending fill (see module doc): first image-y V-cusp
        # in the near half within flight time of the final shot
        iy_s2 = moving_average(ys, SMOOTH)
        viy2 = np.gradient(iy_s2, frames.astype(float))
        swing_min = SWING_NEAR_30 * 30.0 / fps
        lf = last["frame"]
        for i in range(2, len(frames) - 2):
            if frames[i] <= lf + 2 or frames[i] > lf + NEAR_BOUNCE_WIN_S * fps:
                continue
            if frames[i + 1] - frames[i - 1] > 6:
                continue      # cusp straddling a track hole is not a bounce
            if (viy2[i - 1] > 0 and viy2[i + 1] < 0
                    and viy2[i - 1] - viy2[i + 1] >= swing_min
                    and NET_Y + 1 < cyc[i] < L_C + NEAR_CY_CEIL):
                by, bx = float(cyc[i]), float(cxc[i])
                deep = by > L_C + NEAR_DEEP_M
                wide = not (SINGLES_MARGIN - OUT_MARGIN <= bx
                            <= W_C - SINGLES_MARGIN + OUT_MARGIN)
                ending = ("x@" if deep and wide else "d@" if deep
                          else "w@" if wide else "*")
                break
    return ending
