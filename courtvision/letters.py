"""The f/b letter read — ball-x vs striker-box-center-x at contact.

The letter is committed only when the ball actually reached the
striker's box: frame checks showed the player boxes go rogue at
exactly the wrong moments (a "far" box on a spectator, a "near" box on
a court shadow), and a letter read 300 px from the ball is a coin flip
in costume. The gate scales with the striker's apparent size — racquet
reach is ~0.6 of body height, plus slack for 1 frame of ball flight.

Height reference: the observed box height under-gates a legs-only
partial blob (h 30 px -> gate 48 px; the ball meets the racquet ~a
body-height above the legs). The clip's own full-body height per side
is the 75th percentile of its boxes — partials drag the median, not
the upper quartile. The gate uses max(observed, typical): rogue boxes
hundreds of px away stay refused, real near-misses commit.

(A multi-frame letter VOTE around contact was measured and rejected:
t3 67 -> 66 strict at ±1 frame, -> 63 at ±3 — post-contact flight
frames poison the median; the single best-frame read stands.)

Logic lifted verbatim from the t*w chart twins (2026-07-10).
"""

import numpy as np

# letter gate scales with the striker's apparent size: racquet reach is
# ~0.6 of body height, plus slack for 1 frame of ball flight
LETTER_GATE = lambda h_px: 0.6 * h_px + 30            # noqa: E731


def typical_heights(players):
    """Per-side 75th-percentile box height (px) across the clip."""
    h_typ = {}
    for side in ("near", "far"):
        hs = [float(p[side]["h"]) * 720
              for p in players.values() if side in p]
        h_typ[side] = float(np.percentile(hs, 75)) if hs else 0.0
    return h_typ


def commit(shots, h_typ, lefty):
    """Set contact_frame / contact_dist_px / letter on every shot from
    its striker-side touch record (sh['touch_<side>'] = (dist, frame,
    ball_x, box_h_px, box_cx_px), found by the chart's contact search)."""
    for sh in shots:
        side = sh["striker"]
        best = sh.get(f"touch_{side}")
        sh["contact_frame"] = best[1] if best else sh["frame"]
        sh["contact_dist_px"] = round(best[0], 1) if best else None
        gate = LETTER_GATE(max(best[3], h_typ[side])) if best else None
        if best and not sh["synth"] and best[0] <= gate:
            dxp = best[2] - best[4]
            right = dxp > 0 if side == "near" else dxp < 0
            forehand = right != lefty[side]
            sh["letter"] = "f" if forehand else "b"
        else:
            sh["letter"] = "?"
