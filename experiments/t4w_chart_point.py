"""T4 — chart loop on WASB tracks, grass adversarial match (Wimbledon
2024 final, Krejcikova-Paolini, both right-handed). Byte-twin of
t2w_chart_point.py except paths, plus the t3 clip-offset correction
(outputs/t4/clip_offsets.csv) for the wandering camera. All frozen-era
constants and the freeze-#3 envelope gate are untouched.


v1 assigned striker by proximity at the detected event frame; frame-checks
showed the cusp often catches the ball mid-air BETWEEN players (the apex),
so proximity flips the striker mid-rally and alternation breaks (point_16
read near,near,near,far,far,far,far,far — tennis doesn't work like that).

v2's fixes, all free on the six tracks already paid for:
  striker   strict alternation anchored on the serve. The first idea —
            vote by ball direction after the hit (vcy sign) — died on
            contact with the data: the homography assumes the ball is ON
            the ground plane, so an airborne ball's "court velocity" is
            dominated by its vertical motion in image space; post-hit vcy
            reads negative for BOTH ends. Physics votes withdrawn.
  misses    alternation over DETECTED hits breaks when a hit hides in a
            ball-track hole (point_53: SAM loses the ball f54-78, one far
            hit vanishes, every striker after it flips). Holes are
            observable, so: at each inter-hit interval containing a hole,
            consider a parity flip, adjudicated by far-half landing votes
            (collapse bounces are far-half-only by construction — the one
            side the geometry lets us see).
  contact   refined to the frame the ball is nearest the ASSIGNED
            striker's box — proximity is fine once the striker is already
            known. The f/b letter is committed only when the ball actually
            reached the box (<= LETTER_MAX_PX); frame checks showed the
            player boxes go rogue at exactly the wrong moments (a "far"
            box on a spectator, a "near" box on a court shadow), and a
            letter read 300 px from the ball is a coin flip in costume.
  serve     three pilot clips had a confident serve call but no 's' in the
            string: the detector missed serve contact. v2 synthesizes the
            serve at the gated serve frame so it anchors alternation, and
            its landing codes 4/5/6 via service-box thirds (v1 defined
            serve_zone() and never called it — strings showed full-court
            thirds for serves).

Landing-vote conflicts with the final chain are counted and printed, not
hidden: they are the residual missed-hit / bogus-bounce signal.

Usage:
    uv run experiments/t4w_chart_point.py t4_point_01 ...
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import events_v5
import player_boxes
import shot_direction

# EVENT DETECTOR: "v5" = the crossing-skeleton detector (events_v5.py,
# 2026-07-10) — net crossings partition the rally and each partition
# contains exactly one hit; the M2-era cusp classifier only locates
# the hit inside its slot. "v4" reruns the old detector unchanged.
EVENTS = "v5"

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "t4"
CHART_DIR = OUT_BASE / "charts_wasb"
ROOT = Path(__file__).resolve().parent.parent

SMOOTH = 3
WIN = 6
MIN_GAP_S = 8 / 30.0
SWING_NEAR_30 = 6.0
SWING_FAR_30 = 1.2
FAR_Y_PX = 250
HIT_SPEED = 4.5
COLLAPSE = 3.0
W_C, L_C = 10.97, 23.77
NET_Y = L_C / 2
CENTER_X = W_C / 2
# freeze #3 (WASB re-tune, the ONLY constants changed from the frozen
# loop): dense specialist tracks outlive the point. SAM's track died
# with the rally; WASB keeps detecting the dead ball — bouncing at the
# near player's feet after the winner, drifting through the crowd above
# the back fence — and the cusp detector charts the coda as extra shots
# (t1 rally ±1 fell 7/11 -> 5/11 on identical logic). Frame-checked
# phantoms (03 f243/f252, 04 f142, 06 f141/f158, 25 f412) ALL live
# outside the playable envelope in court-y: >= 26.4 behind the near
# baseline (foreground dead ball) or <= -18 behind the far one (a ball
# in the crowd is nowhere near the ground plane the homography assumes).
# Real rally hits across BOTH trees span -8.3 .. 25.5. Serves are
# exempt inside SERVE_WIN of the gated serve frame: a high toss maps far
# outside the envelope by the same distortion (04's real far serve reads
# cy -20.3; t2's near serves swing past every magnitude cap we tried).
HIT_CY_MIN = -12.0            # m; deepest real far-end hit observed -8.3
HIT_CY_MAX = L_C + 2.2        # m; highest real near-end hit observed 25.5
SERVE_WIN = 14                # frames; same window is_serve already uses
REFINE = 14                   # frames each side of the cusp to hunt contact
# letter gate scales with the striker's apparent size: racquet reach is
# ~0.6 of body height, plus slack for 1 frame of ball flight
LETTER_GATE = lambda h_px: 0.6 * h_px + 30
# (a multi-frame letter VOTE around contact was measured and rejected:
# t3 67 -> 66 strict at ±1 frame, -> 63 at ±3 — post-contact flight
# frames poison the median; the single best-frame read stands)
HOLE_FRAMES = 8               # ball-track gap that can hide a missed hit
OUT_MARGIN = 0.25             # meters of slack before calling a ball out
NET_ZONE_M = 1.5              # track dying this close to the net = net error
# t4 coda truncation — the in-plateau dead-ball coda. Wimbledon never
# cuts after points, so the ball's aftermath (bouncing at the near
# player's feet, drifting past the far baseline to the kids, swatted
# back across between points) lives inside the same score-bug plateau
# AND inside the playable court-y envelope; freeze #3 can't see it and
# the cusp detector charted it as extra shots (21/49 rally ±1, 17 clips
# over by >1). Pixel-diagnosed on points 01/05/24/28/45/48: after the
# true final shot BOTH players break rally posture and walk while the
# ball keeps moving — and per-event geometry (landing positions, vcy,
# player speed, track coverage, static-lock runs) does NOT separate
# coda events from live ones (all measured, all refuted; landings read
# 5-45 m deep on LIVE rally shots because the collapse position is
# projection garbage on a rising ball). What DOES hold: a live rally
# sends the ball across the net every shot, so the point's spine is the
# track's net-crossing sequence. Two sequence-level rules:
#   dead-gap  a shot followed by > DEAD_GAP_S with no crossing inside
#             the gap ended the point (longest live inter-shot gap
#             observed 2.5 s; codas idle 3.1-3.8 s). Not applied at a
#             synth serve — dead air after a stance-gated serve frame
#             is the between-points shuffle, not a coda.
#   anchor    shots more than LAUNCH_SLACK_S after the LAST crossing's
#             start didn't launch it; the first of them within RECV_S
#             of the crossing's end is kept (the receiving shot — a net
#             error or a hole-hidden out-flight never crosses); the
#             rest are coda.
# Crossing gates, measured on this reel: smoothed court-y monotone runs,
# >= 4 samples, span 5-40 m (real deep-lob flights project up to 32 m;
# track teleports onto crowd/kids run 43-53 m), speed 4-90 m/s, no
# single step > 6 m/frame. On the 21 rally-correct clips the pass
# regresses three (12/18/20: long rallies whose late track wanders the
# far run-off with no crossings — byte-similar to point_28's TRUE coda;
# the evidence cannot tell them apart) and fixes eight. t4-only: the
# same pass on t3 is net negative (clay undercounts, crossings recall
# pays), so the t3 twin doesn't grow it.
DEAD_GAP_S = 3.0              # s; no-crossing gap that ends the point
LAUNCH_SLACK_S = 0.6          # s; shot -> its crossing's start, max lag
RECV_S = 1.2                  # s; crossing end -> receiving shot, max lag
CROSS_SPAN_MIN_M = 5.0        # m; runs shorter than this aren't flights
CROSS_SPAN_MAX_M = 40.0       # m; longer = track teleport, not a ball
CROSS_SPEED_MIN, CROSS_SPEED_MAX = 4.0, 90.0   # m/s, projected
CROSS_MIN_SAMPLES = 4
CROSS_MAX_STEP_M = 6.0        # m/frame; single-step teleport guard
CROSS_MAX_FRAME_GAP = 5       # frames; track hole that breaks a run
# near-half ending fill — near-half landings are invisible to the
# collapse detector BY CONSTRUCTION (it skips cy > NET_Y), so a final
# shot struck by the FAR player never got an ending. The dense WASB
# track recovers the bounce as an image-y V-cusp below the net line,
# and the position AT the cusp is trustworthy exactly there: the ball
# is ON the ground plane at the bounce instant. The trap is the SECOND
# bounce / the dead ball at the collector's feet (freeze #3's own
# boundary: real near hits <= 25.5, ball-at-feet >= 26.4) — winners'
# late cusps read cy 25.6-30 and miscode d/x. The true first bounce
# arrives within flight time, so the search stops at NEAR_BOUNCE_WIN_S:
# measured on t3+t4, the 2.0-s window commits 6 fills with 3 misses
# (all late cusps, all >= 25.6), the 1.2-s window commits 3 with 0.
# Fill-only: it never overrides a far-half landing or a net death.
NEAR_BOUNCE_WIN_S = 1.2       # s after the final shot; flight time only
NEAR_DEEP_M = 0.5             # m behind the near baseline before 'deep'
NEAR_CY_CEIL = 8.0            # m; cusps beyond L_C+this are not bounces
# Per-match handedness — freeze #2's one allowed knob. All four players
# in t3/t4 are right-handed (Djokovic, Ruud, Krejcikova, Paolini —
# looked up, not assumed), so no mirror anywhere.
LEFTY = {"near": False, "far": False}


def moving_average(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def detect_events(frames, iy, cyc, fps, serve_frame=None):
    """M2 v4, fps-parameterized; hits carry their post-hit vcy median.
    freeze #3: cusps outside the playable court-y envelope are dropped
    (dead-ball coda on dense WASB tracks), serve window exempt."""
    min_gap = max(3, int(round(MIN_GAP_S * fps)))
    swing_near = SWING_NEAR_30 * 30.0 / fps
    swing_far = SWING_FAR_30 * 30.0 / fps

    iy_s = moving_average(iy, SMOOTH)
    viy = np.gradient(iy_s, frames.astype(float))
    vcy = np.gradient(cyc, frames.astype(float)) * fps

    cusps = []
    for i in range(2, len(frames) - 2):
        if viy[i - 1] > 0 and viy[i + 1] < 0:
            swing = viy[i - 1] - viy[i + 1]
            need = swing_near if iy_s[i] >= FAR_Y_PX else swing_far
            if swing >= need:
                cusps.append(i)
    merged = []
    for i in sorted(cusps, key=lambda i: -(viy[i - 1] - viy[i + 1])):
        if all(abs(int(frames[i]) - int(frames[j])) >= min_gap for j in merged):
            merged.append(i)
    merged.sort()
    # freeze #3: the playable-envelope gate (see constants block)
    merged = [i for i in merged
              if HIT_CY_MIN <= cyc[i] <= HIT_CY_MAX
              or (serve_frame is not None
                  and abs(int(frames[i]) - serve_frame) <= SERVE_WIN)]

    collapse = []
    for i in range(WIN, len(frames) - WIN):
        if cyc[i] > NET_Y:
            continue
        mb = np.median(vcy[i - WIN:i])
        ma = np.median(vcy[i + 1:i + 1 + WIN])
        if np.sign(mb) == np.sign(ma) and abs(ma) > 0.3 and abs(mb) / abs(ma) >= COLLAPSE:
            collapse.append(i)
    bounds = [-1] + [int(frames[i]) for i in merged] + [10 ** 9]
    last_per_seg = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = [i for i in collapse if a + min_gap <= frames[i] <= b - min_gap]
        if seg:
            last_per_seg.append(seg[-1])

    events = []
    for i in sorted(merged + last_per_seg):
        how = "cusp" if i in merged else "collapse"
        ma = float(np.median(vcy[i + 1:i + 1 + WIN]))
        kind = "bounce" if how == "collapse" else ("hit" if abs(ma) > HIT_SPEED else "bounce")
        pos_y = float(np.median(cyc[i + 1:i + 1 + WIN])) if how == "collapse" else float(cyc[i])
        events.append({"idx": i, "frame": int(frames[i]), "kind": kind,
                       "signal": how, "court_y": pos_y, "vcy_after": ma})
    return events


def net_crossings(frames, cyc, fps):
    """Sustained monotone court-y runs that pass the net at rally speed —
    the point's spine (see the coda constants block for the gates).
    Returns [(start_frame, end_frame)]."""
    cs = moving_average(cyc, 3)
    out = []
    n = len(frames)
    i = 0
    while i < n - 1:
        d = np.sign(cs[i + 1] - cs[i])
        if d == 0 or frames[i + 1] - frames[i] > CROSS_MAX_FRAME_GAP:
            i += 1
            continue
        j = i
        while (j + 1 < n and np.sign(cs[j + 1] - cs[j]) == d
               and frames[j + 1] - frames[j] <= CROSS_MAX_FRAME_GAP):
            j += 1
        lo, hi = cs[i], cs[j]
        span = abs(hi - lo)
        dur = (frames[j] - frames[i]) / fps
        steps = np.abs(np.diff(cs[i:j + 1])) / np.maximum(np.diff(frames[i:j + 1]), 1)
        if (j - i + 1 >= CROSS_MIN_SAMPLES
                and CROSS_SPAN_MIN_M <= span <= CROSS_SPAN_MAX_M
                and min(lo, hi) < NET_Y - 1 and max(lo, hi) > NET_Y + 1
                and dur > 0
                and CROSS_SPEED_MIN <= span / dur <= CROSS_SPEED_MAX
                and steps.max() <= CROSS_MAX_STEP_M):
            out.append((int(frames[i]), int(frames[j])))
        i = j
    return out


def truncate_coda(shots, xruns, fps):
    """Drop shots after the point already ended (dead-gap and anchor rules,
    see the coda constants block). Returns (kept_shots, n_dropped, why)."""
    n = len(shots)
    new_n, why = n, ""
    for k in range(n - 1):
        if k == 0 and shots[0].get("synth"):
            continue
        f0, f1 = shots[k]["frame"], shots[k + 1]["frame"]
        if (f1 - f0) / fps > DEAD_GAP_S and not any(f0 < a < f1 for a, _ in xruns):
            new_n, why = k + 1, f"dead-gap({(f1 - f0) / fps:.1f}s)"
            break
    if xruns:
        ax, axe = xruns[-1]
        keep = sum(1 for s in shots[:new_n]
                   if s["frame"] <= ax + LAUNCH_SLACK_S * fps)
        rest = [s for s in shots[:new_n]
                if s["frame"] > ax + LAUNCH_SLACK_S * fps]
        if rest and rest[0]["frame"] <= axe + RECV_S * fps:
            keep += 1
        if keep < new_n:
            new_n = keep
            why = (why + " " if why else "") + f"anchor(f{ax}-{axe})"
    new_n = max(new_n, 1)
    return shots[:new_n], n - new_n, why


def serve_zone(bx, deuce, server):
    """Thirds of the receiving service box toward the center line: 4 wide,
    5 body, 6 T. Coarse, and honest about it."""
    if server == "near":
        left = deuce
    else:
        left = not deuce
    x0, x1 = (0 + 1.372, CENTER_X) if left else (CENTER_X, W_C - 1.372)
    t = (bx - x0) / (x1 - x0)
    t = min(max(t, 0.0), 1.0)
    toward_center = t if not left else 1 - t
    return 6 if toward_center < 1 / 3 else (5 if toward_center < 2 / 3 else 4)


def box_dist(px, py, p):
    x1 = (float(p["cx"]) - float(p["w"]) / 2) * 1280
    y1 = (float(p["cy"]) - float(p["h"]) / 2) * 720
    x2 = (float(p["cx"]) + float(p["w"]) / 2) * 1280
    y2 = (float(p["cy"]) + float(p["h"]) / 2) * 720
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


W_TOUCH, W_SERVE, W_LAND = 3.0, 2.0, 1.0
FLIP_COST = 0.75


def assign_strikers(shots, server, holes, lock=None):
    """Strict alternation, with a parity flip allowed at each inter-hit
    interval containing a ball-track hole (a hidden hit keeps true
    alternation but repeats the striker over DETECTED shots). The chain is
    chosen by weighted votes, strongest first:
      touch  (3)  the ball reached exactly one player's box in the
                  contact window — the racquet adjudicates
      serve  (2)  the gated serve call — a PRIOR, not an anchor; point_53's
                  wrong-end serve call must be outvotable by the ball
      landing(1)  a far-half bounce after shot k says NEAR struck it
                  (collapse bounces are far-half-only by construction)
    lock: when a confident serve call anchors shot 0, the chain's first
    striker is LOCKED to it and only the flip slots stay in play —
    measured on this staging pair, the serve detector beats the touch
    votes (t4: all 16 chart overrides of a confident serve flipped a
    CORRECT end to wrong; the white-on-white boxes that fed those
    votes are the same ones the letter gate already distrusts)."""
    other = {"near": "far", "far": "near"}

    votes = []          # per shot: list of (side, weight)
    for sh in shots:
        v = []
        near_d, far_d = sh.get("touch_near"), sh.get("touch_far")
        near_ok = near_d is not None and near_d[0] <= LETTER_GATE(near_d[3])
        far_ok = far_d is not None and far_d[0] <= LETTER_GATE(far_d[3])
        if near_ok != far_ok:
            v.append(("near" if near_ok else "far", W_TOUCH))
        if sh.get("anchor"):
            v.append((server, W_SERVE))
        if sh.get("landing_y") is not None and -1.0 < sh["landing_y"] < NET_Y:
            v.append(("near", W_LAND))
        votes.append(v)

    flip_slots = [k for k in range(len(shots) - 1)
                  if any(a < shots[k + 1]["frame"] and b > shots[k]["frame"]
                         for a, b in holes)][:4]

    best = None
    for first in (("near", "far") if lock is None else (lock,)):
        for mask in range(2 ** len(flip_slots)):
            chain, cur = [], first
            for k in range(len(shots)):
                chain.append(cur)
                cur = other[cur]
                if k in flip_slots and mask >> flip_slots.index(k) & 1:
                    cur = other[cur]      # hidden hit: striker repeats
            score = -FLIP_COST * bin(mask).count("1")
            for v, c in zip(votes, chain):
                for side, w in v:
                    score += w if side == c else -w
            if best is None or score > best[0]:
                best = (score, chain)

    chain = best[1]
    conflicts = sum(1 for v, c in zip(votes, chain)
                    for side, _ in v if side != c)
    for sh, c in zip(shots, chain):
        sh["striker"] = c
    return conflicts


def chart_clip(stem, Hm, serves):
    clip = ROOT / "clips/points_t4" / f"{stem}.mp4"
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ball = list(csv.DictReader(open(OUT_BASE / "ball_wasb" / f"ball_{stem}.csv")))
    if len(ball) < 20:
        return None
    frames = np.array([int(r["frame"]) for r in ball])
    xs = np.array([float(r["x_stab"]) for r in ball])
    ys = np.array([float(r["y_stab"]) for r in ball])
    # wandering-camera correction: clip-stabilized coords -> fit-camera
    # coords via the probe-measured clip offset, BEFORE the projection
    odx, ody = OFFSETS.get(stem, (0.0, 0.0))
    pts = np.stack([xs - odx, ys - ody], axis=1).reshape(-1, 1, 2).astype(np.float32)
    court = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    cyc = court[:, 1]
    cxc = court[:, 0]

    s = serves.get(stem, {})
    server = s.get("server", "?")
    serve_frame = int(s["serve_frame"]) if server != "?" else None

    if EVENTS == "v5":
        events, serve_frame, v5info = events_v5.detect_events(
            frames, ys, cyc, fps, serve_frame,
            server if server in ("near", "far") else None)
        if serve_frame is None:
            server = "?"        # spine-refuted serve call (see events_v5)
    else:
        events = detect_events(frames, ys, cyc, fps, serve_frame)

    # box hygiene (player_boxes.py): court-half plausibility, teleport
    # rejection, short-gap interp — the audit-measured letter sink lives
    # in the raw boxes
    players = player_boxes.load(
        OUT_BASE / "players" / f"players_{stem}.csv", Hm,
        OFFSETS.get(stem, (0.0, 0.0)))

    ball_exact = dict(zip(frames.tolist(), zip(xs.tolist(), ys.tolist())))

    if serve_frame is not None:
        if EVENTS != "v5":      # v5's skeleton already handled pre-serve
            events = [e for e in events if e["frame"] >= serve_frame - 3]
    else:
        first_strike = next((e for e in events if e["kind"] == "hit"), None)
        if first_strike:
            events = [e for e in events if e["frame"] >= first_strike["frame"]]

    hits = [e for e in events if e["kind"] == "hit"]
    bounces = [e for e in events if e["kind"] == "bounce"]

    shots = []
    if serve_frame is not None and (not hits or hits[0]["frame"] > serve_frame + 14):
        # detector missed serve contact; the gated serve call stands in
        shots.append({"frame": serve_frame, "vcy_after": None,
                      "is_serve": True, "synth": True, "anchor": True})
    for k, h in enumerate(hits):
        is_serve = (not shots and serve_frame is not None
                    and abs(h["frame"] - serve_frame) <= 14)
        shots.append({"frame": h["frame"], "vcy_after": h["vcy_after"],
                      "is_serve": is_serve, "synth": False,
                      "anchor": is_serve})
    if not shots:
        return None

    # coda truncation: drop shots after the point already ended (the
    # in-plateau dead-ball coda; rules + receipts in the constants block).
    # Under v5 the crossing chain already excluded the coda structurally,
    # so the post-hoc pass stands down and reports the detector's cuts.
    if EVENTS == "v5":
        n_coda, coda_why = v5info["n_dropped"], v5info["why"]
    else:
        xruns = net_crossings(frames, cyc, fps)
        shots, n_coda, coda_why = truncate_coda(shots, xruns, fps)

    # landings first (striker-independent): first bounce before the next shot
    for k, sh in enumerate(shots):
        nxt = shots[k + 1]["frame"] if k + 1 < len(shots) else 10 ** 9
        landing = next((b for b in bounces if sh["frame"] < b["frame"] < nxt), None)
        if landing is None and sh["is_serve"] and EVENTS == "v5":
            # near-half serve-landing fill: a far server's serve bounces
            # in the NEAR half, where the collapse detector is blind by
            # construction. v4 charts hid this by handing the serve the
            # RETURN's far-half landing (v5 places the return at its true
            # launch, so the theft stopped and the zones went '?'). A
            # ball flying INTO the camera never reverses image-y at the
            # bounce — it decelerates: the bounce is the first big
            # descending-velocity KINK in the near half, and the
            # projection is honest exactly there (ending-fill physics).
            iy_sv = moving_average(ys, SMOOTH)
            viy_sv = np.gradient(iy_sv, frames.astype(float))
            swing_sv = SWING_NEAR_30 * 30.0 / fps
            for i in range(2, len(frames) - 2):
                if not sh["frame"] + 2 < frames[i] < nxt:
                    continue
                if frames[i + 1] - frames[i - 1] > 6:
                    continue      # kink straddling a track hole
                if (viy_sv[i - 1] > 0
                        and viy_sv[i - 1] - viy_sv[i + 1] >= swing_sv
                        and NET_Y + 1 < cyc[i] < L_C):
                    landing = {"frame": int(frames[i]),
                               "court_y": float(cyc[i])}
                    break
        lx = None
        if landing is not None:
            li = np.searchsorted(frames, landing["frame"])
            lx = float(np.median(cxc[max(0, li - 2):li + 3]))
        sh["landing_x"] = round(lx, 2) if lx is not None else None
        sh["landing_y"] = round(landing["court_y"], 1) if landing else None
        if sh["is_serve"]:
            # deuce/ad can be uncommitted (ball-called serves whose
            # stance was unreadable); no side, no zone claim
            sh["zone"] = (serve_zone(lx, s["side"] == "deuce", server)
                          if lx is not None
                          and s.get("side") in ("deuce", "ad") else "?")
        # rally-shot directions come from shot_direction.annotate() below

    # letter-gate height reference: the observed box height under-gates
    # a legs-only partial blob (h 30 px -> gate 48 px; the ball meets the
    # racquet ~a body-height above the legs). The clip's own full-body
    # height per side is the 75th percentile of its boxes — partials
    # drag the median, not the upper quartile. The gate uses
    # max(observed, typical): rogue boxes hundreds of px away stay
    # refused, real near-misses commit.
    h_typ = {}
    for side in ("near", "far"):
        hs = [float(p[side]["h"]) * 720
              for p in players.values() if side in p]
        h_typ[side] = float(np.percentile(hs, 75)) if hs else 0.0

    # per-side contact search: nearest the ball comes to EACH player's box
    # inside the shot's window — feeds both the touch votes and the letter
    for k, sh in enumerate(shots):
        fi = sh["frame"]
        reach = REFINE
        if k > 0:
            reach = min(reach, (fi - shots[k - 1]["frame"]) // 2)
        if k + 1 < len(shots):
            reach = min(reach, (shots[k + 1]["frame"] - fi) // 2)
        for side in ("near", "far"):
            best = None
            for d in range(max(reach, 1) + 1):
                for f2 in ({fi} if d == 0 else {fi - d, fi + d}):
                    if f2 not in ball_exact:
                        continue
                    p = players.get(f2, {}).get(side)
                    if p is None:
                        continue
                    bx, by = ball_exact[f2]
                    dist = box_dist(bx, by, p)
                    if best is None or dist < best[0]:
                        best = (dist, f2, bx, float(p["h"]) * 720,
                                float(p["cx"]) * 1280)
            sh[f"touch_{side}"] = best

    holes = [(int(a), int(b)) for a, b in zip(frames, frames[1:])
             if b - a > HOLE_FRAMES]
    lock = server if (server != "?" and shots[0].get("anchor")) else None
    conflicts = assign_strikers(shots, server if server != "?" else "near",
                                holes, lock)

    # the chart can outvote the serve detector's end call; when it does,
    # the detector's deuce/ad side is suspect too, so the serve zone
    # degrades to '?'
    server_used = server
    if shots[0]["is_serve"]:
        server_used = shots[0]["striker"]
        if server != "?" and server_used != server:
            shots[0]["zone"] = "?"

    for sh in shots:
        side = sh["striker"]
        best = sh.get(f"touch_{side}")
        sh["contact_frame"] = best[1] if best else sh["frame"]
        sh["contact_dist_px"] = round(best[0], 1) if best else None
        gate = LETTER_GATE(max(best[3], h_typ[side])) if best else None
        if best and not sh["synth"] and best[0] <= gate:
            dxp = best[2] - best[4]
            right = dxp > 0 if side == "near" else dxp < 0
            forehand = right != LEFTY[side]
            sh["letter"] = "f" if forehand else "b"
        else:
            sh["letter"] = "?"

    # rally-shot direction digits (MCP 1/2/3), BOTH halves: the
    # receiver-mirrored mapping + the measured signal ladder (near-half
    # landing > receiver contact > crossing+slope > far-half landing) —
    # shot_direction.py, semantics calibrated on all 4 matches, signal
    # quality/precedence tuned on t3 only
    shot_direction.annotate(shots, frames, cyc, cxc, fps)

    # ending v1 — observable evidence only. The last shot's own landing
    # (far-half only, by construction) codes out-deep/-wide; a ball track
    # that dies at the net right after the last hit codes a net error.
    # Winner vs forced vs unforced is HUMAN judgment the pipeline does not
    # attempt: '@' means "error, attribution not judged" and eval compares
    # the ending TYPE only.
    ending = "?"
    last = shots[-1]
    if last.get("landing_y") is not None:
        ly, lx = last["landing_y"], last.get("landing_x")
        deep = ly < -OUT_MARGIN
        wide = lx is not None and not (
            1.372 - OUT_MARGIN <= lx <= W_C - 1.372 + OUT_MARGIN)
        ending = ("x@" if deep and wide else "d@" if deep
                  else "w@" if wide else "*")   # in, and nothing came back
    else:
        li = int(np.searchsorted(frames, last["frame"]))
        tail_y, tail_f = cyc[li:], frames[li:]
        if (len(tail_y) >= 3
                and abs(float(tail_y[-1]) - NET_Y) < NET_ZONE_M
                and tail_f[-1] - last["frame"] <= 1.2 * fps):
            ending = "n@"
    if ending == "?":
        # near-half ending fill (see constants block): first image-y
        # V-cusp in the near half within flight time of the final shot
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
                wide = not (1.372 - OUT_MARGIN <= bx <= W_C - 1.372 + OUT_MARGIN)
                ending = ("x@" if deep and wide else "d@" if deep
                          else "w@" if wide else "*")
                break

    mcp = ""
    for sh in shots:
        mcp += f"s{sh['zone']}" if sh["is_serve"] else f"{sh['letter']}{sh['zone']}"
    mcp += ending

    return {"clip": stem, "fps": fps, "server": server,
            "server_used": server_used, "side": s.get("side", ""),
            "n_hits": len(hits), "n_bounces": len(bounces),
            "conflicts": conflicts, "n_holes": len(holes),
            "n_coda": n_coda, "coda_why": coda_why,
            "ending": ending, "mcp": mcp, "shots": shots}


OFFSETS = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    args = parser.parse_args()
    for r in csv.DictReader(open(OUT_BASE / "clip_offsets.csv")):
        OFFSETS[r["clip"]] = (float(r["dx"]), float(r["dy"]))

    Hm = np.load(ROOT / "outputs/t4/H_img_to_court.npy")
    serves = {r["clip"]: r for r in csv.DictReader(open(OUT_BASE / "serves.csv"))}
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    fields = ["shot", "frame", "contact_frame", "contact_dist_px", "is_serve",
              "synth", "striker", "letter", "zone", "landing_x", "landing_y",
              "vcy_after"]
    rows = []
    for stem in args.clips:
        r = chart_clip(stem, Hm, serves)
        if r is None:
            print(f"{stem}: track too thin or no shots, skipped")
            continue
        with open(CHART_DIR / f"chart2_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for k, sh in enumerate(r["shots"]):
                wr.writerow({"shot": k + 1, **{k2: sh.get(k2) for k2 in fields[1:]}})
        rows.append({k: r[k] for k in
                     ("clip", "server", "server_used", "side", "n_hits",
                      "n_bounces", "n_holes", "conflicts", "n_coda",
                      "coda_why", "ending", "mcp")})
        strikers = "".join(sh["striker"][0].upper() for sh in r["shots"])
        flip = (" SERVER-OVERRIDE" if r["server"] not in ("?", r["server_used"])
                else "")
        coda = f" coda-{r['n_coda']}[{r['coda_why']}]" if r["n_coda"] else ""
        print(f"{r['clip']}: server={r['server']}->({r['server_used']}) "
              f"strikers={strikers} holes={r['n_holes']} "
              f"conflicts={r['conflicts']}{flip}{coda}  MCP: {r['mcp']}")

    with open(CHART_DIR / "match_chart_v2.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {CHART_DIR / 'match_chart_v2.csv'} ({len(rows)} points)")


if __name__ == "__main__":
    main()
