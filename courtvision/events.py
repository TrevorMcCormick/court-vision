"""EVENT DETECTOR V5 — the crossing skeleton.

The M2-era detector classified every image-y cusp as hit-or-bounce and
counted the survivors; rally length ±1 sat around 50% on t3/t4 and every
letter, zone, and ending inherited the miscount. Diagnosis on the dense
WASB tracks (2026-07-10, receipts in the LOG) split the misses into:

  OVER   dead-ball codas charted as shots — on BOTH adversarial feeds
         (t3 has 4 of them; the old coda pass was t4-only) — plus
         duplicate cusps on slow clay loops.
  UNDER  far-end hits the cusp detector never sees (the swing gate is
         tuned for near-end geometry; a far-end hit at 250+ px is a
         1-px/frame wiggle), and hits hidden in track holes.

What generalizes is the t4 coda entry's insight, inverted into the
DESIGN instead of a post-hoc truncation: a live rally sends the ball
across the net EVERY shot, so the track's net-crossing sequence is the
point's spine. Each crossing was launched by exactly one hit. So:

  1. crossings = sustained monotone court-y runs through the net
     (gates measured for the coda pass, inherited verbatim).
  2. the live chain = crossings from the serve onward, split where
     consecutive crossings are > DEAD_GAP_S apart (a live rally cannot
     go 3 s without crossing; codas, swats, and let-replays can) — but
     ONLY when the ball track actually covers the gap: a gap that is
     mostly track hole says "we didn't see", not "nothing crossed"
     (clay's recall pays; t3_point_29's mid-rally hole must bridge
     while t4's tracked dead-ball codas must cut).
  3. one hit per chain crossing, located by the strongest evidence in
     a window around the crossing's start: the biggest image-y cusp
     (envelope-valid, ANY speed class — far-end hits often classify
     as "bounce" by post-hit vcy, which is exactly the old detector's
     miss), else synthesized at the crossing start.
  4. the serve: if the first chain crossing flies server->receiver and
     starts within SERVE_SNAP_S of the gated serve frame, its launch IS
     the serve and snaps into the twin's serve window. A serve call
     that lands AFTER the whole crossing story (t4_point_15 called
     f406 of a 441-frame clip) is refuted by the spine and returned as
     None — the twins fall back to their no-serve path.
  5. the trailing shot: a net error / failed return never crosses; a
     hit cusp within RECV_S of the last crossing's end is kept.
  6. hidden hits: a partition between consecutive crossings that runs
     longer than any single flight (> EXTRA_PART_S) must contain hits
     whose crossings the track lost; envelope-valid v4 HITS deeper
     than EXTRA_SEP_S inside such a partition are re-inserted. Only
     BETWEEN chain crossings — after the last crossing lives the
     trailing-shot rule alone (the coda starts there).

Bounces are untouched v4 machinery (far-half collapse + weak cusps) —
landings, zones, and endings read them exactly as before. If the track
has NO crossings at all there is no spine, and the v4 events stand
unmodified (thin/short tracks, net-cord points).

TRAIN/TEST: the new constants below (LAUNCH_*, SERVE_SNAP_S,
EXTRA_SEP_S, SUSPECT_*) were tuned on the t3 tree + dev-reel spot
checks ONLY; t1/t2/t4 are scored untouched. CROSS_* / DEAD_GAP_S /
RECV_S are inherited as frozen from the t4 coda pass; the cusp,
collapse, envelope, and classification constants are the frozen M2/
freeze-#3 values shared by every t*w twin.

Lifted verbatim from experiments/events_v5.py (2026-07-10) for the
courtvision package; the experiment script stays frozen as history.

Usage (from courtvision.chart):

    from courtvision import events
    evs, serve_frame, info = events.detect_events(
        frames, iy, cyc, fps, serve_frame, server)
"""

import numpy as np

# ---- frozen M2 / freeze-#3 constants (identical in all t*w twins) ----
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
HIT_CY_MIN = -12.0
HIT_CY_MAX = L_C + 2.2
SERVE_WIN = 14

# ---- crossing gates, inherited frozen from the t4 coda pass ----
CROSS_SPAN_MIN_M = 5.0
CROSS_SPAN_MAX_M = 40.0
CROSS_SPEED_MIN, CROSS_SPEED_MAX = 4.0, 90.0
CROSS_MIN_SAMPLES = 4
CROSS_MAX_STEP_M = 6.0
CROSS_MAX_FRAME_GAP = 5
DEAD_GAP_S = 3.0              # s; crossing-to-crossing gap that ends a chain
RECV_S = 1.2                  # s; last crossing end -> trailing shot, max lag

# ---- v5 constants, tuned on t3 + dev-reel spot checks ONLY ----
LAUNCH_BACK_S = 0.50          # s before a crossing start the launch cusp may sit
LAUNCH_FWD_S = 0.20           # s after (smoothing can push the cusp past it)
SERVE_SNAP_S = 1.00           # s; serve call -> first crossing start, max lag
SERVE_PRE_S = 0.30            # s; crossings starting earlier than serve-this drop
GAP_COV = 0.5                 # dead gaps must be at least half TRACKED to cut
EXTRA_PART_S = 2.0            # s; partitions longer than any flight hide hits
EXTRA_SEP_S = 0.80            # s; hidden hit must sit this deep inside one
SUSPECT_AFTER_S = 1.0         # s; serve call this far after the LAST crossing start
SUSPECT_MIN_CROSS = 2         # ... with at least this many crossings = refuted


def moving_average(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _v4_events(frames, iy, cyc, fps, serve_frame):
    """The frozen M2-v4 cusp/collapse detector, verbatim (shared source of
    cusp candidates and of ALL bounce events)."""
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
    return events, vcy


def net_crossings(frames, cyc, fps, weak=False):
    """Sustained monotone court-y runs that pass the net at rally speed.
    Byte-identical gates to the t4 coda pass. [(start_f, end_f, dir)] with
    dir = +1 toward the near baseline (cy increasing), -1 toward the far.

    weak=True relaxes the span cap and sample floor (keeps the teleport
    step guard): deep-lob flights project past 40 m and short tracked
    stubs still pass the net — recall the strict gates trade away. Weak
    runs only ever supplement hits INSIDE an already-live chain."""
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
        min_samples = 3 if weak else CROSS_MIN_SAMPLES
        span_max = 10 ** 9 if weak else CROSS_SPAN_MAX_M
        if (j - i + 1 >= min_samples
                and CROSS_SPAN_MIN_M <= span <= span_max
                and min(lo, hi) < NET_Y - 1 and max(lo, hi) > NET_Y + 1
                and dur > 0
                and CROSS_SPEED_MIN <= span / dur
                and (weak or span / dur <= CROSS_SPEED_MAX)
                and steps.max() <= CROSS_MAX_STEP_M):
            out.append((int(frames[i]), int(frames[j]), int(d)))
        i = j
    return out


def detect_events(frames, iy, cyc, fps, serve_frame=None, server=None):
    """V5: derive the hit list structurally from the crossing skeleton;
    keep v4's bounces untouched. Returns (events, serve_frame, info) —
    serve_frame comes back None when the spine refutes the serve call."""
    v4, vcy = _v4_events(frames, iy, cyc, fps, serve_frame)
    xr = net_crossings(frames, cyc, fps)
    info = {"crossings": [(a, b) for a, b, _ in xr], "n_dropped": 0,
            "why": "", "serve_suspect": False}
    if not xr:
        return v4, serve_frame, info      # no spine -> v4 stands

    min_gap = max(3, int(round(MIN_GAP_S * fps)))

    # serve refutation: a call after the whole crossing story charts nothing
    if (serve_frame is not None and len(xr) >= SUSPECT_MIN_CROSS
            and serve_frame - xr[-1][0] > SUSPECT_AFTER_S * fps):
        serve_frame = None
        info["serve_suspect"] = True

    # live chain: crossings from the serve onward, split at dead gaps
    if serve_frame is not None:
        live = [c for c in xr if c[0] >= serve_frame - SERVE_PRE_S * fps]
        if not live:
            live = xr[:]
    else:
        live = xr[:]
    cs_net = moving_average(cyc, 3) - NET_Y

    def gap_dead(f0, f1):
        """A gap is DEAD evidence only if the track observed it (a hole
        says "didn't see", not "nothing crossed") and the ball never
        passed the net line inside it (deep-lob flights project past
        the 40 m span cap and fail the run gates while being perfectly
        alive — t3_point_29's mid-rally excursions taught this)."""
        i0, i1 = np.searchsorted(frames, f0), np.searchsorted(frames, f1)
        if f1 <= f0 or (i1 - i0) / (f1 - f0) < GAP_COV:
            return False
        seg = np.sign(cs_net[i0:i1])
        return not np.any(seg[:-1] * seg[1:] < 0)

    chains, cur = [], [live[0]]
    for c in live[1:]:
        if c[0] - cur[-1][1] > DEAD_GAP_S * fps and gap_dead(cur[-1][1], c[0]):
            chains.append(cur)
            cur = [c]
        else:
            cur.append(c)
    chains.append(cur)
    chain = chains[0] if serve_frame is not None else max(chains, key=len)
    n_cut = len(live) - len(chain)
    if n_cut:
        info["why"] = f"chain-cut({n_cut}xr)"

    # recall supplement: weak net passes fully inside an over-long
    # inter-crossing partition are flights the strict gates dropped —
    # splice them into the spine (weak evidence never EXTENDS the
    # chain PAST its end, only fills it). The PRE-chain region counts
    # as a partition too: when the track's first strict crossing comes
    # late, the rally's opening shots live before it (t1's night
    # highlights amputated their fronts without this). The POST-chain
    # region never does — that's where the coda lives.
    def long_partitions():
        parts = [(chain[k][1], chain[k + 1][0]) for k in range(len(chain) - 1)
                 if chain[k + 1][0] - chain[k][1] > EXTRA_PART_S * fps]
        front0 = serve_frame if serve_frame is not None else int(frames[0]) - 1
        if chain[0][0] - front0 > EXTRA_PART_S * fps:
            parts.insert(0, (front0, chain[0][0]))
        return parts

    long_parts = long_partitions()
    spliced = []
    for wa, wb, wd in net_crossings(frames, cyc, fps, weak=True):
        if any(p0 < wa and wb < p1 for p0, p1 in long_parts):
            spliced.append((wa, wb, wd))
    if spliced:
        chain = sorted(chain + spliced)
        info["why"] = (info["why"] + " " if info["why"] else "") + \
            f"spliced({len(spliced)}xr)"

    # candidate launch cusps: every envelope-valid v4 cusp, ANY speed class
    cand = [e for e in v4 if e["signal"] == "cusp"]

    def synth_at(f):
        i = int(np.clip(np.searchsorted(frames, f), 1, len(frames) - 2))
        ma = float(np.median(vcy[i + 1:i + 1 + WIN]))
        return {"idx": i, "frame": int(frames[i]), "kind": "hit",
                "signal": "xr-synth", "court_y": float(cyc[i]),
                "vcy_after": ma}

    hits, used = [], set()
    for k, (a, b, d) in enumerate(chain):
        w0, w1 = a - LAUNCH_BACK_S * fps, a + LAUNCH_FWD_S * fps
        pool = [e for e in cand if w0 <= e["frame"] <= w1
                and id(e) not in used
                and (not hits or e["frame"] >= hits[-1]["frame"] + min_gap)]
        # prefer hit-classified cusps; a bounce-class cusp in the window
        # is often the PREVIOUS shot's landing (it stole six t3 serve
        # zones before this guard), so it only stands in when no
        # hit-class cusp exists (far-end hits classify as bounce — the
        # very miss the skeleton exists to recover)
        strong = [e for e in pool if e["kind"] == "hit"]
        if pool:
            e = min(strong or pool, key=lambda e: abs(e["frame"] - a))
            used.add(id(e))
            hits.append(dict(e, kind="hit"))
        else:
            if hits and int(frames[min(np.searchsorted(frames, a), len(frames) - 1)]) \
                    < hits[-1]["frame"] + min_gap:
                continue          # shared-endpoint run already owned by last hit
            hits.append(synth_at(a))
        # serve snap: first chain crossing flying server->receiver, close
        # to the gated serve frame -> its launch IS the serve
        if (k == 0 and serve_frame is not None and server in ("near", "far")
                and d == (1 if server == "far" else -1)
                and 0 <= a - serve_frame <= SERVE_SNAP_S * fps
                and hits and hits[0]["frame"] - serve_frame > SERVE_WIN):
            hits[0] = synth_at(serve_frame)
            hits[0]["signal"] = "xr-serve"

    # hidden hits: strong v4 hits buried deep inside an over-long
    # partition (a crossing the track lost still had a launch). Only
    # in pre-chain/inter-crossing partitions — past the last crossing
    # is trailing/coda land. Recomputed: the splice may have filled some.
    starts = [c[0] for c in chain]
    long_parts = long_partitions()
    extras = []
    for e in v4:
        if e["kind"] != "hit" or e["signal"] != "cusp":
            continue
        if not any(p0 < e["frame"] < p1 for p0, p1 in long_parts):
            continue
        if any(abs(e["frame"] - h["frame"]) < min_gap for h in hits + extras):
            continue
        if all(abs(e["frame"] - s) > EXTRA_SEP_S * fps for s in starts):
            extras.append(e)
    hits += extras

    # trailing shot: the receiver's no-cross shot (net error / failed return)
    end_last = chain[-1][1]
    tail = [e for e in cand if id(e) not in used and e["kind"] == "hit"
            and end_last < e["frame"] <= end_last + RECV_S * fps
            and all(abs(e["frame"] - h["frame"]) >= min_gap for h in hits)]
    if tail:
        hits.append(dict(tail[0], kind="hit"))

    hits.sort(key=lambda e: e["frame"])
    n_v4_hits = sum(1 for e in v4 if e["kind"] == "hit")
    info["n_dropped"] = max(n_v4_hits - len(hits), 0)

    hit_frames = {h["frame"] for h in hits}
    bounces = [e for e in v4 if e["kind"] == "bounce"
               and e["frame"] not in hit_frames]
    events = sorted(hits + bounces, key=lambda e: e["frame"])
    return events, serve_frame, info
