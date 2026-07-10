"""Serve detection v3 — the ball adjudicates where it can, the players
corroborate; on feeds where the ball can't, the sliding stance stands.

Lifted from t3_serve_detect.py / t4_serve_detect.py (frozen outputs).
Two variants, selected per match (serve_detect.variant in the YAML):

  ball    (t3) primary signal is the WASB ball track: the serve is the
          first sustained net crossing — a monotone run of court-y that
          spans the net fast. Its DIRECTION gives the server end (cy
          falling = near serve), its start frame the serve frame.
          Measured against MCP truth via changeover parity BEFORE
          wiring: 48/56 (86%) on clips with a crossing. Deuce/ad still
          needs the ball-voted side's stance; when the stance is
          unreadable the side is left blank and the chart degrades the
          serve zone to '?' rather than guess. No-crossing clips fall
          back to the stance path below.

  stance  (t4) the players are the signal: search the whole clip for
          each side's first 1-s window passing coverage+center+baseline
          (the sliding settle — point-boundary clips start at the
          score-bug plateau, and the between-points dead air lives
          INSIDE it), then want the toss within 4 s of settling.
          Two-candidate ties break to the side that SETTLED first
          (21/25 against parity truth), simultaneous settles to the
          stronger toss. FOR THE RECORD: the ball does NOT adjudicate
          on grass — white ball, white lines, 72% coverage read 26/49,
          a coin flip, so it is not wired in there.

Per-match knob: center_tol_m (clay servers stand WIDE — Ruud serves
from x=1.4, 4.1 m off the mark; t3 uses 4.3 vs the default 2.0).

The t1/t2 serves.csv artifacts predate v3 and stay frozen; rerunning
those trees through this module uses the v3 stance recipe (the forward
path — see the README divergence note).

fps is read from the clip, never assumed.

Usage:
    uv run python -m courtvision serve t3
"""

import csv

import cv2
import numpy as np

from .court import W_C, L_C, CENTER_X, NET_Y

SMOOTH = 5
COVERAGE_MIN = 0.60
BASELINE_TOL = {"far": (-3.5, 1.0), "near": (L_C - 1.0, L_C + 3.5)}
TOSS_RATIO = 1.12
# ball-launch constants, read off the 56 measured t3 crossings:
LAUNCH_WIN_S = 0.6   # a serve crosses the net well inside 0.6 s
LAUNCH_SPAN_M = 4.0  # min court-y span of the run (rally floats are slower)
LAUNCH_MONO = 0.75   # fraction of steps moving the crossing's way
LAUNCH_GAP = 3       # max tracked-frame gap inside a run
STANCE_PRE_S = 1.2   # stance window ending at the launch (server stands still)
STANCE_COV = 0.50    # ...needs half that window tracked to trust the x
# The stance read wants the X (deuce/ad); its y band only rejects
# impostors, so it is looser than the serve gate's BASELINE_TOL: the
# near server idles at y 27.9-28.9 pre-toss and the far one wanders to
# y ~3; the tape ghost sits at y 6.1-8.4, which the far cap stays clear of.
STANCE_TOL = {"far": (-4.5, 3.0), "near": (L_C - 1.0, L_C + 5.5)}
# sliding settle (stance variant + ball-variant fallback):
SETTLE_WIN_S = 1.0
SETTLE_STEP_S = 0.25
SERVE_AFTER_SETTLE_S = 4.0

FIELDS = ["clip", "server", "server_x_m", "margin_m", "side",
          "serve_frame", "serve_s", "toss_h_norm", "launch_cy",
          "src", "reason"]


def find_launch(frames, cy, fps):
    """First sustained net crossing: (end, start_frame, cy0) or None."""
    max_win = LAUNCH_WIN_S * fps
    for i in range(len(frames) - 1):
        j = i
        while (j + 1 < len(frames) and frames[j + 1] - frames[j] <= LAUNCH_GAP
               and frames[j + 1] - frames[i] <= max_win):
            j += 1
        if j == i:
            continue
        seg = cy[i:j + 1]
        if (seg[0] - NET_Y) * (seg[-1] - NET_Y) < 0 and abs(seg[-1] - seg[0]) > LAUNCH_SPAN_M:
            d = np.diff(seg)
            mono = float(np.mean(np.sign(d) == np.sign(seg[-1] - seg[0])))
            if mono >= LAUNCH_MONO:
                end = "near" if seg[-1] < seg[0] else "far"
                return end, int(frames[i]), float(cy[i])
    return None


def toss_peak(per_side, lo, hi):
    """Smoothed-height peak ratio in [lo, hi): (frame, ratio) or None.
    Keeps the t2 dissolve lesson: blobs taller than 2x the series median
    are transition junk, dropped before peak-finding."""
    hs = [(fi, float(per_side[fi]["h"])) for fi in sorted(per_side) if lo <= fi < hi]
    if len(hs) < SMOOTH:
        return None
    h_med = float(np.median([h for _, h in hs]))
    hs = [(fi, h) for fi, h in hs if h <= 2.0 * h_med]
    if len(hs) < SMOOTH:
        return None
    fis = [f for f, _ in hs]
    hh = np.array([h for _, h in hs])
    pad = np.pad(hh, SMOOTH // 2, mode="edge")
    hh = np.array([np.median(pad[j:j + SMOOTH]) for j in range(len(hh))])
    peak = int(np.argmax(hh))
    return fis[peak], float(hh[peak]) / float(np.median(hh))


def detect_match(cfg):
    """Detect the serve for every clip with a players CSV; write
    serves.csv into the match's output tree."""
    variant = cfg.serve_detect.get("variant", "stance")
    center_tol = float(cfg.serve_detect.get("center_tol_m", 2.0))
    Hm = np.load(cfg.homography)
    offsets = cfg.load_offsets()
    rows_out = []
    tracks = sorted(cfg.players_dir.glob("players_*.csv"))
    for tpath in tracks:
        stem = tpath.stem.replace("players_", "")
        cap = cv2.VideoCapture(str(cfg.clip_path(stem)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        per = {"near": {}, "far": {}}
        with open(tpath) as f:
            for r in csv.DictReader(f):
                per[r["player"]][int(r["frame"])] = r

        odx, ody = offsets.get(stem, (0.0, 0.0))

        def court_xy(r):
            pt = np.float32([[float(r["foot_x"]) - odx,
                              float(r["foot_y"]) - ody]]).reshape(-1, 1, 2)
            xy = cv2.perspectiveTransform(pt, Hm).reshape(2)
            return float(xy[0]), float(xy[1])

        def stance(side, lo, hi):
            """Median court x/y of a side over [lo, hi) if tracked and at
            its baseline; None when the track is absent or an impostor."""
            rows = [per[side][fi] for fi in sorted(per[side]) if lo <= fi < hi]
            if len(rows) < STANCE_COV * (hi - lo):
                return None
            xys = [court_xy(r) for r in rows]
            mx = float(np.median([p[0] for p in xys]))
            my = float(np.median([p[1] for p in xys]))
            blo, bhi = STANCE_TOL[side]
            if not (blo <= my <= bhi) or abs(mx - CENTER_X) > center_tol:
                return None
            return mx, my

        # ---- primary: the ball adjudicates (ball variant only) ----
        launch = None
        if variant == "ball":
            bpath = cfg.ball_dir / f"ball_{stem}.csv"
            if bpath.exists():
                ball = list(csv.DictReader(open(bpath)))
                if len(ball) >= 10:
                    bfr = np.array([int(r["frame"]) for r in ball])
                    pts = np.stack([[float(r["x_stab"]) - odx for r in ball],
                                    [float(r["y_stab"]) - ody for r in ball]],
                                   axis=1).reshape(-1, 1, 2).astype(np.float32)
                    bcy = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)[:, 1]
                    launch = find_launch(bfr, bcy, fps)

        if launch is not None:
            server, f0, cy0 = launch
            st = stance(server, max(0, int(f0 - STANCE_PRE_S * fps)), f0 + 1)
            side = ""
            sx = None
            if st is not None:
                sx = st[0]
                deuce = sx > CENTER_X if server == "near" else sx < CENTER_X
                side = "deuce" if deuce else "ad"
            rows_out.append({
                "clip": stem, "server": server,
                "server_x_m": round(sx, 2) if sx is not None else "",
                "margin_m": round(abs(sx - CENTER_X), 2) if sx is not None else "",
                "side": side, "serve_frame": f0,
                "serve_s": round(f0 / fps, 2),
                "launch_cy": round(cy0, 1), "src": "ball",
                "reason": "" if side else "stance_unreadable_no_side",
            })
            continue

        # ---- player gates with a sliding settle window ----
        win = int(SETTLE_WIN_S * fps)
        step = max(1, int(SETTLE_STEP_S * fps))
        candidates = []
        gate_log = []
        for side in ("near", "far"):
            settle = None
            for start in range(0, max(1, nfr - win), step):
                rows = [per[side][fi] for fi in sorted(per[side])
                        if start <= fi < start + win]
                if len(rows) / win < COVERAGE_MIN:
                    continue
                xys = [court_xy(r) for r in rows]
                mx = float(np.median([p[0] for p in xys]))
                my = float(np.median([p[1] for p in xys]))
                lo, hi = BASELINE_TOL[side]
                if abs(mx - CENTER_X) <= center_tol and lo <= my <= hi:
                    settle = (start, mx, my)
                    break
            if settle is None:
                gate_log.append(f"{side}:never-settles")
                continue
            s0, mx, my = settle
            tp = toss_peak(per[side], s0, s0 + int(SERVE_AFTER_SETTLE_S * fps))
            if tp is None or tp[1] < TOSS_RATIO:
                gate_log.append(f"{side}:no-toss"
                                + (f" x{tp[1]:.2f}" if tp else ""))
                continue
            candidates.append({"side": side, "mx": mx, "my": my,
                               "settle": s0,
                               "serve_frame": tp[0], "toss_ratio": tp[1]})

        if not candidates:
            rows_out.append({"clip": stem, "server": "?", "src": "none",
                             "reason": "no_confident_serve: " + "; ".join(gate_log)})
            continue
        # earlier settle wins, stronger toss breaks the remaining tie
        best = min(candidates, key=lambda c: (c["settle"], -c["toss_ratio"]))
        server = best["side"]
        sx = best["mx"]
        # deuce = server's right of the center mark: court x > center for a
        # near server (faces -y), x < center for a far server (faces +y)
        deuce = sx > CENTER_X if server == "near" else sx < CENTER_X
        rows_out.append({
            "clip": stem, "server": server,
            "server_x_m": round(sx, 2),
            "margin_m": round(abs(sx - CENTER_X), 2),
            "side": "deuce" if deuce else "ad",
            "serve_frame": best["serve_frame"],
            "serve_s": round(best["serve_frame"] / fps, 2),
            "toss_h_norm": round(best["toss_ratio"], 3),
            "src": "players",
            "reason": "both_ends_passed" if len(candidates) == 2 else "",
        })

    with open(cfg.serves, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for r in rows_out:
            wr.writerow({k: r.get(k, "") for k in FIELDS})

    called = [r for r in rows_out if r["server"] != "?"]
    print(f"{len(called)}/{len(rows_out)} clips got a server + serve frame")
    n_ball = sum(1 for r in called if r["src"] == "ball")
    if variant == "ball":
        print(f"src: ball {n_ball}, players {len(called) - n_ball}")
    n_near = sum(1 for r in called if r["server"] == "near")
    print(f"servers: near {n_near}, far {len(called) - n_near}")
    no_side = [r["clip"] for r in called if not r.get("side")]
    if no_side:
        print(f"end committed but no deuce/ad (stance unreadable): {no_side}")
    print(f"-> {cfg.serves}")
