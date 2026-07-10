"""T4 — serve detection v3: same gates, but the window slides.

v2 (fixed early window at frame 0, toss in the first 4 s) committed
37/49 and was RIGHT on 33 of them — the Wimbledon feed's players are
the reliable signal here. Its misses were all one failure mode:

  DEAD-AIR STANCE — the point-boundary clips start at the score-bug
  plateau boundary, and Wimbledon's long between-points dead time
  lives INSIDE the plateau. In 10 of the 12 no-call clips both
  players DO settle into a legal serve/receive stance — just 0.5-9.6
  seconds in, where a window nailed to frame 0 never looks (and a
  toss at 4.7-6.6 s sits outside the old fixed 4-s toss window).

v3 searches the whole clip for each side's first 1-s window passing
coverage+center+baseline, then wants the toss within 4 s of settling.
Two-candidate ties break to the side that SETTLED first (the server's
stance forms while the receiver is still wandering — 21/25 dual-
candidate clips against parity truth, vs 13/25 for v2's toward-the-
center-mark rule), and simultaneous settles to the stronger toss
(3/4; the receiver rarely fakes a real stretch).

FOR THE RECORD — the ball does NOT adjudicate on grass: the t3 v3
primary (first sustained net crossing of the WASB track) was measured
against parity truth here first and scored 26/49 (53%, a coin flip).
White ball, white lines, 72% coverage, and dead-air ball-kid throws
bury the launch signal, so it is not wired in on t4.

fps is read from the clip, never assumed.

Usage:
    uv run experiments/t4_serve_detect.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "t4"
TRACK_DIR = OUT_BASE / "players"
ROOT = Path(__file__).resolve().parent.parent

W_C, L_C = 10.97, 23.77
CENTER_X = W_C / 2
SMOOTH = 5
COVERAGE_MIN = 0.60
CENTER_TOL_M = 2.0
BASELINE_TOL = {"far": (-3.5, 1.0), "near": (L_C - 1.0, L_C + 3.5)}
TOSS_RATIO = 1.12
# v3 — sliding settle:
SETTLE_WIN_S = 1.0
SETTLE_STEP_S = 0.25
SERVE_AFTER_SETTLE_S = 4.0


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


def main():
    Hm = np.load(ROOT / "outputs/t4/H_img_to_court.npy")
    # this camera wanders between points: each clip's frame-0 position is
    # offset from the homography-fit camera by up to ~25 px (the probe's
    # shift search measured it). Foot pixels are in clip-stabilized
    # coords; subtract the clip offset to land in fit-camera coords
    # before projecting.
    offsets = {r["clip"]: (float(r["dx"]), float(r["dy"]))
               for r in csv.DictReader(open(OUT_BASE / "clip_offsets.csv"))}
    rows_out = []
    tracks = sorted(TRACK_DIR.glob("players_t4_point_*.csv"))
    for tpath in tracks:
        stem = tpath.stem.replace("players_", "")
        cap = cv2.VideoCapture(str(ROOT / "clips/points_t4" / f"{stem}.mp4"))
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
                if abs(mx - CENTER_X) <= CENTER_TOL_M and lo <= my <= hi:
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

    out = OUT_BASE / "serves.csv"
    fields = ["clip", "server", "server_x_m", "margin_m", "side",
              "serve_frame", "serve_s", "toss_h_norm", "launch_cy",
              "src", "reason"]
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows_out:
            wr.writerow({k: r.get(k, "") for k in fields})

    called = [r for r in rows_out if r["server"] != "?"]
    print(f"{len(called)}/{len(rows_out)} clips got a server + serve frame")
    n_near = sum(1 for r in called if r["server"] == "near")
    print(f"servers: near {n_near}, far {len(called) - n_near}")
    tight = [r["clip"] for r in called if r["margin_m"] < 1.0]
    if tight:
        print(f"low-margin server calls (<1 m): {tight}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
