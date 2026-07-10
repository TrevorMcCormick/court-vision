"""T3 — serve detection: m3_serve_detect, t3 paths + clip-offset correction.

Consumes the $0 bg-sub player tracks (players_<clip>.csv).

v1 asked only "who's closer to the center mark early?" and paid for it:
frame-checking found cold opens (reel intro starts mid-rally), garbage
blobs (zoom-tail fragments), and receiver picks — when one side's early
track is junk, "closer to center" degenerates to a coin toss on noise.

v2 gates every claim and refuses to guess:
  coverage gate — a side counts only if >=60% of early frames tracked
  center gate   — server must stand within 2.0 m of the center mark
  baseline gate — ...and within ~3 m of their own baseline
  toss gate     — the blob's smoothed height must peak >=1.12x its median
                  inside the serve window (a real toss stretches the
                  silhouette; receivers don't)
A clip yields either ONE candidate passing all gates (a confident serve)
or an explicit no_confident_serve flag. Side of the center mark gives
deuce/ad for free.

fps is read from the clip, never assumed — the reel runs 25 where
rally.mp4 ran 30, and M2's per-frame thresholds already taught that
lesson once.

Usage:
    uv run experiments/t3_serve_detect.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "t3"
TRACK_DIR = OUT_BASE / "players"
ROOT = Path(__file__).resolve().parent.parent

W_C, L_C = 10.97, 23.77
CENTER_X = W_C / 2
EARLY_S = 1.0
SERVE_WINDOW_S = 4.0
SMOOTH = 5
COVERAGE_MIN = 0.60
CENTER_TOL_M = 4.3   # t2 used 2.0; clay servers stand WIDE (Ruud serves
                     # from x=1.4, 4.1 m off the mark — frame-checked).
                     # The toss + baseline gates carry the discrimination.
BASELINE_TOL = {"far": (-3.5, 1.0), "near": (L_C - 1.0, L_C + 3.5)}
TOSS_RATIO = 1.12


def main():
    Hm = np.load(ROOT / "outputs/t3/H_img_to_court.npy")
    # this camera wanders between points: each clip's frame-0 position is
    # offset from the homography-fit camera by up to ~25 px (the probe's
    # shift search measured it). Foot pixels are in clip-stabilized
    # coords; subtract the clip offset to land in fit-camera coords
    # before projecting.
    offsets = {r["clip"]: (float(r["dx"]), float(r["dy"]))
               for r in csv.DictReader(open(OUT_BASE / "clip_offsets.csv"))}
    rows_out = []
    tracks = sorted(TRACK_DIR.glob("players_t3_point_*.csv"))
    for tpath in tracks:
        stem = tpath.stem.replace("players_", "")
        cap = cv2.VideoCapture(str(ROOT / "clips/points_t3" / f"{stem}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

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

        early_n = int(EARLY_S * fps)
        win = int(SERVE_WINDOW_S * fps)

        candidates = []
        gate_log = []
        for side in ("near", "far"):
            early = [per[side][fi] for fi in sorted(per[side]) if fi < early_n]
            cov = len(early) / early_n
            if cov < COVERAGE_MIN:
                gate_log.append(f"{side}:coverage {cov:.0%}")
                continue
            xys = [court_xy(r) for r in early]
            mx = float(np.median([p[0] for p in xys]))
            my = float(np.median([p[1] for p in xys]))
            if abs(mx - CENTER_X) > CENTER_TOL_M:
                gate_log.append(f"{side}:off-center {mx:.1f}m")
                continue
            lo, hi = BASELINE_TOL[side]
            if not (lo <= my <= hi):
                gate_log.append(f"{side}:off-baseline y={my:.1f}m")
                continue
            hs = [(fi, float(per[side][fi]["h"])) for fi in sorted(per[side]) if fi < win]
            # t2 lesson: this reel's cuts are DISSOLVES — the first ~5
            # frames diff against the plate as one giant blob (h~0.82 of
            # frame vs ~0.25 for a real near player), and the toss gate's
            # argmax grabbed it at f0 with ratio ~4x. A real toss stretch
            # is ~1.1-1.3x: drop blobs taller than 2x the series median
            # before peak-finding.
            h_med = float(np.median([h for _, h in hs]))
            hs = [(fi, h) for fi, h in hs if h <= 2.0 * h_med]
            fis = [f for f, _ in hs]
            hh = np.array([h for _, h in hs])
            if len(hh) >= SMOOTH:
                pad = np.pad(hh, SMOOTH // 2, mode="edge")
                hh = np.array([np.median(pad[j:j + SMOOTH]) for j in range(len(hh))])
            peak = int(np.argmax(hh))
            ratio = float(hh[peak]) / float(np.median(hh))
            if ratio < TOSS_RATIO:
                gate_log.append(f"{side}:no-toss x{ratio:.2f}")
                continue
            candidates.append({"side": side, "mx": mx, "my": my,
                               "serve_frame": fis[peak], "toss_ratio": ratio})

        if not candidates:
            rows_out.append({"clip": stem, "server": "?",
                             "reason": "no_confident_serve: " + "; ".join(gate_log)})
            continue
        best = min(candidates, key=lambda c: abs(c["mx"] - CENTER_X))
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
            "reason": "both_ends_passed" if len(candidates) == 2 else "",
        })

    out = OUT_BASE / "serves.csv"
    fields = ["clip", "server", "server_x_m", "margin_m", "side",
              "serve_frame", "serve_s", "toss_h_norm", "reason"]
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows_out:
            wr.writerow({k: r.get(k, "") for k in fields})

    good = [r for r in rows_out if not r.get("reason")]
    print(f"{len(good)}/{len(rows_out)} clips got a server + serve frame")
    n_near = sum(1 for r in good if r["server"] == "near")
    print(f"servers: near {n_near}, far {len(good) - n_near}")
    tight = [r for r in good if r["margin_m"] < 1.0]
    if tight:
        print(f"low-margin server calls (<1 m): {[r['clip'] for r in tight]}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
