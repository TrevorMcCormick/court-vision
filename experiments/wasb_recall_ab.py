"""WASB crossing-recall A/B — shipped step-3 stride vs step-1 overlap.

docs/benchmark.md names crossing recall the binding constraint on rally
length: v5 counts what the spine sees, and track holes eat crossings
(t1 night reel worst). The shipped ball pass (courtvision/ball.py) runs
WASB on NON-overlapping triplet windows (step=3), so each frame gets
exactly one heatmap look, and the per-detection heatmap score is
discarded at CSV write. Hypothesis: step=1 overlapping windows pool up
to three looks per frame — the SAME detector, same SCORE_T/MAX_DISP
gates, run denser — and recover detections a single unlucky window
misses, raising track recall and therefore net-crossing recall.

Measure-only. Shipped ball CSVs, charts, and scorecards are untouched;
this writes ONLY:

  outputs/<t>/ball_wasb_s1/          B tracks (step=1, + retained score col)
  outputs/<t>/charts_wasb_s1/        B twin charts
  outputs/diag/wasb_ab_<t>_chartsA/  A twin charts (same clip subset,
                                     so subset scorecards stay paired)
  outputs/diag/wasb_ab_<t>.md        the A/B report

Both twins chart with the SHIPPED constants, serves.csv, players, and
homography — the ball track density is the only difference. The B CSVs
carry an extra `score` column (heatmap-weighted blob mass, the value
ball.py drops); every downstream reader is a DictReader, so the column
is inert.

Usage:
    uv run python experiments/wasb_recall_ab.py t1 --clips 3   # pilot
    uv run python experiments/wasb_recall_ab.py t1             # full match
    uv run python experiments/wasb_recall_ab.py t1 --skip-track  # reuse B tracks
"""

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import ball, chart, config, evaluate, events

ROOT = Path(__file__).resolve().parent.parent
STEP = 1                      # the one experimental knob (shipped: FRAMES_IN=3)


def track_clip_s1(model, device, clip_path, batch=8):
    """ball.track_clip with step=STEP windows: every frame is predicted
    by up to FRAMES_IN window positions and the blob candidates POOL
    before the online-tracker pass (best score wins, same MAX_DISP
    gate). Everything else is verbatim from the shipped pass."""
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n < ball.FRAMES_IN:
        return {}, n
    h, w = frames[0].shape[:2]
    fwd, inv = ball.frame_transforms(w, h)

    starts = list(range(0, n - ball.FRAMES_IN + 1, STEP))
    if starts[-1] + ball.FRAMES_IN < n:
        starts.append(n - ball.FRAMES_IN)

    dets_per_frame = {fi: [] for fi in range(n)}
    with torch.no_grad():
        for b0 in range(0, len(starts), batch):
            chunk = starts[b0:b0 + batch]
            inp = np.stack([
                np.concatenate([ball.preprocess(frames[s + k], fwd)
                                for k in range(ball.FRAMES_IN)], axis=0)
                for s in chunk])
            out = model(torch.from_numpy(inp).to(device))[0]
            hms = torch.sigmoid(out).cpu().numpy()
            for i, s in enumerate(chunk):
                for k in range(ball.FRAMES_IN):
                    dets_per_frame[s + k].extend(ball.detect_blobs(hms[i, k], inv))

    track = {}
    last_xy = None
    for fi in range(n):
        dets = dets_per_frame.get(fi, [])
        if last_xy is not None:
            dets = [d for d in dets
                    if np.linalg.norm(d["xy"] - last_xy) < ball.MAX_DISP]
        if dets:
            best = max(dets, key=lambda d: d["score"])
            track[fi] = (float(best["xy"][0]), float(best["xy"][1]),
                         best["score"])
            last_xy = best["xy"]
        else:
            last_xy = None
    return track, n


def retrack(cfg, s1_dir, stems, batch=8):
    """B pass: track stems with step=STEP into s1_dir, keeping the score.
    Returns {stem: seconds}."""
    s1_dir.mkdir(parents=True, exist_ok=True)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = ball.load_model(device)
    print(f"WASB tennis weights on {device}, step={STEP}")
    secs = {}
    for stem in stems:
        t0 = time.perf_counter()
        track, n = track_clip_s1(model, device, cfg.clip_path(stem), batch)
        secs[stem] = time.perf_counter() - t0
        if not track:
            print(f"{stem}: 0/{n} frames detected — nothing written")
            continue
        shifts = ball.load_shifts(cfg.out_dir, stem)
        rows = []
        for fi in sorted(track):
            x, y, sc = track[fi]
            dx, dy = shifts.get(fi, (0, 0))
            rows.append({"frame": fi,
                         "cx_raw": round(x / 1280, 6),
                         "cy_raw": round(y / 720, 6),
                         "w": 0.01, "h": 0.01,
                         "x_stab": round(x - dx, 2),
                         "y_stab": round(y - dy, 2),
                         "score": round(sc, 4)})
        with open(s1_dir / f"ball_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"{stem}: {len(rows)}/{n} frames ({100 * len(rows) / n:.0f}%), "
              f"{secs[stem]:.1f}s")
    return secs


def track_stats(cfg, ball_dir, stem, Hm, offsets):
    """Per-clip recall metrics off one track CSV: frames, coverage vs the
    full clip, max hole, and net crossings at both gate levels — the
    projection reproduces confidence.point_signals exactly."""
    rows = list(csv.DictReader(open(ball_dir / f"ball_{stem}.csv")))
    frames = np.array([int(r["frame"]) for r in rows])
    xs = np.array([float(r["x_stab"]) for r in rows])
    ys = np.array([float(r["y_stab"]) for r in rows])
    odx, ody = offsets.get(stem, (0.0, 0.0))
    pts = np.stack([xs - odx, ys - ody], axis=1).reshape(-1, 1, 2).astype(np.float32)
    cyc = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)[:, 1]

    cap = cv2.VideoCapture(str(cfg.clip_path(stem)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(frames[-1]) + 1
    cap.release()

    gaps = np.diff(frames)
    return {"n": len(frames), "nfr": nfr,
            "cov": 100.0 * len(frames) / nfr,
            "hole": int(gaps.max()) if len(gaps) else 0,
            "xr": len(events.net_crossings(frames, cyc, fps)),
            "xrw": len(events.net_crossings(frames, cyc, fps, weak=True))}


def twin_scorecard(cfg, ball_dir, charts_dir, stems):
    """Chart stems from ball_dir into charts_dir (shipped chart loop,
    shipped constants) and score with the shipped evaluator."""
    cfg2 = dataclasses.replace(cfg, ball_dir=ball_dir)
    chart.chart_match(cfg2, stems=stems, charts_dir=charts_dir, quiet=True)
    tally, records = evaluate.evaluate(cfg2, charts_dir=charts_dir, verbose=False)
    mean_dtok = (float(np.mean([r["d_tok"] for r in records]))
                 if records else float("nan"))
    return tally, mean_dtok, records


def frac(pair):
    return f"{pair[0]}/{pair[1]}"


def why(cfg, s1_dir, stem):
    """Name the mechanism behind a strict-crossing delta on one clip:
    prints A/B crossing intervals plus shared-frame agreement. (t1 full
    run: coverage UP 89.8->93.3% but strict xr DOWN 82->78 — the filled
    hole frames insert non-monotone samples inside runs the holes kept
    clean, and a pooled higher-score candidate occasionally swaps the
    pick on a shared frame, diverging the tracker chain.)"""
    Hm = np.load(cfg.homography)
    odx, ody = cfg.load_offsets().get(stem, (0.0, 0.0))

    def load(d):
        rows = list(csv.DictReader(open(d / f"ball_{stem}.csv")))
        fr = np.array([int(r["frame"]) for r in rows])
        xs = np.array([float(r["x_stab"]) for r in rows])
        ys = np.array([float(r["y_stab"]) for r in rows])
        pts = np.stack([xs - odx, ys - ody], 1).reshape(-1, 1, 2).astype(np.float32)
        cyc = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)[:, 1]
        return fr, xs, ys, cyc

    fa, xa, ya, ca = load(cfg.ball_dir)
    fb, xb, yb, cb = load(s1_dir)
    cap = cv2.VideoCapture(str(cfg.clip_path(stem)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    for tag, fr, cy in (("A", fa, ca), ("B", fb, cb)):
        xr = events.net_crossings(fr, cy, fps)
        print(f"{stem} {tag} xr: {[(a, b) for a, b, _ in xr]}")
    da = dict(zip(fa.tolist(), zip(xa, ya)))
    db = dict(zip(fb.tolist(), zip(xb, yb)))
    shared = sorted(set(da) & set(db))
    d = np.array([np.hypot(da[f][0] - db[f][0], da[f][1] - db[f][1])
                  for f in shared])
    print(f"shared {len(shared)} frames: {np.mean(d < 0.5) * 100:.0f}% within "
          f"0.5px, p95 {np.percentile(d, 95):.1f}px, max {d.max():.0f}px; "
          f"B-only {len(set(db) - set(da))}, A-only {len(set(da) - set(db))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--clips", type=int, default=None,
                    help="subset: first N of the shipped tracked stems")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-track", action="store_true",
                    help="reuse existing ball_wasb_s1 CSVs")
    ap.add_argument("--why", metavar="STEM",
                    help="diagnose one clip's crossing delta and exit")
    args = ap.parse_args()

    cfg = config.load(args.match)
    s1_dir = cfg.out_dir / "ball_wasb_s1"
    if args.why:
        why(cfg, s1_dir, args.why)
        return
    charts_b = cfg.out_dir / "charts_wasb_s1"
    charts_a = ROOT / "outputs" / "diag" / f"wasb_ab_{cfg.id}_chartsA"
    report = ROOT / "outputs" / "diag" / f"wasb_ab_{cfg.id}.md"

    # paired set: A's tracked stems only, so every comparison row has
    # both arms (clips step=3 never tracked at all are a separate story)
    stems = cfg.ball_stems()
    if args.clips:
        stems = stems[:args.clips]
    print(f"{cfg.id}: {len(stems)} clips {stems[0]}..{stems[-1]}")

    if args.skip_track:
        secs = {s: float("nan") for s in stems}
        stems = [s for s in stems if (s1_dir / f"ball_{s}.csv").exists()]
    else:
        secs = retrack(cfg, s1_dir, stems, args.batch)
        stems = [s for s in stems if (s1_dir / f"ball_{s}.csv").exists()]

    Hm = np.load(cfg.homography)
    offsets = cfg.load_offsets()
    per_clip = []
    for stem in stems:
        a = track_stats(cfg, cfg.ball_dir, stem, Hm, offsets)
        b = track_stats(cfg, s1_dir, stem, Hm, offsets)
        per_clip.append((stem, a, b))

    tal_a, dtok_a, _ = twin_scorecard(cfg, cfg.ball_dir, charts_a, stems)
    tal_b, dtok_b, _ = twin_scorecard(cfg, s1_dir, charts_b, stems)

    # ---- report ----
    hdr = (f"| clip | A fr | B fr | A cov% | B cov% | A hole | B hole "
           f"| A xr | B xr | A xrw | B xrw | B secs |")
    sep = "|" + "---|" * 12
    lines = [f"# WASB step-3 vs step-1 A/B — {cfg.id} "
             f"({len(stems)} clips, {time.strftime('%Y-%m-%d')})", "",
             f"A = shipped `ball_wasb` (step=3, non-overlapping windows). "
             f"B = `ball_wasb_s1` (step=1 overlap, pooled blob candidates, "
             f"score retained). Same weights, SCORE_T, MAX_DISP, chart "
             f"constants, serves, players, homography — density is the "
             f"only change. Generated by `experiments/wasb_recall_ab.py`.",
             "", "## Per-clip track recall", "", hdr, sep]
    for stem, a, b in per_clip:
        lines.append(
            f"| {stem} | {a['n']} | {b['n']} | {a['cov']:.0f} | {b['cov']:.0f} "
            f"| {a['hole']} | {b['hole']} | {a['xr']} | {b['xr']} "
            f"| {a['xrw']} | {b['xrw']} | {secs.get(stem, float('nan')):.1f} |")
    tots = {k: (sum(a[k] for _, a, _ in per_clip),
                sum(b[k] for _, _, b in per_clip))
            for k in ("n", "xr", "xrw")}
    cov = (100 * tots["n"][0] / sum(a["nfr"] for _, a, _ in per_clip),
           100 * tots["n"][1] / sum(a["nfr"] for _, a, _ in per_clip))
    lines += ["",
              f"Totals: frames {tots['n'][0]} -> {tots['n'][1]}, "
              f"coverage {cov[0]:.1f}% -> {cov[1]:.1f}%, "
              f"strict crossings {tots['xr'][0]} -> {tots['xr'][1]}, "
              f"weak crossings {tots['xrw'][0]} -> {tots['xrw'][1]}.",
              "", "## Twin scorecards (shipped evaluator, paired clip set)", "",
              "| metric | A (step=3) | B (step=1) |", "|---|---|---|"]
    for key, label in [("server", "server end"), ("rally_pm1", "rally len ±1"),
                       ("serve_zone", "serve zone"), ("ending", "ending type"),
                       ("accept", "acceptance ≤1 edit")]:
        lines.append(f"| {label} | {frac(tal_a[key])} | {frac(tal_b[key])} |")
    lines.append(f"| letters (aligned) | "
                 f"{tal_a['letters_al_match']}/{tal_a['letters_al_total']} | "
                 f"{tal_b['letters_al_match']}/{tal_b['letters_al_total']} |")
    lines.append(f"| mean token distance | {dtok_a:.2f} | {dtok_b:.2f} |")
    ts = [v for v in secs.values() if v == v]
    if ts:
        lines += ["", f"B wall clock: {np.mean(ts):.1f} s/clip mean "
                      f"({np.sum(ts):.0f} s for {len(ts)} clips)."]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[lines.index("## Per-clip track recall"):]))
    print(f"\n-> {report}")


if __name__ == "__main__":
    main()
