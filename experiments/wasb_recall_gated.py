"""WASB crossing-recall — CONFIDENCE-GATED hole bridging (variant C).

Successor to the refuted blind step=1 A/B (experiments/wasb_recall_ab.py,
LOG 2026-07-20). That run raised track coverage (t1 89.8->93.3%) but
LOWERED strict net crossings (82->78): dense step=1 detections inserted
non-monotone samples INSIDE the court-y runs the holes had kept clean,
so the crossing gates (sign monotonicity, CROSS_MAX_STEP_M, min samples)
split or refused runs.

Variant C keeps the shipped step-3 track as the spine and fills holes
ONLY where the added sample is physically consistent with the flight the
hole interrupts:

  C1 (detection-gated)  for each missing frame in a hole, pull in the
                        step-1 detection for that frame IFF its retained
                        heatmap score >= SCORE_GATE AND it sits within
                        CROSS_MAX_STEP_M (court metres, inherited from
                        events.py — NOT a free knob) of the straight-line
                        interpolant between the hole endpoints.
  C2 (+gravity bridge)  additionally, for any still-empty frame in a hole
                        no longer than MAX_HOLE, fabricate the linear
                        (image-space) interpolant point. Image-space ball
                        gravity is a validated unimodal signal
                        (courtvision/boundaries.py); over a short hole the
                        parabola is ~linear, and the interpolant is monotone
                        and within-step by construction.

The point: recover the crossing recall the holes ate WITHOUT injecting the
noise that split runs in the blind variant.

Measure-only. Shipped ball CSVs, charts, scorecards, model, and constants
are untouched. This writes ONLY:

  outputs/<t>/ball_wasb_s2/<arm>/       C tracks (+score col, inert)
  outputs/<t>/charts_wasb_s2/<arm>/     C twin charts
  outputs/diag/recall_gated_<t>_chartsA/  A twin charts (paired subset)
  outputs/diag/recall_gated_<t>.md        the 3-arm report

SCORE_GATE / MAX_HOLE are tuned on t1 ONLY and labelled; t4 is run HELD
OUT with the frozen t1 config to test transfer.

Usage:
    uv run python experiments/wasb_recall_gated.py t1               # tune
    uv run python experiments/wasb_recall_gated.py t4               # held out
    uv run python experiments/wasb_recall_gated.py t1 --clips 5     # pilot
"""

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import ball, chart, config, evaluate, events

ROOT = Path(__file__).resolve().parent.parent

# ---- gate: court tolerance inherited (NOT tuned), score+hole tuned on t1 ----
CY_TOL_M = events.CROSS_MAX_STEP_M       # 6.0 m; one allowed crossing step
SCORE_GATE = 6.0                         # tuned on t1 (see report)
MAX_HOLE = 12                            # C2 bridge cap, frames; tuned on t1


def load_track(ball_dir, stem):
    rows = list(csv.DictReader(open(ball_dir / f"ball_{stem}.csv")))
    return [(int(r["frame"]), float(r["x_stab"]), float(r["y_stab"]),
             float(r.get("score", "nan"))) for r in rows]


def court_xy(Hm, offsets, stem, xs, ys):
    odx, ody = offsets.get(stem, (0.0, 0.0))
    pts = np.stack([np.asarray(xs, float) - odx, np.asarray(ys, float) - ody],
                   axis=1).reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)


def candidates_from_disk(cfg, stems):
    """{stem: {frame: (x_stab, y_stab, score)}} from an existing s1 track."""
    s1_dir = cfg.out_dir / "ball_wasb_s1"
    cand = {}
    for stem in stems:
        p = s1_dir / f"ball_{stem}.csv"
        if p.exists():
            cand[stem] = {f: (x, y, s) for f, x, y, s in load_track(s1_dir, stem)}
    return cand


def candidates_in_memory(cfg, stems, batch=8):
    """Generate step-1 dense per-frame detections in memory (no disk write),
    for matches with no ball_wasb_s1 on disk. x_stab uses the shipped
    stabilization shifts, matching the s1 writer exactly."""
    import torch
    from experiments.wasb_recall_ab import track_clip_s1
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = ball.load_model(device)
    print(f"step-1 candidates in memory on {device}")
    cand = {}
    for stem in stems:
        t0 = time.perf_counter()
        track, n = track_clip_s1(model, device, cfg.clip_path(stem), batch)
        if not track:
            print(f"{stem}: 0/{n} step-1 dets"); continue
        shifts = ball.load_shifts(cfg.out_dir, stem)
        cand[stem] = {}
        for fi in sorted(track):
            x, y, sc = track[fi]
            dx, dy = shifts.get(fi, (0, 0))
            cand[stem][fi] = (round(x - dx, 2), round(y - dy, 2), round(sc, 4))
        print(f"{stem}: {len(track)}/{n} step-1 dets ({time.perf_counter()-t0:.1f}s)")
    return cand


def build_arm(cfg, stem, cand, Hm, offsets, score_gate, bridge, max_hole):
    """Return (frames, xs, ys, n_det, n_int): shipped spine + gated fills."""
    A = load_track(cfg.ball_dir, stem)
    rows = {f: (x, y) for f, x, y, _ in A}
    B = cand.get(stem, {})
    fa = [f for f, _, _, _ in A]
    n_det = n_int = 0
    for k in range(len(fa) - 1):
        f0, f1 = fa[k], fa[k + 1]
        gap = f1 - f0
        if gap <= 1:
            continue
        x0, y0 = rows[f0]
        x1, y1 = rows[f1]
        for f in range(f0 + 1, f1):
            t = (f - f0) / gap
            ix, iy = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            placed = False
            if f in B:
                bx, by, bs = B[f]
                co = court_xy(Hm, offsets, stem, [ix, bx], [iy, by])
                dist = float(np.hypot(co[0, 0] - co[1, 0], co[0, 1] - co[1, 1]))
                if bs >= score_gate and dist <= CY_TOL_M:
                    rows[f] = (bx, by)
                    n_det += 1
                    placed = True
            if not placed and bridge and gap <= max_hole:
                rows[f] = (ix, iy)
                n_int += 1
    fr = sorted(rows)
    return fr, [rows[f][0] for f in fr], [rows[f][1] for f in fr], n_det, n_int


def write_track(ball_dir, stem, fr, xs, ys):
    ball_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"frame": f,
             "cx_raw": round(x / 1280, 6), "cy_raw": round(y / 720, 6),
             "w": 0.01, "h": 0.01,
             "x_stab": round(x, 2), "y_stab": round(y, 2)}
            for f, x, y in zip(fr, xs, ys)]
    with open(ball_dir / f"ball_{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def track_stats(cfg, ball_dir, stem, Hm, offsets):
    rows = load_track(ball_dir, stem)
    frames = np.array([r[0] for r in rows])
    co = court_xy(Hm, offsets, stem, [r[1] for r in rows], [r[2] for r in rows])
    cyc = co[:, 1]
    cap = cv2.VideoCapture(str(cfg.clip_path(stem)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(frames[-1]) + 1
    cap.release()
    gaps = np.diff(frames)
    return {"n": len(frames), "nfr": nfr, "cov": 100.0 * len(frames) / nfr,
            "hole": int(gaps.max()) if len(gaps) else 0,
            "xr": len(events.net_crossings(frames, cyc, fps)),
            "xrw": len(events.net_crossings(frames, cyc, fps, weak=True))}


def twin_scorecard(cfg, ball_dir, charts_dir, stems):
    cfg2 = dataclasses.replace(cfg, ball_dir=ball_dir)
    chart.chart_match(cfg2, stems=stems, charts_dir=charts_dir, quiet=True)
    tally, records = evaluate.evaluate(cfg2, charts_dir=charts_dir, verbose=False)
    mean_dtok = (float(np.mean([r["d_tok"] for r in records]))
                 if records else float("nan"))
    return tally, mean_dtok


def frac(pair):
    return f"{pair[0]}/{pair[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--clips", type=int, default=None)
    ap.add_argument("--score-gate", type=float, default=SCORE_GATE)
    ap.add_argument("--max-hole", type=int, default=MAX_HOLE)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    cfg = config.load(args.match)
    Hm = np.load(cfg.homography)
    offsets = cfg.load_offsets()

    stems = cfg.ball_stems()
    if args.clips:
        stems = stems[:args.clips]

    # candidate source: reuse s1 on disk if present, else generate in memory
    s1_dir = cfg.out_dir / "ball_wasb_s1"
    have_s1 = [s for s in stems if (s1_dir / f"ball_{s}.csv").exists()]
    if len(have_s1) == len(stems):
        cand = candidates_from_disk(cfg, stems)
        cand_note = f"step-1 candidates read from {s1_dir.name} (prior A/B run)"
    else:
        cand = candidates_in_memory(cfg, stems, args.batch)
        cand_note = "step-1 candidates generated in memory (no ball_wasb_s1 on disk)"
    # only compare on stems that actually have a candidate source, so every
    # arm sees the same clip set
    stems = [s for s in stems if s in cand]
    print(f"{cfg.id}: {len(stems)} clips, {cand_note}")

    # ---- build C1 and C2 tracks into ball_wasb_s2/<arm> ----
    arms = {"c1": dict(bridge=False), "c2": dict(bridge=True)}
    counts = {}
    for arm, kw in arms.items():
        bdir = cfg.out_dir / "ball_wasb_s2" / arm
        det = intp = 0
        for stem in stems:
            fr, xs, ys, nd, ni = build_arm(
                cfg, stem, cand, Hm, offsets,
                args.score_gate, kw["bridge"], args.max_hole)
            write_track(bdir, stem, fr, xs, ys)
            det += nd
            intp += ni
        counts[arm] = (det, intp)
        print(f"{arm}: +{det} gated detections, +{intp} bridge points")

    # ---- track recall per arm ----
    Aball = cfg.ball_dir
    C1ball = cfg.out_dir / "ball_wasb_s2" / "c1"
    C2ball = cfg.out_dir / "ball_wasb_s2" / "c2"
    per_clip = []
    for stem in stems:
        per_clip.append((stem,
                         track_stats(cfg, Aball, stem, Hm, offsets),
                         track_stats(cfg, C1ball, stem, Hm, offsets),
                         track_stats(cfg, C2ball, stem, Hm, offsets)))

    # ---- scorecards (shipped evaluator, paired clip set) ----
    charts_a = ROOT / "outputs" / "diag" / f"recall_gated_{cfg.id}_chartsA"
    charts_c1 = cfg.out_dir / "charts_wasb_s2" / "c1"
    charts_c2 = cfg.out_dir / "charts_wasb_s2" / "c2"
    tal_a, dtok_a = twin_scorecard(cfg, Aball, charts_a, stems)
    tal_c1, dtok_c1 = twin_scorecard(cfg, C1ball, charts_c1, stems)
    tal_c2, dtok_c2 = twin_scorecard(cfg, C2ball, charts_c2, stems)

    # B arm (blind step=1) from the prior report if present
    b_row = None
    prior = ROOT / "outputs" / "diag" / f"wasb_ab_{cfg.id}.md"
    if prior.exists():
        b_row = prior.read_text()

    # ---- report ----
    def tot(key, idx):
        return sum(c[idx][key] for c in per_clip)
    nfr = sum(c[1]["nfr"] for c in per_clip)
    lines = [f"# WASB confidence-gated crossing recall — {cfg.id} "
             f"({len(stems)} clips, {time.strftime('%Y-%m-%d')})", "",
             f"A = shipped `ball_wasb` (step=3). "
             f"C1 = A + gated step-1 detections. "
             f"C2 = C1 + gravity (linear) hole bridge (<= {args.max_hole} f). "
             f"Gate: score >= {args.score_gate}, court dist <= "
             f"{CY_TOL_M} m (CROSS_MAX_STEP_M, inherited). "
             f"{cand_note}. "
             f"{'Tuned on this match.' if cfg.id=='t1' else 'HELD OUT: t1 config frozen.'}",
             "", "## Track recall (totals)", "",
             "| arm | frames | cov% | strict xr | weak xr | +det | +bridge |",
             "|---|---|---|---|---|---|---|"]
    lines.append(f"| A | {tot('n',1)} | {100*tot('n',1)/nfr:.1f} | "
                 f"{tot('xr',1)} | {tot('xrw',1)} | - | - |")
    lines.append(f"| C1 | {tot('n',2)} | {100*tot('n',2)/nfr:.1f} | "
                 f"{tot('xr',2)} | {tot('xrw',2)} | {counts['c1'][0]} | 0 |")
    lines.append(f"| C2 | {tot('n',3)} | {100*tot('n',3)/nfr:.1f} | "
                 f"{tot('xr',3)} | {tot('xrw',3)} | {counts['c2'][0]} | {counts['c2'][1]} |")

    lines += ["", "## Twin scorecards (shipped evaluator, paired clip set)", "",
              "| metric | A | C1 | C2 |", "|---|---|---|---|"]
    for key, label in [("server", "server end"), ("rally_pm1", "rally len +/-1"),
                       ("serve_zone", "serve zone"), ("ending", "ending type"),
                       ("accept", "acceptance <=1 edit")]:
        lines.append(f"| {label} | {frac(tal_a[key])} | {frac(tal_c1[key])} "
                     f"| {frac(tal_c2[key])} |")
    lines.append(f"| letters (aligned) | "
                 f"{tal_a['letters_al_match']}/{tal_a['letters_al_total']} | "
                 f"{tal_c1['letters_al_match']}/{tal_c1['letters_al_total']} | "
                 f"{tal_c2['letters_al_match']}/{tal_c2['letters_al_total']} |")
    lines.append(f"| mean token distance | {dtok_a:.2f} | {dtok_c1:.2f} "
                 f"| {dtok_c2:.2f} |")

    lines += ["", "## Per-clip track recall (strict xr, cov%)", "",
              "| clip | A xr | C1 xr | C2 xr | A cov | C2 cov | A hole | C2 hole |",
              "|---|---|---|---|---|---|---|---|"]
    for stem, a, c1, c2 in per_clip:
        lines.append(f"| {stem} | {a['xr']} | {c1['xr']} | {c2['xr']} "
                     f"| {a['cov']:.0f} | {c2['cov']:.0f} "
                     f"| {a['hole']} | {c2['hole']} |")
    if b_row:
        lines += ["", f"B arm (blind step=1) context: see "
                  f"outputs/diag/wasb_ab_{cfg.id}.md."]

    report = ROOT / "outputs" / "diag" / f"recall_gated_{cfg.id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {report}")


if __name__ == "__main__":
    main()
