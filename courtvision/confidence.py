"""Calibrated per-point confidence — WHICH drafted points to trust.

The benchmark's honest read: 57% of points sit within 5 token edits of
MCP truth but only ~11% within 2 — a charter correcting the draft
needs to know which is which BEFORE looking. Every signal here is
already computed by the pipeline; nothing new is measured:

  serve      call committed / refused, stance margin (m), chart
             override of the detector's end
  strikers   striker-chain conflict count (residual missed-hit signal)
  ball       track coverage %, max hole (s), hole count
  letters    refused-'?' fraction, mean contact distance (px)
  directions signal-tier quality per shot (the measured ladder:
             near-landing .86 > receiver contact .77 > crossing .67 >
             far landing .47), recomputed from the chart artifacts
  ending     committed vs '?'
  structure  crossings-vs-shots consistency (a live rally crosses the
             net every shot), shots per tracked second, mean charted
             inter-shot gap (a live exchange runs ~0.7-1.1 s; a chart
             pacing 1.5 s+ between shots is recording every other
             stroke — the t4 grass autopsy, named from pixels), and the
             mid-rally-start signature: a "serve" called 0 s into the
             clip whose ball-launch cy sits INSIDE the court is a rally
             crossing in costume (a real serve's toss projects far
             beyond the baselines) — the observable trace of the clay
             editor cutting INTO rallies
  pre-serve  weak-gated net crossings that END before the charted
             serve: rally-speed ball flight before our "serve" means
             the clip joined the point mid-rally (the grass editor's
             dissolve-cut), and the chart cannot be the whole point —
             the stance-called-serve blind spot the launch gate can't
             see (it only inspects src=ball serves)
  spine      a chart claiming 3+ shots while the ball never once
             crosses the net even at the weak gates has no crossing
             spine at all — a rally story invented from track noise

Scorer: logistic regression (numpy, deterministic, L2) on standardized
signals, predicting P(point within 5 token edits of MCP truth) — the
"the draft is a usable starting point" bar — PLUS three mechanistic
gates (rules from geometry and physics, not fitted): serve-launch
plausibility, no pre-serve rally crossings (< 2), and the rally has a
crossing spine.
CALIBRATION DISCIPLINE: thresholds and the honest performance numbers
come from leave-one-match-out fits across the 4 benchmark matches (fit
on 3, score the held-out 4th); the shipped model in
data/confidence_model.json is then fit on all 135 points by the same
threshold rule.

TIERS — two, not three, and here is why. The wished-for top tier
("sign off at a glance": within 2 token edits, >=85% precision) was
built first and does NOT survive leave-one-match-out: 33-50% held-out
precision at <=6% coverage — 135 points at an 11% base rate cannot
support it, full record in docs/benchmark.md. What survives:

  HIGH  P(<=5 edits) past the fold's 90%-train-precision threshold AND
        all three mechanistic gates pass. LOMO (7 matches, 491 points):
        ~94% of high-flagged points are within 5 edits, at ~20%
        coverage — the table in docs/benchmark.md.
  LOW   everything else — expect heavy correction or a re-chart.

Usage:
    uv run python -m courtvision calibrate       # LOMO report + ship model
"""

import csv
import json

import cv2
import numpy as np

from . import config, directions, events, evaluate
from .config import ROOT

MODEL_PATH = ROOT / "data" / "confidence_model.json"

TIER_QUALITY = {"land_near": 0.86, "contact": 0.77, "cross": 0.67,
                "land_far": 0.47, "none": 0.0}

FEATURES = ["serve_committed", "serve_margin_m", "serve_overridden",
            "serve_zone_committed", "serve_s", "serve_launch_plausible",
            "conflicts", "n_holes", "coverage", "max_hole_s",
            "letters_refused_frac", "mean_contact_dist",
            "dir_quality_mean", "dirs_refused_frac", "ending_committed",
            "crossings_gap", "n_shots", "shots_per_s",
            "xr_pre_serve", "rally_spineless", "mean_shot_gap_s"]

PRECISION_TARGET = 0.90       # train precision required to flag HIGH
GOOD_EDITS = 5                # the "usable starting point" bar (see module doc)
STRICT_EDITS = 2              # the unsupported sign-off bar, reported anyway

# the model's inputs (a subset of the signals point_signals records —
# the rest still travel to the export sidecar for the charter's eyes)
MODEL_FEATURES = ["n_shots", "letters_refused_frac", "dirs_refused_frac",
                  "serve_zone_committed", "ending_committed",
                  "serve_committed", "crossings_gap", "max_hole_s",
                  "serve_s", "serve_launch_plausible", "xr_pre_serve"]

# the mechanistic gates, applied OUTSIDE the fit (rules, not weights)
GATES = ("serve_launch_plausible == 1 and xr_pre_serve < 2 "
         "and rally_spineless == 0")


def gate_pass(f):
    """The three mechanistic gates, from a point's signal dict."""
    return (f["serve_launch_plausible"] == 1.0
            and f["xr_pre_serve"] < 2
            and f["rally_spineless"] == 0.0)


def _shots_from_chart(rows):
    """chart2_*.csv rows -> minimal shot dicts the direction signals read."""
    shots = []
    for r in rows:
        shots.append({
            "frame": int(r["frame"]),
            "contact_frame": int(r["contact_frame"]) if r["contact_frame"] else None,
            "is_serve": r["is_serve"] == "True",
            "synth": r["synth"] == "True",
            "striker": r["striker"],
            "letter": r["letter"],
            "zone": r["zone"],
            "contact_dist_px": float(r["contact_dist_px"]) if r["contact_dist_px"] else None,
            "landing_x": float(r["landing_x"]) if r["landing_x"] else None,
            "landing_y": float(r["landing_y"]) if r["landing_y"] else None,
        })
    return shots


def _dir_tiers(shots, frames, cyc, cxc, fps):
    """Which rung of the measured signal ladder each rally shot's
    direction came from (recomputed exactly as directions.estimate)."""
    tiers = []
    for k, sh in enumerate(shots):
        if sh["is_serve"]:
            continue
        nxt = shots[k + 1] if k + 1 < len(shots) else None
        land = directions.landing_signal(sh)
        cont = directions.contact_signal(sh, nxt, frames, cxc)
        cross = directions.crossing_signal(sh, nxt, frames, cyc, cxc, fps)
        if land is not None and land[1] == "near":
            tiers.append("land_near")
        elif cont is not None and cont[1] in ("near", "far"):
            tiers.append("contact")
        elif cross is not None:
            tiers.append("cross")
        elif land is not None:
            tiers.append("land_far")
        else:
            tiers.append("none")
    return tiers


def point_signals(cfg, clip, mc, Hm, offsets, serves, charts_dir=None):
    """The per-point signal vector, from artifacts already on disk."""
    charts_dir = charts_dir or cfg.charts_dir
    shots = _shots_from_chart(
        list(csv.DictReader(open(charts_dir / f"chart2_{clip}.csv"))))

    ball = list(csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv")))
    frames = np.array([int(r["frame"]) for r in ball])
    xs = np.array([float(r["x_stab"]) for r in ball])
    ys = np.array([float(r["y_stab"]) for r in ball])
    odx, ody = offsets.get(clip, (0.0, 0.0))
    pts = np.stack([xs - odx, ys - ody], axis=1).reshape(-1, 1, 2).astype(np.float32)
    court = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    cyc, cxc = court[:, 1], court[:, 0]

    cap = cv2.VideoCapture(str(cfg.clip_path(clip)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(frames[-1]) + 1
    cap.release()

    span = int(frames[-1]) - int(frames[0]) + 1
    gaps = np.diff(frames)
    max_hole = int(gaps.max()) if len(gaps) else 0

    s = serves.get(clip, {})
    margin = s.get("margin_m", "")
    serve_committed = 1.0 if mc["server"] != "?" else 0.0
    serve_overridden = 1.0 if (mc["server"] not in ("?", mc["server_used"])) else 0.0
    # mid-rally-start signature (the clay editor cuts INTO rallies): the
    # ball-launch serve detector fires on the clip's FIRST crossing, so a
    # "serve" 0 s into the clip whose launch cy sits INSIDE the court is
    # a rally crossing in costume — a real serve's toss projects far
    # beyond the baselines (04's far serve reads cy -20.3). Stance-based
    # calls (src=players) verified a real serve posture.
    serve_s = float(s["serve_s"]) if s.get("serve_s") else 5.0
    if s.get("src") == "ball" and s.get("launch_cy"):
        cy0 = float(s["launch_cy"])
        launch_plausible = 1.0 if (cy0 <= -10.0 or cy0 >= 30.0) else 0.0
    else:
        launch_plausible = serve_committed

    rally = [sh for sh in shots if not sh["is_serve"]]
    n_rally = len(rally)
    # an ace/serve-winner has no rally shots and therefore refused nothing
    letters_refused = (sum(1 for sh in rally if sh["letter"] not in ("f", "b"))
                       / n_rally if n_rally else 0.0)
    dists = [sh["contact_dist_px"] for sh in shots
             if sh["contact_dist_px"] is not None and not sh["synth"]]
    mean_dist = float(np.mean(dists)) if dists else 300.0

    tiers = _dir_tiers(shots, frames, cyc, cxc, fps)
    dir_quality = (float(np.mean([TIER_QUALITY[t] for t in tiers]))
                   if tiers else 0.0)
    dirs_refused = (sum(1 for sh in rally if sh["zone"] not in ("1", "2", "3"))
                    / n_rally if n_rally else 0.0)
    serve_sh = next((sh for sh in shots if sh["is_serve"]), None)
    serve_zone_committed = 1.0 if (serve_sh is not None
                                   and serve_sh["zone"] in ("4", "5", "6")) else 0.0

    xr = events.net_crossings(frames, cyc, fps)
    crossings_gap = abs(len(shots) - (len(xr) + 1))

    # the whole-point gates the t4 autopsy earned (2026-07-11): weak
    # crossings ending before the charted serve = the clip joined the
    # rally mid-flight (the grass editor's dissolve-cut, invisible to
    # the launch gate on stance-called serves); a 3+-shot chart whose
    # window holds ZERO weak crossings has no spine at all.
    xw = events.net_crossings(frames, cyc, fps, weak=True)
    shot_frames = [sh["frame"] for sh in shots]
    serve_f = serve_sh["frame"] if serve_sh is not None else None
    xr_pre_serve = float(sum(1 for (a, b, _) in xw
                             if serve_f is not None and b < serve_f))
    w0 = serve_f if serve_f is not None else shot_frames[0]
    w1 = max(shot_frames) + int(1.5 * fps)
    xw_in = sum(1 for (a, b, _) in xw if b >= w0 and a <= w1)
    rally_spineless = 1.0 if (len(shots) >= 3 and xw_in == 0) else 0.0
    mean_shot_gap = (float(np.mean(np.diff(shot_frames))) / fps
                     if len(shot_frames) >= 2 else 0.0)

    return {
        "serve_committed": serve_committed,
        "serve_margin_m": min(float(margin), 3.0) if margin else 0.0,
        "serve_overridden": serve_overridden,
        "conflicts": float(mc["conflicts"]),
        "n_holes": float(mc["n_holes"]),
        "coverage": len(frames) / span if span else 0.0,
        "max_hole_s": max_hole / fps,
        "letters_refused_frac": letters_refused,
        "mean_contact_dist": min(mean_dist, 300.0),
        "dir_quality_mean": dir_quality,
        "dirs_refused_frac": dirs_refused,
        "serve_zone_committed": serve_zone_committed,
        "serve_s": min(serve_s, 5.0),
        "serve_launch_plausible": launch_plausible,
        "ending_committed": 1.0 if mc["ending"] != "?" else 0.0,
        "crossings_gap": float(crossings_gap),
        "n_shots": float(len(shots)),
        "shots_per_s": len(shots) / (nfr / fps) if nfr else 0.0,
        "xr_pre_serve": xr_pre_serve,
        "rally_spineless": rally_spineless,
        "mean_shot_gap_s": mean_shot_gap,
    }


def match_signals(cfg, charts_dir=None):
    """{clip: signal dict} for every charted point of a match."""
    charts_dir = charts_dir or cfg.charts_dir
    Hm = np.load(cfg.homography)
    offsets = cfg.load_offsets()
    serves = cfg.load_serves()
    match = {r["clip"]: r
             for r in csv.DictReader(open(charts_dir / "match_chart_v2.csv"))}
    return {clip: point_signals(cfg, clip, mc, Hm, offsets, serves, charts_dir)
            for clip, mc in match.items()}


# ---------------------------------------------------------------------------
# the scorer: standardized logistic regression, numpy, deterministic
# ---------------------------------------------------------------------------

def _fit_logistic(X, y, l2=1.0, iters=4000, lr=0.1):
    n, d = X.shape
    w = np.zeros(d + 1)
    Xb = np.hstack([np.ones((n, 1)), X])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / n
        g[1:] += l2 * w[1:] / n
        w -= lr * g
    return w


def _predict(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1 / (1 + np.exp(-Xb @ w))


def _standardize(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _high_threshold(p_train, good_train, target=PRECISION_TARGET, min_flags=5):
    """Smallest probability threshold whose TRAIN precision for
    'within GOOD_EDITS edits' meets the target with at least min_flags
    points flagged; inf when no such region exists."""
    order = np.argsort(-p_train)
    best = np.inf
    flagged_good, flagged = 0, 0
    for idx in order:
        flagged += 1
        flagged_good += good_train[idx]
        if flagged >= min_flags and flagged_good / flagged >= target:
            best = p_train[idx]
    return best


def collect(charts_dirs=None):
    """All four matches: (match_id, clip, features array, d_tok)."""
    rows = []
    for mid in config.match_ids():
        cfg = config.load(mid)
        cdir = (charts_dirs or {}).get(mid)
        sigs = match_signals(cfg, cdir)
        _, records = evaluate.evaluate(cfg, charts_dir=cdir, verbose=False)
        for rec in records:
            f = sigs[rec["clip"]]
            rows.append({"match": mid, "clip": rec["clip"],
                         "x": np.array([f[k] for k in FEATURES]),
                         "d_tok": rec["d_tok"]})
    return rows


def _lomo_flags(X, good, gate, match_of, mids, target=PRECISION_TARGET,
                min_flags=5):
    """Leave-one-match-out flags: fit on 3 matches, flag the held-out
    4th; the mechanistic gate is applied outside the fit."""
    flags = np.zeros(len(good), bool)
    for held in mids:
        tr, te = match_of != held, match_of == held
        Xtr, mu, sd = _standardize(X[tr])
        w = _fit_logistic(Xtr, good[tr])
        t_high = _high_threshold(_predict(w, Xtr), good[tr], target, min_flags)
        flags[te] = _predict(w, (X[te] - mu) / sd) >= t_high
    return flags & gate


def _tier_table(title, flags, good, d, match_of, mids, bar):
    print(title)
    print(f"{'match':10}{'high prec.':>14}{'coverage':>10}"
          f"{'low <=' + str(bar) + ' rate':>14}")
    for mid in mids + ["pooled"]:
        m = np.ones(len(d), bool) if mid == "pooled" else match_of == mid
        hi, lo = m & flags, m & ~flags
        prec = (f"{good[hi].mean():.0%} ({int(good[hi].sum())}/{hi.sum()})"
                if hi.sum() else "—  (0/0)")
        print(f"{mid:10}{prec:>14}{hi.sum() / m.sum():>10.1%}"
              f"{good[lo].mean():>13.1%}")


def calibrate_and_report():
    """LOMO calibration report + ship the all-data model."""
    rows = collect()
    mids = sorted({r["match"] for r in rows})
    Xall = np.stack([r["x"] for r in rows])
    midx = [FEATURES.index(k) for k in MODEL_FEATURES]
    X = Xall[:, midx]
    gate = ((Xall[:, FEATURES.index("serve_launch_plausible")] == 1.0)
            & (Xall[:, FEATURES.index("xr_pre_serve")] < 2)
            & (Xall[:, FEATURES.index("rally_spineless")] == 0.0))
    d = np.array([r["d_tok"] for r in rows])
    match_of = np.array([r["match"] for r in rows])
    good = (d <= GOOD_EDITS).astype(float)
    strict = (d <= STRICT_EDITS).astype(float)

    print(f"confidence calibration — {len(rows)} scored points; HIGH bar = "
          f"within {GOOD_EDITS} token edits (base rate {good.mean():.1%})\n")

    # ---- the shipped bar, leave-one-match-out ----
    flags = _lomo_flags(X, good, gate, match_of, mids)
    _tier_table("LOMO (each match scored by a model that never saw it):",
                flags, good, d, match_of, mids, GOOD_EDITS)

    print("\nflag × edit-distance confusion (LOMO, pooled):")
    bins = [("0-1", d <= 1), ("2", d == 2), ("3-5", (d >= 3) & (d <= 5)),
            ("6+", d >= 6)]
    print(f"{'flag':8}" + "".join(f"{b:>8}" for b, _ in bins) + f"{'total':>8}")
    for fl, m in (("high", flags), ("low", ~flags)):
        print(f"{fl:8}" + "".join(f"{int((m & bm).sum()):>8}" for _, bm in bins)
              + f"{int(m.sum()):>8}")

    # ---- the strict bar, on the record: it does not survive LOMO ----
    sflags = _lomo_flags(X, strict, gate, match_of, mids, min_flags=3)
    hi = sflags
    prec = f"{strict[hi].mean():.0%} ({int(strict[hi].sum())}/{hi.sum()})" if hi.sum() else "—"
    print(f"\nstrict <= {STRICT_EDITS}-edit tier (attempted, NOT shipped): "
          f"LOMO precision {prec} at {hi.sum() / len(d):.1%} coverage — "
          f"{len(d)} points at a {strict.mean():.0%} base rate can't support it")

    # ---- the shipped model: fit on all points, threshold by the same rule ----
    Xs, mu, sd = _standardize(X)
    w = _fit_logistic(Xs, good)
    p = _predict(w, Xs)
    t_high = _high_threshold(p, good)
    model = {"features": MODEL_FEATURES, "mu": mu.tolist(), "sd": sd.tolist(),
             "weights": w.tolist(), "t_high": float(t_high),
             "good_edits": GOOD_EDITS,
             "precision_target": PRECISION_TARGET,
             "gate": GATES,
             "n_train": len(rows),
             "note": f"fit on all {len(rows)} points across {len(mids)} "
                     "matches; honest numbers are the LOMO table in "
                     "docs/benchmark.md"}
    MODEL_PATH.write_text(json.dumps(model, indent=1))
    hi = (p >= t_high) & gate
    print(f"\nshipped model (all-data fit): flags {hi.sum()}/{len(rows)} high "
          f"at {good[hi].mean():.0%} in-sample precision, t_high={t_high:.3f}")
    print(f"-> {MODEL_PATH}")
    return flags, d


def load_model():
    return json.loads(MODEL_PATH.read_text())


def score_match(cfg, charts_dir=None):
    """{clip: (flag, probability, signals)} using the shipped model."""
    model = load_model()
    mu = np.array(model["mu"])
    sd = np.array(model["sd"])
    w = np.array(model["weights"])
    sigs = match_signals(cfg, charts_dir)
    out = {}
    for clip, f in sigs.items():
        x = (np.array([f[k] for k in model["features"]]) - mu) / sd
        p = float(1 / (1 + np.exp(-(w[0] + x @ w[1:]))))
        flag = "high" if p >= model["t_high"] and gate_pass(f) else "low"
        out[clip] = (flag, p, f)
    return out
