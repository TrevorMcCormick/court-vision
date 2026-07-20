"""Why do t5/t7 starve the HIGH tier? — coverage autopsy, analysis only.

The 7-match recalibration (docs/benchmark.md) ships 99/508 HIGH, but the
three newest matches hold 356/491 scored points and yield almost none of
the flags: LOMO coverage t5 4.2%, t7 7.6% vs t6 28.1%, t4 40.8% — while
t7 posts the BEST structural scorecard on record (rally ±1 83%, server
end 85%). Something suppresses those feeds' flags that is not chart
quality. Suspects, in order:

  1. the mechanistic gates — t5/t7 run serve variant=ball, so
     serve_launch_plausible is a LIVE gate there (on t4's stance-called
     serves it was vacuous); a feed-specific launch_cy artifact would
     zero the tier before the model ever votes
  2. the logistic — 5 of 11 MODEL_FEATURES are serve-derived; a feed
     whose serve detector refuses/mis-times pays five features at once
  3. the threshold — t_high is set by train precision; if the held-out
     feed's p distribution just sits lower, coverage collapses with no
     single culprit

This script re-runs the exact LOMO loop and, for every good-but-LOW
point (d_tok <= 5, not flagged), names which suspect blocked it; then
per-feature standardized-mean deltas x weight show what drags the
held-out logit. Nothing in the pipeline changes.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import config
from courtvision.confidence import (FEATURES, GOOD_EDITS, MODEL_FEATURES,
                                    _fit_logistic, _high_threshold,
                                    _predict, _standardize, collect)

GATE_NAMES = ("serve_launch_plausible", "xr_pre_serve", "rally_spineless")


def gate_bits(xall_row):
    f = {k: xall_row[FEATURES.index(k)] for k in GATE_NAMES}
    return {"launch": f["serve_launch_plausible"] == 1.0,
            "pre_serve": f["xr_pre_serve"] < 2,
            "spine": f["rally_spineless"] == 0.0}


def main():
    rows = collect()
    mids = sorted({r["match"] for r in rows})
    Xall = np.stack([r["x"] for r in rows])
    midx = [FEATURES.index(k) for k in MODEL_FEATURES]
    X = Xall[:, midx]
    d = np.array([r["d_tok"] for r in rows])
    match_of = np.array([r["match"] for r in rows])
    good = (d <= GOOD_EDITS).astype(float)

    print(f"{len(rows)} scored points; base rate (<= {GOOD_EDITS} edits) "
          f"{good.mean():.1%}\n")

    # serve-source mix per match (ball vs players vs refused) — the gate's
    # vacuity boundary
    print("serve src mix + gate pass rates, per match:")
    print(f"{'match':6}{'n':>5}{'launch ok':>11}{'pre-serve ok':>14}"
          f"{'spine ok':>10}{'all gates':>11}")
    for mid in mids:
        m = match_of == mid
        bits = [gate_bits(Xall[i]) for i in np.where(m)[0]]
        n = m.sum()
        lk = sum(b["launch"] for b in bits) / n
        ps = sum(b["pre_serve"] for b in bits) / n
        sp = sum(b["spine"] for b in bits) / n
        al = sum(all(b.values()) for b in bits) / n
        print(f"{mid:6}{n:>5}{lk:>11.1%}{ps:>14.1%}{sp:>10.1%}{al:>11.1%}")

    # the exact LOMO loop, instrumented
    print("\nLOMO autopsy — good points (d<=5) that stayed LOW, by blocker:")
    print(f"{'match':6}{'good':>6}{'flagged':>9}{'gate-blocked':>14}"
          f"{'below t_high':>14}{'t_high':>9}{'p_med(held)':>13}")
    fold_diag = {}
    for held in mids:
        tr, te = match_of != held, match_of == held
        Xtr, mu, sd = _standardize(X[tr])
        w = _fit_logistic(Xtr, good[tr])
        t_high = _high_threshold(_predict(w, Xtr), good[tr])
        p_te = _predict(w, (X[te] - mu) / sd)
        gates_te = np.array([all(gate_bits(Xall[i]).values())
                             for i in np.where(te)[0]])
        flag_te = (p_te >= t_high) & gates_te
        g_te = good[te].astype(bool)
        gate_blocked = g_te & (p_te >= t_high) & ~gates_te
        below = g_te & (p_te < t_high)
        print(f"{held:6}{int(g_te.sum()):>6}{int((flag_te & g_te).sum()):>9}"
              f"{int(gate_blocked.sum()):>14}{int(below.sum()):>14}"
              f"{t_high:>9.3f}{np.median(p_te):>13.3f}")
        # which single gate failed for the gate-blocked good points
        blockers = Counter()
        for i, idx in enumerate(np.where(te)[0]):
            if gate_blocked[i]:
                b = gate_bits(Xall[idx])
                for name, ok in b.items():
                    if not ok:
                        blockers[name] += 1
        # per-feature drag: mean held-out z * weight (negative = pushes LOW)
        z_te = (X[te] - mu) / sd
        contrib = z_te.mean(axis=0) * w[1:]
        drag = sorted(zip(MODEL_FEATURES, contrib), key=lambda t: t[1])
        fold_diag[held] = {"blockers": dict(blockers), "drag": drag[:4],
                           "lift": drag[-2:]}

    print("\nper-fold detail — gate blockers on good points, and the "
          "features dragging the held-out logit down:")
    for mid in mids:
        fd = fold_diag[mid]
        drag_s = ", ".join(f"{k} {v:+.2f}" for k, v in fd["drag"])
        lift_s = ", ".join(f"{k} {v:+.2f}" for k, v in fd["lift"])
        print(f"  {mid}: gate blockers {fd['blockers'] or '{}'}\n"
              f"      drag: {drag_s}\n      lift: {lift_s}")


if __name__ == "__main__":
    main()
