"""LOMO A/B of repaired launch-plausibility rules — the decisive test.

launch_gate_transfer.py showed the shipped cy-only band mistakes late
WASB serve acquisition for a mid-rally join (t7: 138 launch-inside
clips, 116 with a verified pre-launch baseline stance — a real server,
not a rally in costume). Candidate repairs re-AND the signature with an
independent real-serve observable:

  V0  cy-inside                                (shipped)
  V1  cy-inside AND serve_s < 0.5              (the original time clause)
  V2  cy-inside AND no stance read             (the stance observable)
  V3  cy-inside AND no stance AND serve_s<1.0  (both)

Each variant recomputes serve_launch_plausible everywhere it lives —
the mechanistic gate AND the model feature — then reruns the exact
LOMO loop from courtvision.confidence. The bar: pooled precision holds
(>= the shipped 94%) while t5/t7 coverage recovers. Analysis only;
courtvision/ does not change until a variant survives this table.

collect() output is cached (signals don't depend on the gate rule).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import config
from courtvision.confidence import (FEATURES, GOOD_EDITS, MODEL_FEATURES,
                                    _fit_logistic, _high_threshold,
                                    _predict, _standardize, collect)

CACHE = Path(__file__).resolve().parent.parent / "outputs" / "conf_rows_cache.json"


def load_rows():
    if CACHE.exists():
        blob = json.loads(CACHE.read_text())
        return [{"match": m, "clip": c, "x": np.array(x), "d_tok": d}
                for m, c, x, d in blob]
    rows = collect()
    CACHE.write_text(json.dumps(
        [[r["match"], r["clip"], r["x"].tolist(), r["d_tok"]] for r in rows]))
    return rows


def serve_meta():
    """{(match, clip): (src, launch_cy, serve_s, has_stance)}"""
    meta = {}
    for mid in config.match_ids():
        cfg = config.load(mid)
        for r in csv.DictReader(open(cfg.out_dir / "serves.csv")):
            meta[(mid, r["clip"])] = (
                r.get("src", ""),
                float(r["launch_cy"]) if r.get("launch_cy") else None,
                float(r["serve_s"]) if r.get("serve_s") else None,
                bool(r.get("margin_m")))
    return meta


def plausible(variant, src, cy, s_s, stance, committed):
    """serve_launch_plausible under a candidate rule (1.0/0.0)."""
    if src != "ball" or cy is None:
        return committed          # stance-called / refused: unchanged
    inside = -10.0 < cy < 30.0
    if not inside:
        return 1.0
    if variant == "V0":
        return 0.0
    if variant == "V1":
        return 0.0 if (s_s is not None and s_s < 0.5) else 1.0
    if variant == "V2":
        return 0.0 if not stance else 1.0
    if variant == "V3":
        return 0.0 if (not stance and (s_s is None or s_s < 1.0)) else 1.0
    raise ValueError(variant)


def lomo(X, good, gate, match_of, mids):
    flags = np.zeros(len(good), bool)
    for held in mids:
        tr, te = match_of != held, match_of == held
        Xtr, mu, sd = _standardize(X[tr])
        w = _fit_logistic(Xtr, good[tr])
        t_high = _high_threshold(_predict(w, Xtr), good[tr])
        flags[te] = _predict(w, (X[te] - mu) / sd) >= t_high
    return flags & gate


def main():
    rows = load_rows()
    meta = serve_meta()
    mids = sorted({r["match"] for r in rows})
    Xall0 = np.stack([r["x"] for r in rows])
    d = np.array([r["d_tok"] for r in rows])
    match_of = np.array([r["match"] for r in rows])
    good = (d <= GOOD_EDITS).astype(float)
    i_lp = FEATURES.index("serve_launch_plausible")
    i_xr = FEATURES.index("xr_pre_serve")
    i_sp = FEATURES.index("rally_spineless")
    i_sc = FEATURES.index("serve_committed")
    midx = [FEATURES.index(k) for k in MODEL_FEATURES]

    for variant in ("V0", "V1", "V2", "V3"):
        Xall = Xall0.copy()
        for i, r in enumerate(rows):
            src, cy, s_s, stance = meta.get((r["match"], r["clip"]),
                                            ("", None, None, False))
            Xall[i, i_lp] = plausible(variant, src, cy, s_s, stance,
                                      Xall0[i, i_sc])
        gate = ((Xall[:, i_lp] == 1.0) & (Xall[:, i_xr] < 2)
                & (Xall[:, i_sp] == 0.0))
        flags = lomo(Xall[:, midx], good, gate, match_of, mids)
        hi = flags
        pooled = (f"{good[hi].mean():.0%} ({int(good[hi].sum())}/{hi.sum()})"
                  if hi.sum() else "—")
        print(f"\n{variant}: pooled precision {pooled} at "
              f"{hi.sum() / len(d):.1%} coverage; disasters(6+) "
              f"{int((hi & (d >= 6)).sum())}")
        print(f"  {'match':6}{'precision':>16}{'coverage':>10}")
        for mid in mids:
            m = match_of == mid
            h = m & flags
            prec = (f"{good[h].mean():.0%} ({int(good[h].sum())}/{h.sum()})"
                    if h.sum() else "—  (0/0)")
            print(f"  {mid:6}{prec:>16}{h.sum() / m.sum():>10.1%}")


if __name__ == "__main__":
    main()
