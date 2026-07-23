"""Wiring measurement harness — the before/after receipt for roadmap #6.

Charts + evals every benchmark match with the CURRENT code, aggregates the
per-field scorecard AND a wide/deep ending breakdown, and writes a labelled
JSON snapshot. Run it once before wiring a helper and once after; --diff
prints the delta table.

    uv run python experiments/wiring_measure.py snapshot before
    uv run python experiments/wiring_measure.py snapshot after_court
    uv run python experiments/wiring_measure.py diff before after_court

Snapshots land in outputs/diag/wiring_<label>.json (gitignored); the
committed receipt is whatever summary gets written into LOG.md / docs.
"""

import json
import sys
from collections import Counter

from courtvision import chart, config, evaluate
from courtvision.config import ROOT

MATCHES = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "g1"]
OUT = ROOT / "outputs" / "diag"
FIELDS = ["server", "rally_pm1", "serve_zone", "ending", "accept"]


def snapshot(label, matches=MATCHES, rechart=True, force=None):
    agg = {f: [0, 0] for f in FIELDS}
    agg["letters_al"] = [0, 0]
    conf = {}                       # true ending type -> Counter(our type)
    per_match = {}
    for mid in matches:
        cfg = config.load(mid)
        for k, v in (force or {}).items():   # override staging flags to A/B a helper
            setattr(cfg.staging, k, v)
        if rechart:
            chart.chart_match(cfg, quiet=True)
        tally, _ = evaluate.evaluate(cfg, verbose=False)
        for f in FIELDS:
            agg[f][0] += tally[f][0]
            agg[f][1] += tally[f][1]
        agg["letters_al"][0] += tally["letters_al_match"]
        agg["letters_al"][1] += tally["letters_al_total"]
        for t, c in tally["ending_conf"].items():
            conf.setdefault(t, Counter()).update(c)
        per_match[mid] = {
            "server": tally["server"], "ending": tally["ending"],
            "accept": tally["accept"],
            "ending_conf": {t: dict(c) for t, c in tally["ending_conf"].items()},
        }
    snap = {"label": label, "matches": matches,
            "agg": agg, "letters_al": agg["letters_al"],
            "ending_conf": {t: dict(c) for t, c in conf.items()},
            "per_match": per_match}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"wiring_{label}.json"
    path.write_text(json.dumps(snap, indent=2))
    _print_snap(snap)
    print(f"\n[saved] {path}")
    return snap


def _pct(pair):
    a, b = pair
    return f"{a}/{b} ({100*a/b:.0f}%)" if b else f"{a}/{b} (—)"


def _wide_deep(conf):
    """Recall on out-balls: true type in {w,d,x}, did we say the same?"""
    got = tot = 0
    lines = []
    for t in ("w", "d", "x"):
        c = Counter(conf.get(t, {}))
        n = sum(c.values())
        hit = c.get(t, 0)
        got += hit
        tot += n
        if n:
            lines.append(f"    {t}: {hit}/{n} recall   calls={dict(c)}")
    return got, tot, lines


def _print_snap(snap):
    print(f"=== snapshot: {snap['label']} ({len(snap['matches'])} matches) ===")
    for f in FIELDS:
        print(f"  {f:12}: {_pct(snap['agg'][f])}")
    print(f"  {'letters_al':12}: {_pct(snap['letters_al'])}")
    got, tot, lines = _wide_deep(snap["ending_conf"])
    print(f"  wide/deep    : {_pct([got, tot])}   <- the blind spot")
    for ln in lines:
        print(ln)


def diff(a_label, b_label):
    a = json.loads((OUT / f"wiring_{a_label}.json").read_text())
    b = json.loads((OUT / f"wiring_{b_label}.json").read_text())
    print(f"=== diff: {a_label} -> {b_label} ===")
    for f in FIELDS + ["letters_al"]:
        ka = a["agg"][f] if f in a["agg"] else a[f]
        kb = b["agg"][f] if f in b["agg"] else b[f]
        pa = 100 * ka[0] / ka[1] if ka[1] else 0
        pb = 100 * kb[0] / kb[1] if kb[1] else 0
        arrow = "→" if abs(pb - pa) < 0.5 else ("↑" if pb > pa else "↓")
        print(f"  {f:12}: {_pct(ka)}  {arrow}  {_pct(kb)}   ({pb-pa:+.0f}pp)")
    ga, ta, _ = _wide_deep(a["ending_conf"])
    gb, tb, _ = _wide_deep(b["ending_conf"])
    print(f"  wide/deep   : {_pct([ga, ta])}  ->  {_pct([gb, tb])}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "snapshot":
        # optional trailing flag names force staging flags on, e.g.
        #   snapshot after_landing landing_race
        force = {k: True for k in sys.argv[3:]}
        snapshot(sys.argv[2], force=force)
    elif len(sys.argv) >= 4 and sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
