"""Does the mid-rally-start signature need its TIME clause back?

conf_coverage_autopsy.py named the launch-plausibility gate as the
t5/t7 HIGH-starvation mechanism: pass rates 11.5% (t7) / 16.9% (t5)
vs ~90% (t4), and 69 of t7's 71 gate-blocked good points blocked by
launch alone. Hypothesis: launch_cy is the ball's court-y at the START
of the first sustained crossing run, so a feed where WASB acquires the
serve late (night ball, small far-end ball) reads "inside the court"
on a real serve. The ORIGINAL t3 insight was a conjunction — "a serve
called 0 s into the clip whose launch cy sits INSIDE the court" — and
the shipped gate kept only the cy half.

This script tabulates, for every ball-src serve on every match:
launch-band verdict x serve_s x stance-read presence x point goodness
(d_tok <= 5), to test whether serve_s (and/or the stance read) restores
the gate's transfer without freeing t3's genuine mid-rally joins.
Analysis only; the pipeline does not change.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import config, evaluate

INSIDE = lambda cy: -10.0 < cy < 30.0   # the shipped implausible band


def main():
    print(f"{'match':6}{'grp':>26}{'n':>5}{'good':>6}{'serve_s med':>12}"
          f"{'s<1.0s':>8}{'stance ok':>10}")
    for mid in config.match_ids():
        cfg = config.load(mid)
        _, records = evaluate.evaluate(cfg, verbose=False)
        d_of = {r["clip"]: r["d_tok"] for r in records}
        serves = {r["clip"]: r for r in
                  csv.DictReader(open(cfg.out_dir / "serves.csv"))}
        groups = {"ball launch-inside": [], "ball launch-plausible": [],
                  "non-ball src": []}
        for clip, d in d_of.items():
            s = serves.get(clip)
            if s is None:
                continue
            if s.get("src") != "ball" or not s.get("launch_cy"):
                groups["non-ball src"].append((clip, s, d))
            elif INSIDE(float(s["launch_cy"])):
                groups["ball launch-inside"].append((clip, s, d))
            else:
                groups["ball launch-plausible"].append((clip, s, d))
        for g, items in groups.items():
            if not items:
                continue
            ss = [float(s["serve_s"]) for _, s, _ in items if s.get("serve_s")]
            early = sum(1 for v in ss if v < 1.0)
            stance = sum(1 for _, s, _ in items if s.get("margin_m"))
            good = sum(1 for _, _, d in items if d <= 5)
            print(f"{mid:6}{g:>26}{len(items):>5}{good:>6}"
                  f"{np.median(ss) if ss else float('nan'):>12.2f}"
                  f"{early:>8}{stance:>10}")

    # the candidate repaired rules, scored as mid-rally-join detectors:
    # a clip is a KNOWN join when its chart cannot be the whole point —
    # proxy: launch-inside AND serve_s < 1.0 (the original signature).
    # Report what each rule would do to good points per match.
    print("\nrepair candidates — good points (d<=5) freed / still blocked:")
    print(f"{'match':6}{'rule':>34}{'good freed':>12}{'still blocked':>15}")
    for mid in config.match_ids():
        cfg = config.load(mid)
        _, records = evaluate.evaluate(cfg, verbose=False)
        d_of = {r["clip"]: r["d_tok"] for r in records}
        serves = {r["clip"]: r for r in
                  csv.DictReader(open(cfg.out_dir / "serves.csv"))}
        blocked = [(c, serves[c]) for c, d in d_of.items()
                   if c in serves and serves[c].get("src") == "ball"
                   and serves[c].get("launch_cy")
                   and INSIDE(float(serves[c]["launch_cy"])) and d <= 5]
        for rule, fn in [
            ("cy AND serve_s < 1.0", lambda s: float(s["serve_s"] or 0) < 1.0),
            ("cy AND serve_s < 0.5", lambda s: float(s["serve_s"] or 0) < 0.5),
            ("cy AND no stance read", lambda s: not s.get("margin_m")),
            ("cy AND (s<1.0 OR no stance)",
             lambda s: float(s["serve_s"] or 0) < 1.0 or not s.get("margin_m")),
        ]:
            still = sum(1 for _, s in blocked if fn(s))
            print(f"{mid:6}{rule:>34}{len(blocked) - still:>12}{still:>15}")


if __name__ == "__main__":
    main()
