"""Set-prior parity audit, all seven matches — is t5's truth bug alone?

The t5 diagnosis (diag2_t5_server_end.py) found the server-end TRUTH
inverted for set-2 odd-game-sum states: set_priors "1,0" was 9 where
the odd-total set 1 (6-3) demands 10 — with 9, the formula hands set-2
games 0 and 1 the SAME end, though a changeover sits between them.
Corrected: t5 server end 53/71 -> 59/71. t7 was hand-corrected for the
same class at staging (25 -> 24, the tiebreak-set parity lesson).

This script sweeps every match for the signature WITHOUT trusting my
arithmetic: for each (set-state, within-set game-sum parity) cell it
crosstabs detector-vs-truth agreement. A wrong prior flips truth for
exactly one parity class of one state, so agreement INVERTS there
(t5's tell was 2/10 odd vs 12/3 even). Detector noise can't produce
that pattern — it dilutes both cells equally. Cells with n < 5 are
printed but not judged. Analysis only; nothing changes.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import config
from courtvision.evaluate import OTHER, p1_end
from courtvision.mcp import parse_mcp


def main():
    for mid in config.match_ids():
        cfg = config.load(mid)
        ev = cfg.eval
        mapd = {r["clip"]: r for r in csv.DictReader(open(ev.mcp_map))}
        align = {r["clip"]: r for r in csv.DictReader(open(ev.alignment))}
        match = {r["clip"]: r for r in csv.DictReader(
            open(cfg.charts_dir / "match_chart_v2.csv"))}
        cells = {}
        for clip, mc in match.items():
            m, a = mapd.get(clip), align.get(clip)
            if not m or not a or m["status"] != "matched":
                continue
            if mc["server_used"] not in ("near", "far"):
                continue
            n_end = p1_end(a, ev)
            true_end = n_end if m["svr"] == "1" else OTHER[n_end]
            state = (a["set1"], a["set2"])
            par = (int(a["gm1"]) + int(a["gm2"])) % 2
            key = (state, par)
            ok = mc["server_used"] == true_end
            cells.setdefault(key, [0, 0])
            cells[key][0] += ok
            cells[key][1] += 1
        print(f"\n{mid} — detector-vs-truth agreement by (set-state, "
              f"within-set game parity):")
        flagged = []
        for (state, par), (k, n) in sorted(cells.items()):
            rate = k / n
            mark = ""
            if n >= 5 and rate <= 0.35:
                mark = "  <-- INVERSION SUSPECT"
                flagged.append((state, par, k, n))
            print(f"  state {state[0]},{state[1]}  gmsum%2={par}: "
                  f"{k}/{n} ({rate:.0%}){mark}")
        # a real prior bug flips ONE parity cell while its sibling stays
        # high — call that out explicitly
        for state, par, k, n in flagged:
            sib = cells.get((state, 1 - par))
            if sib and sib[1] >= 5 and sib[0] / sib[1] >= 0.65:
                print(f"  ** {mid} state {state[0]},{state[1]}: parity "
                      f"{par} at {k}/{n} vs sibling {sib[0]}/{sib[1]} — "
                      f"the t5-class signature. Verify on pixels before "
                      f"touching the prior.")


if __name__ == "__main__":
    main()
