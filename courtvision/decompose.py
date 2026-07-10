"""Acceptance decomposition — where the mean token edits live.

Pure analysis over the chart CSVs and MCP strings; nothing in the
pipeline changes. Lifted from experiments/mcp_decompose.py (2026-07-10),
reading matches through the package configs:

  1. edit-type attribution: the token Levenshtein backtraced to an
     alignment, every edit binned (insertion, deletion, substitution
     split by component);
  2. per-component accuracy on length-matched points;
  3. the edit-effort curve: acceptance at <=1/2/3/5 token edits, plus
     a structural variant (shot count + letters only);
  4. direction-digit attempt/accuracy rates;
  5. counterfactual acceptance headroom: fix ONE component to MCP
     truth at aligned positions, re-score.

Usage:
    uv run python -m courtvision decompose
"""

import csv

from . import config
from .mcp import (mcp_point_tokens, chart_point_tokens, token_levenshtein,
                  backtrace, classify_sub, tok_kind, structural,
                  counterfactual)

SUB_CATS = ["sub_serve_zone", "sub_letter", "sub_direction",
            "sub_letter+dir", "sub_ending", "sub_crosstype"]
CATS = SUB_CATS + ["ins", "del"]
THRESHOLDS = [1, 2, 3, 5]
CF_ROWS = [("serve zone perfect", "serve"),
           ("letters perfect", "letters"),
           ("directions perfect (attempted only)", "dirs_attempted"),
           ("directions perfect (all shots)", "dirs_all"),
           ("endings perfect", "endings"),
           ("letters + directions perfect", "letters+dirs_all"),
           ("letters + dirs + endings perfect", "letters+dirs_all+endings"),
           ("all components perfect (structure-only residual)",
            "serve+letters+dirs_all+endings")]


def load_match(cfg):
    """[(clip, mcp_toks, our_toks, shots)] for scored (matched) points."""
    mapd = {r["clip"]: r for r in csv.DictReader(open(cfg.eval.mcp_map))}
    chart = cfg.charts_dir
    match = {r["clip"]: r
             for r in csv.DictReader(open(chart / "match_chart_v2.csv"))}
    pts = []
    for clip, mc in match.items():
        m = mapd[clip]
        if m["status"] != "matched":
            continue
        played = m["second"] if m["second"].strip() else m["first"]
        shots = list(csv.DictReader(open(chart / f"chart2_{clip}.csv")))
        mcp = mcp_point_tokens(played)
        ours = chart_point_tokens(shots, mc.get("ending", "?"))
        pts.append((clip, mcp, ours, shots))
    return pts


def report(match_ids=None):
    match_ids = match_ids or config.match_ids()
    cfgs = {t: config.load(t) for t in match_ids}
    labels = {t: cfgs[t].title for t in match_ids}
    per_match = {t: load_match(cfgs[t]) for t in match_ids}
    all_pts = [p for t in match_ids for p in per_match[t]]
    n_all = len(all_pts)

    # ---------- 1. edit-type decomposition ----------
    print(f"=== 1. edit decomposition ({n_all} points) — mean edits/point ===\n")
    hdr = f"{'category':18}" + "".join(f"{labels[t]:>14}" for t in match_ids) + f"{'overall':>10}"
    print(hdr)
    counts = {t: {c: 0 for c in CATS} for t in match_ids}
    dists = {t: [] for t in match_ids}
    for t in match_ids:
        for clip, mcp, ours, _ in per_match[t]:
            d, ops = backtrace(mcp, ours)
            dists[t].append(d)
            for op, mt, ot in ops:
                if op == "sub":
                    counts[t][classify_sub(mt, ot)] += 1
                elif op in ("ins", "del"):
                    counts[t][op] += 1
    for c in CATS:
        row = f"{c:18}"
        tot = 0
        for t in match_ids:
            n = len(per_match[t])
            row += f"{counts[t][c] / n:>14.2f}"
            tot += counts[t][c]
        row += f"{tot / n_all:>10.2f}"
        print(row)
    row = f"{'TOTAL (mean dist)':18}"
    for t in match_ids:
        row += f"{sum(dists[t]) / len(dists[t]):>14.2f}"
    row += f"{sum(sum(dists[t]) for t in match_ids) / n_all:>10.2f}"
    print(row)

    # refusal vs wrong-commit split inside the substitutions
    ref = {"serve_zone": [0, 0], "letter": [0, 0], "direction": [0, 0],
           "ending": [0, 0]}
    for t in match_ids:
        for clip, mcp, ours, _ in per_match[t]:
            _, ops = backtrace(mcp, ours)
            for op, mt, ot in ops:
                if op != "sub" or tok_kind(mt) != tok_kind(ot):
                    continue
                k = tok_kind(mt)
                if k == "serve":
                    ref["serve_zone"][ot[1] == "?"] += 1
                elif k == "ending":
                    ref["ending"][ot == "?"] += 1
                else:
                    if mt[0] != ot[0]:
                        ref["letter"][ot[0] == "?"] += 1
                    if mt[1] != ot[1]:
                        ref["direction"][ot[1] == "?"] += 1
    print("wrong component: committed-wrong vs refused-'?' — " + ", ".join(
        f"{k}={v[0]}w/{v[1]}r" for k, v in ref.items()))

    # ---------- 2. edit-effort curves ----------
    print("\n=== 2. edit-effort curve — acceptance at <=k token edits ===\n")

    def curve(tok_fn, title):
        print(title)
        print(f"{'<=k':>4}" + "".join(f"{labels[t]:>14}" for t in match_ids) + f"{'overall':>16}")
        for k in THRESHOLDS:
            row = f"{k:>4}"
            tot = 0
            for t in match_ids:
                n = sum(token_levenshtein(*tok_fn(p)) <= k for p in per_match[t])
                tot += n
                row += f"{f'{n}/{len(per_match[t])}':>14}"
            row += f"{f'{tot}/{n_all}':>10} ({100 * tot / n_all:4.1f}%)"
            print(row)
        print()

    curve(lambda p: (p[1], p[2]), "full tokens (the real metric):")
    curve(lambda p: (structural(p[1]), structural(p[2])),
          "structural only (shot count + letters; zones/dirs/endings ignored):")

    # ---------- 3. counterfactual acceptance headroom ----------
    print("=== 3. acceptance headroom — fix ONE component to truth, re-score ===\n")
    base = {t: sum(token_levenshtein(p[1], p[2]) <= 1 for p in per_match[t])
            for t in match_ids}
    print(f"{'counterfactual':50}" + "".join(f"{labels[t]:>14}" for t in match_ids)
          + f"{'overall':>12}{'mean dist':>11}")
    row = f"{'baseline (as charted)':50}"
    for t in match_ids:
        row += f"{f'{base[t]}/{len(per_match[t])}':>14}"
    tot = sum(base.values())
    md = sum(sum(dists[t]) for t in match_ids) / n_all
    print(row + f"{f'{tot}/{n_all}':>8} ({100 * tot / n_all:4.1f}%){md:>7.2f}")
    for label, fix in CF_ROWS:
        row = f"{label:50}"
        tot = 0
        dsum = 0
        for t in match_ids:
            n = 0
            for clip, mcp, ours, _ in per_match[t]:
                cf = counterfactual(mcp, ours, fix)
                d = token_levenshtein(mcp, cf)
                dsum += d
                n += d <= 1
            tot += n
            row += f"{f'{n}/{len(per_match[t])}':>14}"
        print(row + f"{f'{tot}/{n_all}':>8} ({100 * tot / n_all:4.1f}%){dsum / n_all:>7.2f}")
