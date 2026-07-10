"""Acceptance decomposition — where the 7.18 mean token edits live.

Pure analysis over the existing v5 chart CSVs and MCP strings; no
detector, chart, or eval logic is touched. Reuses the exact tokenizers
from mcp_accept.py (the north-star metric) and adds:

  1. an edit-type attribution: the token Levenshtein is backtraced to
     an alignment, and every edit is binned — insertion, deletion, or
     substitution, with substitutions split by component (serve zone,
     rally letter, rally direction, letter+direction both wrong,
     ending, cross-type structural);
  2. per-component accuracy on length-matched points (positional
     compare, the only honest index);
  3. the edit-effort curve: acceptance at <=1/2/3/5 token edits, plus
     a "structural" variant scored on shot count + letters only
     (serve zones, direction digits, and endings ignored);
  4. direction-digit attempt/accuracy rates (landing detection is
     far-half-only by construction — how sparse are the commits?);
  5. counterfactual acceptance headroom: fix ONE component to MCP
     truth at aligned positions, leave everything else as charted,
     re-score. The gap to 3/135 is that component's headroom.

Usage:
    uv run experiments/mcp_decompose.py
"""

import csv
from pathlib import Path

from mcp_accept import (mcp_point_tokens, chart_point_tokens,
                        token_levenshtein)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"

MATCHES = ["t1", "t2", "t3", "t4"]
LABELS = {"t1": "t1 night", "t2": "t2 ctrl", "t3": "t3 clay",
          "t4": "t4 grass"}

FH_SIDE = set("frvoul")
BH_SIDE = set("bszpym")
SHOT_CHARS = FH_SIDE | BH_SIDE | set("hijkt")

ENDINGS = set("*nwdx?")


def parse_mcp(first, second):
    """Same selection rule as every t*w eval: played string is the
    second serve's if the first faulted."""
    s = second if second.strip() else first
    return s


def tok_kind(t):
    if len(t) == 2 and t[0] == "s":
        return "serve"
    if len(t) == 1 and t in ENDINGS:
        return "ending"
    return "rally"


def backtrace(a, b):
    """Full-matrix token Levenshtein with backtrace.

    Returns (distance, ops) where ops is a list of
    (op, a_tok_or_None, b_tok_or_None); op in match/sub/del/ins.
    'del' = token in a (MCP) our draft lacks; 'ins' = extra token in
    b (ours) MCP lacks. Ties prefer diagonal, then del, then ins —
    deterministic, and diagonal-first keeps substitutions honest."""
    m, n = len(a), len(b)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i][j] = min(D[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                          D[i - 1][j] + 1, D[i][j - 1] + 1)
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub",
                        a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    ops.reverse()
    assert sum(op != "match" for op, _, _ in ops) == D[m][n]
    return D[m][n], ops


def classify_sub(mt, ot):
    """Bin a substitution by which component is wrong."""
    km, ko = tok_kind(mt), tok_kind(ot)
    if km != ko:
        return "sub_crosstype"
    if km == "serve":
        return "sub_serve_zone"
    if km == "ending":
        return "sub_ending"
    letter_ok = mt[0] == ot[0]
    dir_ok = mt[1] == ot[1]
    if letter_ok and not dir_ok:
        return "sub_direction"
    if dir_ok and not letter_ok:
        return "sub_letter"
    return "sub_letter+dir"


SUB_CATS = ["sub_serve_zone", "sub_letter", "sub_direction",
            "sub_letter+dir", "sub_ending", "sub_crosstype"]
CATS = SUB_CATS + ["ins", "del"]


def structural(toks):
    """Shot count + letters only: serve -> 's', rally -> letter,
    ending dropped."""
    out = []
    for t in toks:
        k = tok_kind(t)
        if k == "serve":
            out.append("s")
        elif k == "rally":
            out.append(t[0])
    return out


def counterfactual(mcp, ours, fix):
    """Rebuild our token list with one component set to MCP truth at
    Levenshtein-aligned positions (sub or match pairs). Structure —
    insertions and deletions — is left exactly as charted.

    fix in: serve, letters, dirs_all, dirs_attempted, endings, and
    '+'-joined combos."""
    fixes = set(fix.split("+"))
    _, ops = backtrace(mcp, ours)
    out = []
    for op, mt, ot in ops:
        if ot is None:
            continue                      # del: we can't fix a missing shot
        if op in ("match", "sub") and mt is not None:
            km, ko = tok_kind(mt), tok_kind(ot)
            if km == ko == "serve" and "serve" in fixes:
                ot = mt
            elif km == ko == "ending" and "endings" in fixes:
                ot = mt
            elif km == ko == "rally":
                letter, d = ot[0], ot[1]
                if "letters" in fixes:
                    letter = mt[0]
                if "dirs_all" in fixes or ("dirs_attempted" in fixes
                                           and d != "?"):
                    d = mt[1]
                ot = letter + d
        out.append(ot)
    return out


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


def load_match(t):
    mapd = {r["clip"]: r for r in csv.DictReader(open(DATA / f"{t}_mcp_map.csv"))}
    chart = ROOT / "outputs" / t / "charts_wasb"
    match = {r["clip"]: r for r in csv.DictReader(open(chart / "match_chart_v2.csv"))}
    pts = []
    for clip, mc in match.items():
        m = mapd[clip]
        if m["status"] != "matched":
            continue
        played = parse_mcp(m["first"], m["second"])
        shots = list(csv.DictReader(open(chart / f"chart2_{clip}.csv")))
        mcp = mcp_point_tokens(played)
        ours = chart_point_tokens(shots, mc.get("ending", "?"))
        pts.append((clip, mcp, ours, shots))
    return pts


def main():
    per_match = {t: load_match(t) for t in MATCHES}
    all_pts = [p for t in MATCHES for p in per_match[t]]
    n_all = len(all_pts)

    # ---------- 1. edit-type decomposition ----------
    print(f"=== 1. edit decomposition ({n_all} points) — mean edits/point ===\n")
    hdr = f"{'category':18}" + "".join(f"{LABELS[t]:>10}" for t in MATCHES) + f"{'overall':>10}"
    print(hdr)
    counts = {t: {c: 0 for c in CATS} for t in MATCHES}
    dists = {t: [] for t in MATCHES}
    for t in MATCHES:
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
        for t in MATCHES:
            n = len(per_match[t])
            row += f"{counts[t][c] / n:>10.2f}"
            tot += counts[t][c]
        row += f"{tot / n_all:>10.2f}"
        print(row)
    row = f"{'TOTAL (mean dist)':18}"
    for t in MATCHES:
        row += f"{sum(dists[t]) / len(dists[t]):>10.2f}"
    row += f"{sum(sum(dists[t]) for t in MATCHES) / n_all:>10.2f}"
    print(row)
    print("\nraw edit counts (overall): " + ", ".join(
        f"{c}={sum(counts[t][c] for t in MATCHES)}" for c in CATS))

    # refusal vs wrong-commit split inside the substitutions: a '?'
    # component is a refusal (we didn't commit), anything else is a
    # committed error. Different fixes: recall vs correctness.
    ref = {"serve_zone": [0, 0], "letter": [0, 0], "direction": [0, 0],
           "ending": [0, 0]}
    for t in MATCHES:
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

    # ---------- 2. per-component accuracy, matched-length points ----------
    print("\n=== 2. component accuracy on length-matched points (positional) ===\n")
    print(f"{'component':26}" + "".join(f"{LABELS[t]:>12}" for t in MATCHES) + f"{'overall':>12}")
    acc = {t: {k: [0, 0] for k in
               ["serve_zone", "letter", "direction", "both", "ending"]}
           for t in MATCHES}
    n_matched = {t: 0 for t in MATCHES}
    for t in MATCHES:
        for clip, mcp, ours, _ in per_match[t]:
            if len(mcp) != len(ours):
                continue
            n_matched[t] += 1
            for mt, ot in zip(mcp, ours):
                km, ko = tok_kind(mt), tok_kind(ot)
                if km != ko:
                    continue
                if km == "serve":
                    acc[t]["serve_zone"][0] += mt[1] == ot[1]
                    acc[t]["serve_zone"][1] += 1
                elif km == "ending":
                    acc[t]["ending"][0] += mt == ot
                    acc[t]["ending"][1] += 1
                else:
                    acc[t]["letter"][0] += mt[0] == ot[0]
                    acc[t]["letter"][1] += 1
                    acc[t]["direction"][0] += mt[1] == ot[1]
                    acc[t]["direction"][1] += 1
                    acc[t]["both"][0] += mt == ot
                    acc[t]["both"][1] += 1
    for k, label in [("serve_zone", "serve zone"), ("letter", "rally letter"),
                     ("direction", "rally direction"), ("both", "letter+dir both right"),
                     ("ending", "ending token")]:
        row = f"{label:26}"
        num = den = 0
        for t in MATCHES:
            a, b = acc[t][k]
            row += f"{f'{a}/{b}':>12}"
            num, den = num + a, den + b
        pct = f" ({100 * num / den:.0f}%)" if den else ""
        row += f"{f'{num}/{den}':>12}{pct}"
        print(row)
    print(f"{'(length-matched points)':26}" + "".join(
        f"{f'{n_matched[t]}/{len(per_match[t])}':>12}" for t in MATCHES)
        + f"{f'{sum(n_matched.values())}/{n_all}':>12}")

    # ---------- 3. edit-effort curves ----------
    print("\n=== 3. edit-effort curve — acceptance at <=k token edits ===\n")

    def curve(tok_fn, title):
        print(title)
        print(f"{'<=k':>4}" + "".join(f"{LABELS[t]:>12}" for t in MATCHES) + f"{'overall':>14}")
        for k in THRESHOLDS:
            row = f"{k:>4}"
            tot = 0
            for t in MATCHES:
                n = sum(token_levenshtein(*tok_fn(p)) <= k for p in per_match[t])
                tot += n
                row += f"{f'{n}/{len(per_match[t])}':>12}"
            row += f"{f'{tot}/{n_all}':>10} ({100 * tot / n_all:4.1f}%)"
            print(row)
        print()

    curve(lambda p: (p[1], p[2]), "full tokens (the real metric):")
    curve(lambda p: (structural(p[1]), structural(p[2])),
          "structural only (shot count + letters; zones/dirs/endings ignored):")

    # ---------- 4. direction-digit spot check ----------
    print("=== 4. direction digits: attempt rate + accuracy when attempted ===\n")
    print(f"{'':28}" + "".join(f"{LABELS[t]:>12}" for t in MATCHES) + f"{'overall':>12}")
    att = {t: [0, 0] for t in MATCHES}       # our rally tokens with dir != '?'
    hit = {t: [0, 0] for t in MATCHES}       # aligned pairs, both committed
    mcp_dir = {t: [0, 0] for t in MATCHES}   # MCP rally tokens with a direction
    for t in MATCHES:
        for clip, mcp, ours, _ in per_match[t]:
            for tk in ours:
                if tok_kind(tk) == "rally":
                    att[t][1] += 1
                    att[t][0] += tk[1] != "?"
            for tk in mcp:
                if tok_kind(tk) == "rally":
                    mcp_dir[t][1] += 1
                    mcp_dir[t][0] += tk[1] != "?"
            _, ops = backtrace(mcp, ours)
            for op, mt, ot in ops:
                if (op in ("match", "sub") and mt and ot
                        and tok_kind(mt) == tok_kind(ot) == "rally"
                        and ot[1] != "?" and mt[1] != "?"):
                    hit[t][1] += 1
                    hit[t][0] += ot[1] == mt[1]
    for lab, d in [("our dirs attempted (non-?)", att),
                   ("right when attempted*", hit),
                   ("MCP dirs present (context)", mcp_dir)]:
        row = f"{lab:28}"
        num = den = 0
        for t in MATCHES:
            a, b = d[t]
            row += f"{f'{a}/{b}':>12}"
            num, den = num + a, den + b
        pct = f" ({100 * num / den:.0f}%)" if den else ""
        row += f"{f'{num}/{den}':>12}{pct}"
        print(row)
    print("* aligned rally pairs where both sides commit a digit\n")

    # ---------- 5. counterfactual acceptance headroom ----------
    print("=== 5. acceptance headroom — fix ONE component to truth, re-score ===\n")
    base = {t: sum(token_levenshtein(p[1], p[2]) <= 1 for p in per_match[t])
            for t in MATCHES}
    print(f"{'counterfactual':50}" + "".join(f"{LABELS[t]:>10}" for t in MATCHES)
          + f"{'overall':>12}{'mean dist':>11}")
    row = f"{'baseline (as charted)':50}"
    for t in MATCHES:
        row += f"{f'{base[t]}/{len(per_match[t])}':>10}"
    tot = sum(base.values())
    md = sum(sum(dists[t]) for t in MATCHES) / n_all
    print(row + f"{f'{tot}/{n_all}':>8} ({100 * tot / n_all:4.1f}%){md:>7.2f}")
    for label, fix in CF_ROWS:
        row = f"{label:50}"
        tot = 0
        dsum = 0
        for t in MATCHES:
            n = 0
            for clip, mcp, ours, _ in per_match[t]:
                cf = counterfactual(mcp, ours, fix)
                d = token_levenshtein(mcp, cf)
                dsum += d
                n += d <= 1
            tot += n
            row += f"{f'{n}/{len(per_match[t])}':>10}"
        print(row + f"{f'{tot}/{n_all}':>8} ({100 * tot / n_all:4.1f}%){dsum / n_all:>7.2f}")


if __name__ == "__main__":
    main()
