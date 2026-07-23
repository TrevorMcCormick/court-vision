"""The scorecard — auto-charts vs Match Charting Project ground truth.

Consolidation of the four t*w eval scripts (2026-07-10): one loop,
per-match structure (changeover-parity set priors, tiebreak states,
the staged START_END) moved into MatchConfig.eval. Per aligned clip,
compares:

  server end   chart's final server (after overrides) vs MCP Svr mapped
               to an end via changeover parity
  rally length shots in our chart vs strokes in the MCP string (serve
               included; if the 1st serve was a fault, the point was
               played on the 2nd string)
  serve zone   our 4/5/6 vs the MCP serve digit, where both exist
  letters      our committed f/b letters vs the MCP stroke SIDE at the
               same shot index — but the index is only trustworthy when
               rally lengths AGREE (t1 points 01/04 taught that), so
               the honest denominator is length-matched clips only
  ending       TYPE only (winner / net / wide / deep); forced vs
               unforced attribution is charter judgment
  acceptance   ≤1 token edit vs MCP (courtvision.mcp) — the north star

Usage:
    uv run python -m courtvision eval t3
"""

import csv
from collections import Counter

from .mcp import (FH_SIDE, BH_SIDE, mcp_point_tokens, chart_point_tokens,
                  token_levenshtein, parse_mcp, mcp_ending_type,
                  our_ending_type)

OTHER = {"near": "far", "far": "near"}


def p1_end(a, ev):
    """Which end (near/far) MCP player 1 occupies for this clip's game.

    Changeover parity: ends swap after game 1 and every 2 games after
    (swap-units), with per-set game priors from the config (read off
    the MCP rows at staging). Tiebreak clips add end changes every 6
    points in the configured tiebreak set-states."""
    g = int(a["gm1"]) + int(a["gm2"])
    g += ev.prior(int(a["set1"]), int(a["set2"]))
    swaps = (g + 1) // 2
    if (a["gm1"] == "6" and a["gm2"] == "6"
            and ev.is_tiebreak_state(int(a["set1"]), int(a["set2"]))):
        p1, p2 = a["pts"].split("-")
        swaps += (int(p1) + int(p2)) // 6
    return OTHER[ev.start_end] if swaps % 2 else ev.start_end


def evaluate(cfg, charts_dir=None, verbose=True):
    """Score one match's charts. Returns (tally, records); records carry
    per-point token distances for the confidence layer."""
    ev = cfg.eval
    charts_dir = charts_dir or cfg.charts_dir
    mapd = {r["clip"]: r for r in csv.DictReader(open(ev.mcp_map))}
    align = {r["clip"]: r for r in csv.DictReader(open(ev.alignment))}
    match = {r["clip"]: r
             for r in csv.DictReader(open(charts_dir / "match_chart_v2.csv"))}

    tally = {"server": [0, 0], "rally_pm1": [0, 0], "serve_zone": [0, 0],
             "letters_match": 0, "letters_mirror": 0, "letters_total": 0,
             "letters_al_match": 0, "letters_al_total": 0,
             "ending": [0, 0], "ending_committed": 0, "accept": [0, 0],
             "ending_conf": {}}    # true type -> Counter(our type); '?'=uncommitted
    records = []
    if verbose:
        print(f"{'clip':14} {'srv':>3}{'✓':2} {'len ours/mcp':>13} "
              f"{'zone o/m':>9}  letters(ours vs mcp-side)")
    for clip, mc in match.items():
        m, a = mapd[clip], align[clip]
        if m["status"] != "matched":
            if verbose:
                print(f"{clip:14} (ambiguous MCP row, skipped)")
            continue
        serve_d, strokes, played = parse_mcp(m["first"], m["second"])
        n_end = p1_end(a, ev)
        true_end = n_end if m["svr"] == "1" else OTHER[n_end]

        ok_srv = mc["server_used"] == true_end
        tally["server"][0] += ok_srv
        tally["server"][1] += 1

        shots = list(csv.DictReader(open(charts_dir / f"chart2_{clip}.csv")))
        ours_len = len(shots)
        mcp_len = 1 + len(strokes)          # serve + rally strokes
        ok_len = abs(ours_len - mcp_len) <= 1
        tally["rally_pm1"][0] += ok_len
        tally["rally_pm1"][1] += 1

        zone_pair = ""
        srow = next((s for s in shots if s["is_serve"] == "True"), None)
        if srow and srow["zone"] not in ("?", "") and serve_d != "?":
            ok_z = srow["zone"] == serve_d
            tally["serve_zone"][0] += ok_z
            tally["serve_zone"][1] += 1
            zone_pair = f"{srow['zone']}/{serve_d}{'✓' if ok_z else '✗'}"

        aligned = ours_len == mcp_len
        lets = []
        for k, sh in enumerate(shots):
            if sh["is_serve"] == "True" or sh["letter"] in ("?", ""):
                continue
            mcp_idx = k - 1                  # strokes[] excludes the serve
            if mcp_idx < 0 or mcp_idx >= len(strokes):
                continue
            mcp_side = ("f" if strokes[mcp_idx] in FH_SIDE
                        else "b" if strokes[mcp_idx] in BH_SIDE else "?")
            if mcp_side == "?":
                continue
            tally["letters_total"] += 1
            hit = sh["letter"] == mcp_side
            if aligned:
                tally["letters_al_total"] += 1
                tally["letters_al_match"] += hit
            if hit:
                tally["letters_match"] += 1
                lets.append(f"{sh['letter']}={mcp_side}")
            else:
                tally["letters_mirror"] += 1
                lets.append(f"{sh['letter']}≠{mcp_side}")

        # acceptance — the north star: ≤1 token edit vs MCP
        mcp_toks = mcp_point_tokens(played)
        our_toks = chart_point_tokens(shots, mc.get("ending", "?"))
        d_tok = token_levenshtein(mcp_toks, our_toks)
        tally["accept"][0] += d_tok <= 1
        tally["accept"][1] += 1

        ours_end = our_ending_type(mc.get("ending", "?"))
        true_end_t = mcp_ending_type(played)
        if true_end_t != "?":
            tally["ending_conf"].setdefault(true_end_t, Counter())[ours_end] += 1
        end_pair = ""
        if ours_end != "?":
            tally["ending_committed"] += 1
            if true_end_t != "?":
                ok_e = ours_end == true_end_t
                tally["ending"][0] += ok_e
                tally["ending"][1] += 1
                end_pair = f"end {ours_end}/{true_end_t}{'✓' if ok_e else '✗'}"

        if verbose:
            print(f"{clip:14} {mc['server_used'][:3]:>3}{'✓' if ok_srv else '✗':2} "
                  f"{ours_len:>5}/{mcp_len:<7} {zone_pair:>9} {end_pair:>9}  "
                  f"{' '.join(lets)}   [{played[:26]}]")

        records.append({"clip": clip, "d_tok": d_tok,
                        "mcp_toks": mcp_toks, "our_toks": our_toks,
                        "played": played, "aligned": aligned,
                        "mcp_pt": m["mcp_pt"], "svr": m["svr"]})

    if verbose:
        print(f"\n=== scorecard ({ev.title}) ===")
        s = tally
        print(f"server end       : {s['server'][0]}/{s['server'][1]}")
        print(f"rally len ±1     : {s['rally_pm1'][0]}/{s['rally_pm1'][1]}")
        print(f"serve zone       : {s['serve_zone'][0]}/{s['serve_zone'][1]}")
        print(f"letters (all)    : {s['letters_match']}/{s['letters_total']}"
              f"   (index unreliable when lengths differ)")
        print(f"letters (aligned): {s['letters_al_match']}/{s['letters_al_total']}"
              f"   <- length-matched clips only")
        print(f"ending type      : {s['ending'][0]}/{s['ending'][1]}"
              f"   ({s['ending_committed']} committed)")
        print(f"acceptance ≤1edit: {s['accept'][0]}/{s['accept'][1]}"
              f"   <- token-level point acceptance, the north star")
    return tally, records
