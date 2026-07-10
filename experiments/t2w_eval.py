"""T2 — the control scorecard (day, both right-handed): t1_eval, t2 paths.

Per aligned clip, compares:
  server end   chart's final server (after overrides) vs MCP Svr mapped
               to an end via changeover parity (Nadal starts far —
               established 11-3 against the alternative)
  rally length shots in our chart vs strokes in the MCP string (serve
               included; if the 1st serve was a fault, the point was
               played on the 2nd string)
  serve zone   our 4/5/6 vs the MCP serve digit, where both exist
  letters      our committed f/b letters vs the MCP stroke SIDE at the
               same shot index (MCP letters map to sides: f,r,v,o,u,l
               forehand-side; b,s,z,p,y,m backhand-side). BOTH t1
               players are LEFT-handed and the frozen letter logic
               assumes right-handers — the honest prediction is that
               committed letters come out systematically MIRRORED.

Usage:
    uv run experiments/t1_eval.py
"""

import csv
from pathlib import Path

from mcp_accept import mcp_point_tokens, chart_point_tokens, token_levenshtein

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"
CHART = ROOT / "outputs" / "t2" / "charts_wasb"

FH_SIDE = set("frvoul")
BH_SIDE = set("bszpym")
SHOT_CHARS = FH_SIDE | BH_SIDE | set("hijkt")


def parse_mcp(first, second):
    """Return (serve_digit, [stroke letters], played_string)."""
    s = second if second.strip() else first
    serve = s[0] if s and s[0] in "0456" else "?"
    strokes = [c for c in s[1:] if c in SHOT_CHARS]
    return serve, strokes, s


def mcp_ending_type(played):
    """Ending TYPE only: * winner, or n/w/d/x error kind. Forced vs
    unforced (# vs @) is the charter's judgment and is not compared."""
    if played.endswith("*"):
        return "*"
    t = played.rstrip("@#!")
    return t[-1] if t and t[-1] in "nwdx" else "?"


def our_ending_type(ending):
    if ending == "*":
        return "*"
    return ending[0] if ending and ending[0] in "nwdx" else "?"


def federer_end(a):
    g = int(a["gm1"]) + int(a["gm2"])
    prior = 0
    if a["set1"] == "1" or a["set2"] == "1":
        prior += 9            # set 1 went 6-3 (9 games)
    g += prior
    swaps = (g + 1) // 2
    if a["gm1"] == "6" and a["gm2"] == "6":
        p1, p2 = a["pts"].split("-")
        swaps += (int(p1) + int(p2)) // 6
    start = "near"            # established by parity vote 5-1 at staging
    other = {"near": "far", "far": "near"}
    return other[start] if swaps % 2 else start


def main():
    mapd = {r["clip"]: r for r in csv.DictReader(open(DATA / "t2_mcp_map.csv"))}
    align = {r["clip"]: r for r in csv.DictReader(open(DATA / "t2_clip_alignment.csv"))}
    match = {r["clip"]: r for r in csv.DictReader(open(CHART / "match_chart_v2.csv"))}

    tally = {"server": [0, 0], "rally_pm1": [0, 0], "serve_zone": [0, 0],
             "letters_match": 0, "letters_mirror": 0, "letters_total": 0,
             "letters_al_match": 0, "letters_al_total": 0,
             "ending": [0, 0], "ending_committed": 0, "accept": [0, 0]}
    print(f"{'clip':14} {'srv':>3}{'✓':2} {'len ours/mcp':>13} {'zone o/m':>9}  letters(ours vs mcp-side)")
    for clip, mc in match.items():
        m, a = mapd[clip], align[clip]
        if m["status"] != "matched":
            print(f"{clip:14} (ambiguous MCP row, skipped)")
            continue
        serve_d, strokes, played = parse_mcp(m["first"], m["second"])
        other = {"near": "far", "far": "near"}
        n_end = federer_end(a)
        true_end = n_end if m["svr"] == "1" else other[n_end]

        ok_srv = mc["server_used"] == true_end
        tally["server"][0] += ok_srv
        tally["server"][1] += 1

        shots = list(csv.DictReader(open(CHART / f"chart2_{clip}.csv")))
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

        # letters compared per shot index — but the index is only
        # trustworthy when rally lengths AGREE (t1 points 01/04 taught
        # that: both "contact-side misses" were index misalignment from
        # a missed/phantom shot). aligned = length-matched clips only.
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

        # acceptance — the north star: ≤1 token edit vs MCP (mcp_accept.py)
        d_tok = token_levenshtein(
            mcp_point_tokens(played),
            chart_point_tokens(shots, mc.get("ending", "?")))
        tally["accept"][0] += d_tok <= 1
        tally["accept"][1] += 1

        ours_end = our_ending_type(mc.get("ending", "?"))
        true_end_t = mcp_ending_type(played)
        end_pair = ""
        if ours_end != "?":
            tally["ending_committed"] += 1
            if true_end_t != "?":
                ok_e = ours_end == true_end_t
                tally["ending"][0] += ok_e
                tally["ending"][1] += 1
                end_pair = f"end {ours_end}/{true_end_t}{'✓' if ok_e else '✗'}"

        print(f"{clip:14} {mc['server_used'][:3]:>3}{'✓' if ok_srv else '✗':2} "
              f"{ours_len:>5}/{mcp_len:<7} {zone_pair:>9} {end_pair:>9}  "
              f"{' '.join(lets)}   [{played[:26]}]")

    print("\n=== scorecard (control: day, right-handed) ===")
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


if __name__ == "__main__":
    main()
