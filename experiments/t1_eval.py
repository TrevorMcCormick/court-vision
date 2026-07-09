"""T1 — the scorecard: frozen auto-charts vs Match Charting Project.

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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"
CHART = ROOT / "outputs" / "t1" / "charts"

FH_SIDE = set("frvoul")
BH_SIDE = set("bszpym")
SHOT_CHARS = FH_SIDE | BH_SIDE | set("hijkt")


def parse_mcp(first, second):
    """Return (serve_digit, [stroke letters], played_string)."""
    s = second if second.strip() else first
    serve = s[0] if s and s[0] in "0456" else "?"
    strokes = [c for c in s[1:] if c in SHOT_CHARS]
    return serve, strokes, s


def nadal_end(a):
    g = int(a["gm1"]) + int(a["gm2"])
    prior = 0
    if a["set1"] == "1" or a["set2"] == "1":
        prior += 9            # set 1 went 6-3
    if a["set1"] == "1" and a["set2"] == "1":
        prior += 10           # set 2 went 4-6
    g += prior
    swaps = (g + 1) // 2
    if a["gm1"] == "6" and a["gm2"] == "6":
        p1, p2 = a["pts"].split("-")
        swaps += (int(p1) + int(p2)) // 6
    start = "far"             # established: Nadal began the match far-end
    other = {"near": "far", "far": "near"}
    return other[start] if swaps % 2 else start


def main():
    mapd = {r["clip"]: r for r in csv.DictReader(open(DATA / "t1_mcp_map.csv"))}
    align = {r["clip"]: r for r in csv.DictReader(open(DATA / "t1_clip_alignment.csv"))}
    match = {r["clip"]: r for r in csv.DictReader(open(CHART / "match_chart_v2.csv"))}

    tally = {"server": [0, 0], "rally_pm1": [0, 0], "serve_zone": [0, 0],
             "letters_match": 0, "letters_mirror": 0, "letters_total": 0}
    print(f"{'clip':14} {'srv':>3}{'✓':2} {'len ours/mcp':>13} {'zone o/m':>9}  letters(ours vs mcp-side)")
    for clip, mc in match.items():
        m, a = mapd[clip], align[clip]
        if m["status"] != "matched":
            print(f"{clip:14} (ambiguous MCP row, skipped)")
            continue
        serve_d, strokes, played = parse_mcp(m["first"], m["second"])
        other = {"near": "far", "far": "near"}
        n_end = nadal_end(a)
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
            if sh["letter"] == mcp_side:
                tally["letters_match"] += 1
                lets.append(f"{sh['letter']}={mcp_side}")
            else:
                tally["letters_mirror"] += 1
                lets.append(f"{sh['letter']}≠{mcp_side}")

        print(f"{clip:14} {mc['server_used'][:3]:>3}{'✓' if ok_srv else '✗':2} "
              f"{ours_len:>5}/{mcp_len:<7} {zone_pair:>9}  {' '.join(lets)}"
              f"   [{played[:26]}]")

    print("\n=== scorecard (frozen pipeline, lefty night match) ===")
    s = tally
    print(f"server end     : {s['server'][0]}/{s['server'][1]}")
    print(f"rally len ±1   : {s['rally_pm1'][0]}/{s['rally_pm1'][1]}")
    print(f"serve zone     : {s['serve_zone'][0]}/{s['serve_zone'][1]}")
    print(f"letters exact  : {s['letters_match']}/{s['letters_total']}")
    print(f"letters MIRROR : {s['letters_mirror']}/{s['letters_total']}"
          f"   <- right-hand assumption vs two lefties")


if __name__ == "__main__":
    main()
