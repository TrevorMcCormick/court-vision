"""T1 — align extracted clips to Match Charting Project rows by score.

The score bug is the join key: every clip's bug shows (sets, games,
points) and every MCP row carries the score at point start. One trap,
learned the mechanical way: **MCP's Pts column is SERVER-first**, the
broadcast bug is NADAL-first (17/25 matched before the transform, 23/25
after — the misses were exactly the asymmetric scores on Shapovalov's
serve, plus one wrong eyeball read of the serve marker).

Reads data/mcp/t1_clip_alignment.csv (bug transcriptions, Nadal-first)
and writes data/mcp/t1_mcp_map.csv with the matched MCP point number,
server, and ground-truth strings. Ambiguous clips (recurring deuce
states, replay duplicates) keep their candidate list — rally content
will disambiguate them at eval time.

Usage:
    uv run experiments/t1_align_mcp.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"


def main():
    rows, seen = [], set()
    for r in csv.DictReader(open(DATA / "points_20170810_nadal_shapovalov.csv")):
        if r["Pt"] not in seen:
            seen.add(r["Pt"])
            rows.append(r)

    align = list(csv.DictReader(open(DATA / "t1_clip_alignment.csv")))
    out = []
    for a in align:
        cands = []
        for r in rows:
            if not (r["Set1"] == a["set1"] and r["Set2"] == a["set2"]
                    and r["Gm1"] == a["gm1"] and r["Gm2"] == a["gm2"]):
                continue
            bug = a["pts"].upper()          # Nadal-first
            if r["Svr"] == "2" and "-" in bug:
                l, _, rr = bug.partition("-")
                bug = f"{rr}-{l}"           # server-first
            if r["Pts"].upper() == bug:
                cands.append(r)
        rec = {"clip": a["clip"], "note": a["note"]}
        if len(cands) == 1:
            m = cands[0]
            rec.update(mcp_pt=m["Pt"], svr=m["Svr"], first=m["1st"],
                       second=m["2nd"], winner=m["PtWinner"], status="matched",
                       gms=f"{m['Gm1']}-{m['Gm2']}", pts=m["Pts"])
        elif cands:
            rec.update(mcp_pt="|".join(c["Pt"] for c in cands), svr=cands[0]["Svr"],
                       first="", second="", winner="", status="ambiguous")
        else:
            rec.update(mcp_pt="", svr="", first="", second="", winner="",
                       status="NO MATCH")
        out.append(rec)
        print(f"{rec['clip']}: {rec['status']} {rec['mcp_pt']}"
              f"{' 1st=' + rec['first'][:24] if rec['first'] else ''}")

    with open(DATA / "t1_mcp_map.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    n = sum(1 for r in out if r["status"] == "matched")
    print(f"\n{n}/{len(out)} clips uniquely matched -> {DATA / 't1_mcp_map.csv'}")


if __name__ == "__main__":
    main()
