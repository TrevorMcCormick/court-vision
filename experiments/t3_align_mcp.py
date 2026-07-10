"""T3 — align extracted clips to Match Charting Project rows by score.

Match: 20230611 Roland Garros F, player 1 = Casper Ruud,
player 2 = Novak Djokovic. Djokovic won 7-6(1) 6-3 7-5. The RG bug is
DJOKOVIC-top (player 2 top, same trap as t2), so contact-sheet reads
were swapped into RUUD-first (MCP player-1) order when writing
t3_clip_alignment.csv. From there the t1/t2 logic holds verbatim:
MCP's Pts column is SERVER-first, so flip whenever Svr == 2.

The reel's probe splits long rallies and keeps main-camera replays, so
several clips carry the SAME score — they are marked dup in the notes
and all match the same MCP row.

Reads data/mcp/t3_clip_alignment.csv, writes data/mcp/t3_mcp_map.csv.

Usage:
    uv run experiments/t3_align_mcp.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"


def main():
    rows, seen = [], set()
    for r in csv.DictReader(open(DATA / "points_20230611_djokovic_ruud.csv")):
        if r["Pt"] not in seen:
            seen.add(r["Pt"])
            rows.append(r)

    align = list(csv.DictReader(open(DATA / "t3_clip_alignment.csv")))
    out = []
    for a in align:
        cands = []
        for r in rows:
            if not (r["Set1"] == a["set1"] and r["Set2"] == a["set2"]
                    and r["Gm1"] == a["gm1"] and r["Gm2"] == a["gm2"]):
                continue
            bug = a["pts"].upper()          # Federer-first (player 1)
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

    with open(DATA / "t3_mcp_map.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    n = sum(1 for r in out if r["status"] == "matched")
    print(f"\n{n}/{len(out)} clips uniquely matched -> {DATA / 't3_mcp_map.csv'}")


if __name__ == "__main__":
    main()
