"""T4 — align extracted clips to Match Charting Project rows by score.

Match: 20240713 Wimbledon F, player 1 = Jasmine Paolini,
player 2 = Barbora Krejcikova. Krejcikova won 6-2 2-6 6-4 — the bug's
green sets column reading 1-1 in the third set is what corrected the
"straight sets" memory; derive structure from the data, not the vibe.
The Wimbledon bug is KREJCIKOVA-top (player 2 top), so contact-sheet
reads were swapped into PAOLINI-first (MCP player-1) order when
writing t4_clip_alignment.csv. MCP's Pts is SERVER-first: flip
whenever Svr == 2.

Reads data/mcp/t4_clip_alignment.csv, writes data/mcp/t4_mcp_map.csv.

Usage:
    uv run experiments/t4_align_mcp.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"


def main():
    rows, seen = [], set()
    for r in csv.DictReader(open(DATA / "points_20240713_krejcikova_paolini.csv")):
        if r["Pt"] not in seen:
            seen.add(r["Pt"])
            rows.append(r)

    align = list(csv.DictReader(open(DATA / "t4_clip_alignment.csv")))
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

    with open(DATA / "t4_mcp_map.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    n = sum(1 for r in out if r["status"] == "matched")
    print(f"\n{n}/{len(out)} clips uniquely matched -> {DATA / 't4_mcp_map.csv'}")


if __name__ == "__main__":
    main()
