"""T2 — align extracted clips to Match Charting Project rows by score.

Same join as t1 with one twist at the transcription step: this bug is
HAASE-top (t1's was Nadal-top = player-1-top), so the contact-sheet
reads were swapped into FEDERER-first (MCP player-1) order when writing
t2_clip_alignment.csv. From there the t1 logic holds verbatim: MCP's
Pts column is SERVER-first, the alignment CSV is player-1-first, so
flip the pts string whenever Svr == 2 (Haase serving).

Match: 20170812 Canada Masters SF, player 1 = Roger Federer,
player 2 = Robin Haase. Federer won 6-3 7-6(5) — set 1 ran 9 games,
set 2 reached 6-6 plus a 12-point tiebreak (derived from the MCP rows,
not assumed).

Reads data/mcp/t2_clip_alignment.csv and writes data/mcp/t2_mcp_map.csv
with the matched MCP point number, server, and ground-truth strings.
Ambiguous clips (recurring deuce states) keep their candidate list.

Usage:
    uv run experiments/t2_align_mcp.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"


def main():
    rows, seen = [], set()
    for r in csv.DictReader(open(DATA / "points_20170812_federer_haase.csv")):
        if r["Pt"] not in seen:
            seen.add(r["Pt"])
            rows.append(r)

    align = list(csv.DictReader(open(DATA / "t2_clip_alignment.csv")))
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

    with open(DATA / "t2_mcp_map.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    n = sum(1 for r in out if r["status"] == "matched")
    print(f"\n{n}/{len(out)} clips uniquely matched -> {DATA / 't2_mcp_map.csv'}")


if __name__ == "__main__":
    main()
