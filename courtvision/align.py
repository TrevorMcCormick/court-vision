"""Align extracted clips to Match Charting Project rows by score.

The score bug is the join key: every clip's bug shows (sets, games,
points) and every MCP row carries the score at point start. One trap,
learned the mechanical way on t1: **MCP's Pts column is SERVER-first**,
the broadcast bug is player-1-first (the alignment CSVs are normalized
to player-1-first at transcription), so flip whenever Svr == 2.

Point-boundary era (t3/t4; order_pass=True): clips are true points and
the reel is CHRONOLOGICAL, so deuce-recurrence ambiguity is resolvable
by order — two 40-AD clips in the same game are the first and second
AD points. Second pass: an ambiguous clip's candidates are filtered to
Pt numbers strictly between the nearest resolved neighbors, then the
smallest survivor is taken, status "matched", note "order-resolved".

Consolidated from the four t*_align_mcp.py scripts (their outputs —
data/mcp/t*_mcp_map.csv — are frozen ground-truth joins; rerun this
only for a NEW match).

Usage:
    uv run python -m courtvision align t3
"""

import csv


def align_match(cfg, order_pass=True):
    ev = cfg.eval
    rows, seen = [], set()
    for r in csv.DictReader(open(ev.mcp_points)):
        if r["Pt"] not in seen:
            seen.add(r["Pt"])
            rows.append(r)

    align = list(csv.DictReader(open(ev.alignment)))
    out = []
    for a in align:
        cands = []
        for r in rows:
            if not (r["Set1"] == a["set1"] and r["Set2"] == a["set2"]
                    and r["Gm1"] == a["gm1"] and r["Gm2"] == a["gm2"]):
                continue
            bug = a["pts"].upper()          # player-1-first
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
            rec["_cands"] = cands
        else:
            rec.update(mcp_pt="", svr="", first="", second="", winner="",
                       status="NO MATCH")
        out.append(rec)

    # ---- order pass: the reel is chronological, so an ambiguous clip's
    # candidates are bounded by its resolved neighbors' Pt numbers ----
    n_order = 0
    if order_pass:
        for k, rec in enumerate(out):
            if rec["status"] != "ambiguous":
                continue
            prev_pt = next((int(out[j]["mcp_pt"]) for j in range(k - 1, -1, -1)
                            if out[j]["status"] == "matched"), 0)
            next_pt = next((int(out[j]["mcp_pt"]) for j in range(k + 1, len(out))
                            if out[j]["status"] == "matched"), 10 ** 9)
            window = [c for c in rec.pop("_cands")
                      if prev_pt < int(c["Pt"]) < next_pt]
            if not window:
                continue
            m = min(window, key=lambda c: int(c["Pt"]))
            rec.update(mcp_pt=m["Pt"], svr=m["Svr"], first=m["1st"],
                       second=m["2nd"], winner=m["PtWinner"], status="matched",
                       gms=f"{m['Gm1']}-{m['Gm2']}", pts=m["Pts"],
                       note=(rec["note"] + "; " if rec["note"] else "")
                       + "order-resolved")
            n_order += 1
    for rec in out:
        rec.pop("_cands", None)
        print(f"{rec['clip']}: {rec['status']} {rec['mcp_pt']}"
              f"{' 1st=' + rec['first'][:24] if rec['first'] else ''}")
    pts_seq = [int(r["mcp_pt"]) for r in out if r["status"] == "matched"]
    if pts_seq != sorted(pts_seq):
        print("WARNING: matched Pt sequence is not monotonic — check the join")
    if order_pass:
        print(f"order pass resolved {n_order} deuce-recurrence ambiguities")

    with open(ev.mcp_map, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    n = sum(1 for r in out if r["status"] == "matched")
    print(f"\n{n}/{len(out)} clips uniquely matched -> {ev.mcp_map}")
