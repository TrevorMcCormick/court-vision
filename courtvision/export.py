"""The charting-ready draft — MCP points schema + confidence, per match.

This is the artifact a charter would actually correct: one row per
charted clip in the Match Charting Project points-file shape
(match_id, Pt, Set1/Set2, Gm1/Gm2, Pts, Svr, 1st/2nd), with the
machine's draft string in the 1st column (the pipeline cannot see
faults, so every point is written as if played on the first serve —
correcting that is part of the charter's pass), plus the columns MCP
doesn't have and a charter triaging a draft needs:

  confidence   high = the draft is a usable starting point (~94% of
               high-flagged points are within 5 token edits, LOMO —
               data/confidence_model.json, courtvision.confidence);
               low = expect heavy correction or a re-chart
  conf_p       the scorer's raw probability
  clip         which extracted clip this row came from
  serve_s      seconds into the clip of the detected serve — the jump-to
               timestamp for review
  n_shots      shots in the draft (serve included)

Score-state columns come from the frozen alignment (the score bug read
at staging); Pt/Svr come from the MCP join where it matched and stay
blank where it didn't (those rows still need a chart — the draft
doesn't know which MCP point it is, but the point was played).

Usage:
    uv run python -m courtvision export t3
    -> outputs/t3/export/t3_mcp_draft.csv
"""

import csv

from . import confidence


def export_match(cfg, charts_dir=None):
    charts_dir = charts_dir or cfg.charts_dir
    ev = cfg.eval
    mapd = {r["clip"]: r for r in csv.DictReader(open(ev.mcp_map))}
    align = {r["clip"]: r for r in csv.DictReader(open(ev.alignment))}
    match = {r["clip"]: r
             for r in csv.DictReader(open(charts_dir / "match_chart_v2.csv"))}
    serves = cfg.load_serves()
    scores = confidence.score_match(cfg, charts_dir)

    rows = []
    for clip, mc in match.items():
        m = mapd.get(clip, {})
        a = align.get(clip, {})
        s = serves.get(clip, {})
        flag, p, _sig = scores[clip]
        matched = m.get("status") == "matched"
        n_shots = sum(1 for _ in csv.DictReader(
            open(charts_dir / f"chart2_{clip}.csv")))
        rows.append({
            "match_id": ev.mcp_match_id,
            "Pt": m.get("mcp_pt", "") if matched else "",
            "Set1": a.get("set1", ""), "Set2": a.get("set2", ""),
            "Gm1": a.get("gm1", ""), "Gm2": a.get("gm2", ""),
            "Pts": m.get("pts", "") if matched else a.get("pts", ""),
            "Svr": m.get("svr", "") if matched else "",
            "1st": mc["mcp"],
            "2nd": "",
            "confidence": flag,
            "conf_p": round(p, 3),
            "clip": clip,
            "serve_s": s.get("serve_s", ""),
            "n_shots": n_shots,
        })

    rows.sort(key=lambda r: (int(r["Pt"]) if r["Pt"].isdigit() else 10 ** 9,
                             r["clip"]))
    out_dir = cfg.out_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cfg.id}_mcp_draft.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    n_hi = sum(1 for r in rows if r["confidence"] == "high")
    print(f"{cfg.id}: {len(rows)} draft points "
          f"({n_hi} high-confidence) -> {out}")
    return out
