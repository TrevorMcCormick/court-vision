"""M3 — chart v2: the proto-chart with its shot-type '?' column filled.

Merges m3_shot_types.py letters (7/7 frame-verified) and striker court
positions into the proto-chart. The pseudo-MCP string gains its letters:

    before: ??2?1?1?2?2?2???
    after:  ?f2f1f1b2f2f2b??

Remaining '?': the unseen serve, shot 7's landing, and the ending code —
all waiting on full-point clips (M3 requirement #2).

Usage:
    uv run experiments/m3_chart_v2.py
"""

import csv
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"


def main():
    proto = list(csv.DictReader(open(OUT_DIR / "proto_chart.csv")))
    types = {int(r["shot"]): r for r in csv.DictReader(open(OUT_DIR / "shot_types.csv"))}

    shots = []
    for p in proto:
        k = int(p["shot"])
        t = types[k]
        shots.append({
            "shot": k,
            "frame": t["frame"],
            "striker": p["striker"],
            "shot_type": t["shot_type"],
            "direction_zone": p["direction_zone"],
            "depth_code": p["depth_code"],
            "landing_y_m": p["landing_y_m"],
            "landing_trust": p["landing_trust"],
            "striker_court_x": t["striker_court_x"],
            "striker_court_y": t["striker_court_y"],
            "ending": p["ending"],
        })

    with open(OUT_DIR / "chart_v2.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(shots[0].keys()))
        wr.writeheader()
        wr.writerows(shots)

    print("CHART v2 — shot types filled, strikers placed")
    print(f"{'#':>2} {'striker':7} {'type':4} {'dir':3} {'depth':5} "
          f"{'landing':>9} {'struck from':>13}  trust")
    for s in shots:
        pos = f"({s['striker_court_x']},{s['striker_court_y']})"
        print(f"{s['shot']:>2} {s['striker']:7} {s['shot_type']:4} "
              f"{str(s['direction_zone']):3} {str(s['depth_code']):5} "
              f"{str(s['landing_y_m']):>7}m {pos:>13}  {s['landing_trust']}")

    mcp = "?"  # serve: not in clip
    for s in shots:
        mcp += f"{s['shot_type']}{s['direction_zone']}"
    mcp += "?"  # ending
    print(f"\npseudo-MCP string: {mcp}")
    print(f"rallyCount: {len(shots)} (+ unseen serve)")
    print("remaining '?': serve, shot-7 landing, ending -> full-point clips")
    print(f"-> {OUT_DIR / 'chart_v2.csv'}")


if __name__ == "__main__":
    main()
