"""DIAGNOSIS: t4's sane-box wrong letters — which mechanism?

Rebuilds the aligned (length-matched) letter compare exactly as
courtvision.evaluate does, then for every committed-wrong letter dumps
the full forensic record: parity-true striker vs machine striker, the
MCP stroke char (slice/volley/smash vs plain), the ball-vs-box-center
margin at the refined contact frame, the sign stability of that margin
around contact, and the other player's proximity. Pure analysis.

Usage: uv run python experiments/t4_letter_diag.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np

from courtvision.config import load as load_cfg
from courtvision.mcp import FH_SIDE, BH_SIDE, parse_mcp
from courtvision.evaluate import p1_end, OTHER
from courtvision import boxes, letters

cfg = load_cfg("t4")
Hm = np.load(cfg.homography)
offsets = cfg.load_offsets()
mapd = {r["clip"]: r for r in csv.DictReader(open(cfg.eval.mcp_map))}
align = {r["clip"]: r for r in csv.DictReader(open(cfg.eval.alignment))}
match = {r["clip"]: r
         for r in csv.DictReader(open(cfg.charts_dir / "match_chart_v2.csv"))}

SPECIAL = {"r": "fh-slice", "s": "bh-slice", "v": "fh-volley",
           "z": "bh-volley", "o": "smash", "p": "bh-smash",
           "u": "fh-drop", "y": "bh-drop", "l": "fh-lob", "m": "bh-lob"}

rows = []
n_right = n_wrong = 0
right_margins = []

for clip, mc in match.items():
    m, a = mapd[clip], align[clip]
    if m["status"] != "matched":
        continue
    serve_d, strokes, played = parse_mcp(m["first"], m["second"])
    n_end = p1_end(a, cfg.eval)
    true_end = n_end if m["svr"] == "1" else OTHER[n_end]

    shots = list(csv.DictReader(open(cfg.charts_dir / f"chart2_{clip}.csv")))
    if len(shots) != 1 + len(strokes):
        continue  # aligned only

    players = boxes.load(cfg.players_dir / f"players_{clip}.csv", Hm,
                         offsets.get(clip, (0.0, 0.0)))
    ball = {int(r["frame"]): (float(r["x_stab"]), float(r["y_stab"]))
            for r in csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv"))}

    for k, sh in enumerate(shots):
        if sh["is_serve"] == "True" or sh["letter"] in ("?", ""):
            continue
        mcp_idx = k - 1
        if mcp_idx < 0 or mcp_idx >= len(strokes):
            continue
        ch = strokes[mcp_idx]
        mcp_side = "f" if ch in FH_SIDE else "b" if ch in BH_SIDE else "?"
        if mcp_side == "?":
            continue

        side = sh["striker"]
        cf = int(sh["contact_frame"])
        row = players.get(cf, {}).get(side)
        # margin at the frame the letter actually used
        dxp = None
        if row is not None and cf in ball:
            dxp = ball[cf][0] - float(row["cx"]) * 1280

        if sh["letter"] == mcp_side:
            n_right += 1
            if dxp is not None:
                right_margins.append(abs(dxp))
            continue
        n_wrong += 1

        # parity-true striker: serve (k=0) struck by true_end
        parity_striker = true_end if k % 2 == 0 else OTHER[true_end]

        # sign stability around contact +/-5 frames, striker box
        signs = []
        for f2 in range(cf - 5, cf + 6):
            r2 = players.get(f2, {}).get(side)
            if r2 is not None and f2 in ball:
                d2 = ball[f2][0] - float(r2["cx"]) * 1280
                signs.append((f2, d2))
        same = sum(1 for _, d in signs if (d > 0) == (dxp > 0)) if dxp is not None else 0

        # other player's box distance at contact frame
        orow = players.get(cf, {}).get(OTHER[side])
        odist = None
        if orow is not None and cf in ball:
            bx, by = ball[cf]
            x1 = (float(orow["cx"]) - float(orow["w"]) / 2) * 1280
            x2 = (float(orow["cx"]) + float(orow["w"]) / 2) * 1280
            y1 = (float(orow["cy"]) - float(orow["h"]) / 2) * 720
            y2 = (float(orow["cy"]) + float(orow["h"]) / 2) * 720
            dx = max(x1 - bx, 0, bx - x2)
            dy = max(y1 - by, 0, by - y2)
            odist = (dx * dx + dy * dy) ** 0.5

        w_px = float(row["w"]) * 1280 if row is not None else None
        h_px = float(row["h"]) * 720 if row is not None else None
        rows.append({
            "clip": clip, "k": k, "striker": side,
            "parity_striker": parity_striker,
            "mis": side != parity_striker,
            "ours": sh["letter"], "mcp": ch, "mcp_side": mcp_side,
            "special": SPECIAL.get(ch, "plain"),
            "frame": int(sh["frame"]), "cf": cf,
            "dist": sh["contact_dist_px"],
            "dxp": None if dxp is None else round(dxp, 1),
            "w_px": None if w_px is None else round(w_px, 1),
            "h_px": None if h_px is None else round(h_px, 1),
            "interp": row.get("interp", "") if row is not None else "nobox",
            "sign_same": f"{same}/{len(signs)}",
            "other_dist": None if odist is None else round(odist, 1),
        })

print(f"aligned committed letters: {n_right + n_wrong} "
      f"(right {n_right}, wrong {n_wrong})")
print(f"right-letter |dxp| median {np.median(right_margins):.1f} px "
      f"(n={len(right_margins)}, p25 {np.percentile(right_margins, 25):.1f})")
print()
hdr = ("clip", "k", "striker", "parity_striker", "mis", "ours", "mcp",
       "special", "frame", "cf", "dist", "dxp", "w_px", "h_px", "interp",
       "sign_same", "other_dist")
print(",".join(hdr))
for r in rows:
    print(",".join(str(r[h]) for h in hdr))
