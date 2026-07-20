"""Diagnosis 2 — t5 server end 53/71. Classify every wrong-end clip.

Per clip: truth end (MCP svr + changeover parity via evaluate.p1_end),
serves.csv call, chart server_used (override), ball-track shape around
the launch. Pure analysis; writes nothing outside outputs/diag/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import cv2

from courtvision.config import load
from courtvision.evaluate import p1_end, OTHER
from courtvision.court import NET_Y, L_C

cfg = load("t5")
ev = cfg.eval
mapd = {r["clip"]: r for r in csv.DictReader(open(ev.mcp_map))}
align = {r["clip"]: r for r in csv.DictReader(open(ev.alignment))}
match = {r["clip"]: r
         for r in csv.DictReader(open(cfg.charts_dir / "match_chart_v2.csv"))}
serves = {r["clip"]: r for r in csv.DictReader(open(cfg.serves))}
Hm = np.load(cfg.homography)
offsets = cfg.load_offsets()

rows = []
for clip, mc in match.items():
    m, a = mapd.get(clip), align.get(clip)
    if m is None or m["status"] != "matched":
        continue
    n_end = p1_end(a, ev)
    true_end = n_end if m["svr"] == "1" else OTHER[n_end]
    s = serves.get(clip, {})
    server_det = s.get("server", "?")
    server_used = mc["server_used"]
    ok = server_used == true_end

    # ball-track shape around the serve frame
    bpath = cfg.ball_dir / f"ball_{clip}.csv"
    first_tracked = None
    cy_first = None
    n_before = 0
    if bpath.exists():
        ball = list(csv.DictReader(open(bpath)))
        if ball:
            bfr = np.array([int(r["frame"]) for r in ball])
            odx, ody = offsets.get(clip, (0.0, 0.0))
            pts = np.stack([[float(r["x_stab"]) - odx for r in ball],
                            [float(r["y_stab"]) - ody for r in ball]],
                           axis=1).reshape(-1, 1, 2).astype(np.float32)
            bcy = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)[:, 1]
            first_tracked = int(bfr[0])
            cy_first = float(bcy[0])
            if s.get("serve_frame"):
                n_before = int(np.sum(bfr < int(s["serve_frame"])))

    # MCP rally length (was there a return at all?)
    from courtvision.mcp import parse_mcp
    serve_d, strokes, played = parse_mcp(m["first"], m["second"])
    rows.append({
        "clip": clip, "true_end": true_end, "det": server_det,
        "used": server_used, "ok": ok, "src": s.get("src", ""),
        "override": (server_det not in ("?", server_used)),
        "launch_cy": s.get("launch_cy", ""), "serve_f": s.get("serve_frame", ""),
        "serve_s": s.get("serve_s", ""), "margin_m": s.get("margin_m", ""),
        "side": s.get("side", ""), "reason": s.get("reason", ""),
        "first_tracked_f": first_tracked, "cy_first": round(cy_first, 1) if cy_first is not None else "",
        "n_track_before_serve": n_before,
        "mcp_len": 1 + len(strokes), "mcp_str": played[:20],
        "set": f"{a['set1']}-{a['set2']}", "gm": f"{a['gm1']}-{a['gm2']}",
        "pts": a["pts"], "svr": m["svr"],
    })

wrong = [r for r in rows if not r["ok"]]
right = [r for r in rows if r["ok"]]
print(f"scored {len(rows)}; right {len(right)}, wrong {len(wrong)}")
print(f"wrong by src: ball={sum(1 for r in wrong if r['src']=='ball')}, "
      f"players={sum(1 for r in wrong if r['src']=='players')}, "
      f"none/?={sum(1 for r in wrong if r['src'] not in ('ball','players'))}")
print(f"wrong with override (det right, chart flipped): "
      f"{sum(1 for r in wrong if r['override'] and r['det']==r['true_end'])}")
print(f"wrong where detector itself wrong: "
      f"{sum(1 for r in wrong if r['det'] not in ('?',) and r['det']!=r['true_end'])}")
print(f"wrong where det=? : {sum(1 for r in wrong if r['det']=='?')}")

hdr = ("clip", "true_end", "det", "used", "src", "override", "launch_cy",
       "serve_f", "serve_s", "first_tracked_f", "n_track_before_serve",
       "mcp_len", "margin_m", "reason", "set", "gm", "pts", "svr", "mcp_str")
print("\n--- WRONG clips ---")
print(",".join(hdr))
for r in wrong:
    print(",".join(str(r[k]) for k in hdr))

print("\n--- RIGHT clips (for contrast, launch_cy + serve_f) ---")
for r in right:
    print(f"{r['clip']} true={r['true_end']} det={r['det']} src={r['src']} "
          f"launch_cy={r['launch_cy']} serve_f={r['serve_f']} ov={r['override']}")

# also dump full CSV for downstream use
out = Path(__file__).resolve().parent.parent / "outputs" / "diag"
out.mkdir(exist_ok=True)
with open(out / "t5_server_end_diag.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n-> {out/'t5_server_end_diag.csv'}")
