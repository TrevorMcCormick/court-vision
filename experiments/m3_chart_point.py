"""M3 experiment 7 — the charting loop: one point clip in, one MCP line out.

Assembles everything the milestone has built:
  ball track   (m3_track_ball, SAM, stabilized coords)
  players      (m3_bgsub_players, $0)
  serve        (m3_serve_detect, gated)
  events       (M2 v4 detector, ported with fps as a PARAMETER — the reel
                runs 25 where rally.mp4 ran 30; swing thresholds and gap
                windows scale, physical thresholds in m/s don't)
  letters      (m3_shot_types logic: striker by proximity, contact side
                mirrored by end, right-handed assumption on the record)

Per clip: events CSV + a pseudo-MCP string. Across clips: match_chart.csv.
Serve coding is coarse: first hit near the detected serve frame is THE
serve; its landing's third of the service box gives 4/5/6 (wide/body/T).

Usage:
    uv run experiments/m3_chart_point.py point_16 point_29 ...
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "m3"
CHART_DIR = OUT_BASE / "charts"
ROOT = Path(__file__).resolve().parent.parent

SMOOTH = 3
WIN = 6
MIN_GAP_S = 8 / 30.0          # v4's 8 frames at 30 fps, now in seconds
SWING_NEAR_30 = 6.0           # px/frame at 30 fps; scaled by 30/fps
SWING_FAR_30 = 1.2
FAR_Y_PX = 250
HIT_SPEED = 4.5               # m/s — physical, fps-independent
COLLAPSE = 3.0
W_C, L_C = 10.97, 23.77
NET_Y = L_C / 2
SVC_HALF = 6.40
CENTER_X = W_C / 2


def moving_average(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def detect_events(frames, iy, cyc, fps):
    """M2 v4, fps-parameterized. frames/iy/cyc are aligned arrays."""
    min_gap = max(3, int(round(MIN_GAP_S * fps)))
    swing_near = SWING_NEAR_30 * 30.0 / fps
    swing_far = SWING_FAR_30 * 30.0 / fps

    iy_s = moving_average(iy, SMOOTH)
    viy = np.gradient(iy_s, frames.astype(float))
    vcy = np.gradient(cyc, frames.astype(float)) * fps

    cusps = []
    for i in range(2, len(frames) - 2):
        if viy[i - 1] > 0 and viy[i + 1] < 0:
            swing = viy[i - 1] - viy[i + 1]
            need = swing_near if iy_s[i] >= FAR_Y_PX else swing_far
            if swing >= need:
                cusps.append(i)
    merged = []
    for i in sorted(cusps, key=lambda i: -(viy[i - 1] - viy[i + 1])):
        if all(abs(int(frames[i]) - int(frames[j])) >= min_gap for j in merged):
            merged.append(i)
    merged.sort()

    collapse = []
    for i in range(WIN, len(frames) - WIN):
        if cyc[i] > NET_Y:
            continue
        mb = np.median(vcy[i - WIN:i])
        ma = np.median(vcy[i + 1:i + 1 + WIN])
        if np.sign(mb) == np.sign(ma) and abs(ma) > 0.3 and abs(mb) / abs(ma) >= COLLAPSE:
            collapse.append(i)
    bounds = [-1] + [int(frames[i]) for i in merged] + [10 ** 9]
    last_per_seg = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = [i for i in collapse if a + min_gap <= frames[i] <= b - min_gap]
        if seg:
            last_per_seg.append(seg[-1])

    events = []
    for i in sorted(merged + last_per_seg):
        how = "cusp" if i in merged else "collapse"
        ma = np.median(vcy[i + 1:i + 1 + WIN])
        kind = "bounce" if how == "collapse" else ("hit" if abs(ma) > HIT_SPEED else "bounce")
        pos_y = float(np.median(cyc[i + 1:i + 1 + WIN])) if how == "collapse" else float(cyc[i])
        events.append({"idx": i, "frame": int(frames[i]), "kind": kind,
                       "signal": how, "court_y": pos_y})
    return events


def zone(x):
    third = W_C / 3
    return 1 if x < third else (2 if x < 2 * third else 3)


def serve_zone(bx, deuce, server):
    """Thirds of the receiving service box toward the center line: 4 wide,
    5 body, 6 T. Coarse, and honest about it."""
    # receiving box x-range depends on server end + court side
    if server == "near":       # serving toward far half
        left = deuce           # near deuce serve lands in far left box (image/court x < center)
    else:
        left = not deuce
    x0, x1 = (0 + 1.372, CENTER_X) if left else (CENTER_X, W_C - 1.372)
    t = (bx - x0) / (x1 - x0)
    t = min(max(t, 0.0), 1.0)
    toward_center = t if not left else 1 - t
    return 6 if toward_center < 1 / 3 else (5 if toward_center < 2 / 3 else 4)


def box_dist(px, py, p):
    x1 = (float(p["cx"]) - float(p["w"]) / 2) * 1280
    y1 = (float(p["cy"]) - float(p["h"]) / 2) * 720
    x2 = (float(p["cx"]) + float(p["w"]) / 2) * 1280
    y2 = (float(p["cy"]) + float(p["h"]) / 2) * 720
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def chart_clip(stem, Hm, serves):
    clip = ROOT / "clips/points" / f"{stem}.mp4"
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ball = list(csv.DictReader(open(OUT_BASE / "ball" / f"ball_{stem}.csv")))
    if len(ball) < 20:
        return None
    frames = np.array([int(r["frame"]) for r in ball])
    xs = np.array([float(r["x_stab"]) for r in ball])
    ys = np.array([float(r["y_stab"]) for r in ball])
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float32)
    court = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    cyc = court[:, 1]
    cxc = court[:, 0]

    events = detect_events(frames, ys, cyc, fps)

    players = {}
    with open(OUT_BASE / "players" / f"players_{stem}.csv") as f:
        for row in csv.DictReader(f):
            players.setdefault(int(row["frame"]), {})[row["player"]] = row

    ball_exact = dict(zip(frames.tolist(), zip(xs.tolist(), ys.tolist())))

    def ball_at(fi, tol=3):
        for d in range(tol + 1):
            for f2 in (fi - d, fi + d):
                if f2 in ball_exact:
                    return ball_exact[f2]
        return (None, None)

    def players_at(fi, tol=3):
        for d in range(tol + 1):
            for f2 in (fi - d, fi + d):
                if len(players.get(f2, {})) == 2:
                    return players[f2]
        return {}

    s = serves.get(stem, {})
    serve_frame = int(s["serve_frame"]) if s.get("server", "?") != "?" else None

    # the server's pre-serve dribble is REAL hits and bounces to the
    # detector — window events to the point proper
    if serve_frame is not None:
        events = [e for e in events if e["frame"] >= serve_frame - 3]
    else:
        first_strike = next((e for e in events if e["kind"] == "hit"), None)
        if first_strike:
            events = [e for e in events if e["frame"] >= first_strike["frame"]]

    hits = [e for e in events if e["kind"] == "hit"]
    bounces = [e for e in events if e["kind"] == "bounce"]

    shots = []
    for k, h in enumerate(hits):
        fi = h["frame"]
        bx, by = ball_at(fi)
        pl = players_at(fi)
        striker = side = None
        if bx is not None and len(pl) == 2:
            d = {sd: box_dist(bx, by, pl[sd]) for sd in ("near", "far")}
            striker = min(d, key=d.get)
            p = pl[striker]
            dxp = bx - float(p["cx"]) * 1280
            right = dxp > 0 if striker == "near" else dxp < 0
            side = "f" if right else "b"
        nxt = hits[k + 1]["frame"] if k + 1 < len(hits) else 10 ** 9
        landing = next((b for b in bounces if fi < b["frame"] < nxt), None)
        lx = None
        if landing is not None:
            li = np.searchsorted(frames, landing["frame"])
            lx = float(np.median(cxc[max(0, li - 2):li + 3]))
        is_serve = serve_frame is not None and abs(fi - serve_frame) <= 14 and k == 0
        shots.append({
            "shot": k + 1, "frame": fi,
            "is_serve": is_serve,
            "striker": striker or "?",
            "letter": side or "?",
            "zone": zone(lx) if lx is not None else "?",
            "landing_y": round(landing["court_y"], 1) if landing else "?",
        })

    # string: 's' marks the serve (proper 4/5/6 wide/body/T coding is next
    # once serve landings are frame-verified); groundstrokes letter+zone
    mcp = ""
    for sh in shots:
        if sh["is_serve"]:
            mcp += f"s{sh['zone']}"
        else:
            mcp += f"{sh['letter']}{sh['zone']}"
    mcp += "?"  # ending: not yet coded

    return {"clip": stem, "fps": fps,
            "server": s.get("server", "?"), "side": s.get("side", ""),
            "n_events": len(events), "n_hits": len(hits),
            "n_bounces": len(bounces), "mcp": mcp,
            "events": events, "shots": shots}


serve_zone_x = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    args = parser.parse_args()

    Hm = np.load(ROOT / "outputs/m1/H_img_to_court.npy")
    serves = {r["clip"]: r for r in csv.DictReader(open(OUT_BASE / "serves.csv"))}
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for stem in args.clips:
        r = chart_clip(stem, Hm, serves)
        if r is None:
            print(f"{stem}: ball track too thin, skipped")
            continue
        if not r["shots"]:
            print(f"{r['clip']}: no hits detected in window, skipped")
            continue
        with open(CHART_DIR / f"chart_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(r["shots"][0].keys()))
            wr.writeheader()
            wr.writerows(r["shots"])
        rows.append({k: r[k] for k in
                     ("clip", "server", "side", "n_hits", "n_bounces", "mcp")})
        print(f"{r['clip']}: server={r['server']}({r['side']}) "
              f"hits={r['n_hits']} bounces={r['n_bounces']}  MCP: {r['mcp']}")

    with open(CHART_DIR / "match_chart.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {CHART_DIR / 'match_chart.csv'} ({len(rows)} points)")


if __name__ == "__main__":
    main()
