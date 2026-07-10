"""T2 — the frozen chart loop on the control match (day, both right-handed).

v1 assigned striker by proximity at the detected event frame; frame-checks
showed the cusp often catches the ball mid-air BETWEEN players (the apex),
so proximity flips the striker mid-rally and alternation breaks (point_16
read near,near,near,far,far,far,far,far — tennis doesn't work like that).

v2's fixes, all free on the six tracks already paid for:
  striker   strict alternation anchored on the serve. The first idea —
            vote by ball direction after the hit (vcy sign) — died on
            contact with the data: the homography assumes the ball is ON
            the ground plane, so an airborne ball's "court velocity" is
            dominated by its vertical motion in image space; post-hit vcy
            reads negative for BOTH ends. Physics votes withdrawn.
  misses    alternation over DETECTED hits breaks when a hit hides in a
            ball-track hole (point_53: SAM loses the ball f54-78, one far
            hit vanishes, every striker after it flips). Holes are
            observable, so: at each inter-hit interval containing a hole,
            consider a parity flip, adjudicated by far-half landing votes
            (collapse bounces are far-half-only by construction — the one
            side the geometry lets us see).
  contact   refined to the frame the ball is nearest the ASSIGNED
            striker's box — proximity is fine once the striker is already
            known. The f/b letter is committed only when the ball actually
            reached the box (<= LETTER_MAX_PX); frame checks showed the
            player boxes go rogue at exactly the wrong moments (a "far"
            box on a spectator, a "near" box on a court shadow), and a
            letter read 300 px from the ball is a coin flip in costume.
  serve     three pilot clips had a confident serve call but no 's' in the
            string: the detector missed serve contact. v2 synthesizes the
            serve at the gated serve frame so it anchors alternation, and
            its landing codes 4/5/6 via service-box thirds (v1 defined
            serve_zone() and never called it — strings showed full-court
            thirds for serves).

Landing-vote conflicts with the final chain are counted and printed, not
hidden: they are the residual missed-hit / bogus-bounce signal.

Usage:
    uv run experiments/m3_chart_point_v2.py point_16 point_29 ...
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "t2"
CHART_DIR = OUT_BASE / "charts"
ROOT = Path(__file__).resolve().parent.parent

SMOOTH = 3
WIN = 6
MIN_GAP_S = 8 / 30.0
SWING_NEAR_30 = 6.0
SWING_FAR_30 = 1.2
FAR_Y_PX = 250
HIT_SPEED = 4.5
COLLAPSE = 3.0
W_C, L_C = 10.97, 23.77
NET_Y = L_C / 2
CENTER_X = W_C / 2
REFINE = 14                   # frames each side of the cusp to hunt contact
# letter gate scales with the striker's apparent size: racquet reach is
# ~0.6 of body height, plus slack for 1 frame of ball flight
LETTER_GATE = lambda h_px: 0.6 * h_px + 30
HOLE_FRAMES = 8               # ball-track gap that can hide a missed hit
OUT_MARGIN = 0.25             # meters of slack before calling a ball out
NET_ZONE_M = 1.5              # track dying this close to the net = net error
# Per-match handedness — freeze #2's one allowed knob. The letter logic
# reads which side of the striker's body the ball is on in image x and
# calls right-side 'f' — true only for right-handers; a lefty's call
# flips. BOTH t1 players are left-handed, so both ends flip all match
# and changeover end-swapping is moot. A mixed-handedness match needs
# player-identity-per-END via changeover parity — future work.
LEFTY = {"near": False, "far": False}


def moving_average(x, k):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def detect_events(frames, iy, cyc, fps):
    """M2 v4, fps-parameterized; hits carry their post-hit vcy median."""
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
        ma = float(np.median(vcy[i + 1:i + 1 + WIN]))
        kind = "bounce" if how == "collapse" else ("hit" if abs(ma) > HIT_SPEED else "bounce")
        pos_y = float(np.median(cyc[i + 1:i + 1 + WIN])) if how == "collapse" else float(cyc[i])
        events.append({"idx": i, "frame": int(frames[i]), "kind": kind,
                       "signal": how, "court_y": pos_y, "vcy_after": ma})
    return events


def zone(x):
    third = W_C / 3
    return 1 if x < third else (2 if x < 2 * third else 3)


def serve_zone(bx, deuce, server):
    """Thirds of the receiving service box toward the center line: 4 wide,
    5 body, 6 T. Coarse, and honest about it."""
    if server == "near":
        left = deuce
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


W_TOUCH, W_SERVE, W_LAND = 3.0, 2.0, 1.0
FLIP_COST = 0.75


def assign_strikers(shots, server, holes):
    """Strict alternation, with a parity flip allowed at each inter-hit
    interval containing a ball-track hole (a hidden hit keeps true
    alternation but repeats the striker over DETECTED shots). The chain is
    chosen by weighted votes, strongest first:
      touch  (3)  the ball reached exactly one player's box in the
                  contact window — the racquet adjudicates
      serve  (2)  the gated serve call — a PRIOR, not an anchor; point_53's
                  wrong-end serve call must be outvotable by the ball
      landing(1)  a far-half bounce after shot k says NEAR struck it
                  (collapse bounces are far-half-only by construction)"""
    other = {"near": "far", "far": "near"}

    votes = []          # per shot: list of (side, weight)
    for sh in shots:
        v = []
        near_d, far_d = sh.get("touch_near"), sh.get("touch_far")
        near_ok = near_d is not None and near_d[0] <= LETTER_GATE(near_d[3])
        far_ok = far_d is not None and far_d[0] <= LETTER_GATE(far_d[3])
        if near_ok != far_ok:
            v.append(("near" if near_ok else "far", W_TOUCH))
        if sh.get("anchor"):
            v.append((server, W_SERVE))
        if sh.get("landing_y") is not None and -1.0 < sh["landing_y"] < NET_Y:
            v.append(("near", W_LAND))
        votes.append(v)

    flip_slots = [k for k in range(len(shots) - 1)
                  if any(a < shots[k + 1]["frame"] and b > shots[k]["frame"]
                         for a, b in holes)][:4]

    best = None
    for first in ("near", "far"):
        for mask in range(2 ** len(flip_slots)):
            chain, cur = [], first
            for k in range(len(shots)):
                chain.append(cur)
                cur = other[cur]
                if k in flip_slots and mask >> flip_slots.index(k) & 1:
                    cur = other[cur]      # hidden hit: striker repeats
            score = -FLIP_COST * bin(mask).count("1")
            for v, c in zip(votes, chain):
                for side, w in v:
                    score += w if side == c else -w
            if best is None or score > best[0]:
                best = (score, chain)

    chain = best[1]
    conflicts = sum(1 for v, c in zip(votes, chain)
                    for side, _ in v if side != c)
    for sh, c in zip(shots, chain):
        sh["striker"] = c
    return conflicts


def chart_clip(stem, Hm, serves):
    clip = ROOT / "clips/points_t2" / f"{stem}.mp4"
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

    s = serves.get(stem, {})
    server = s.get("server", "?")
    serve_frame = int(s["serve_frame"]) if server != "?" else None

    if serve_frame is not None:
        events = [e for e in events if e["frame"] >= serve_frame - 3]
    else:
        first_strike = next((e for e in events if e["kind"] == "hit"), None)
        if first_strike:
            events = [e for e in events if e["frame"] >= first_strike["frame"]]

    hits = [e for e in events if e["kind"] == "hit"]
    bounces = [e for e in events if e["kind"] == "bounce"]

    shots = []
    if serve_frame is not None and (not hits or hits[0]["frame"] > serve_frame + 14):
        # detector missed serve contact; the gated serve call stands in
        shots.append({"frame": serve_frame, "vcy_after": None,
                      "is_serve": True, "synth": True, "anchor": True})
    for k, h in enumerate(hits):
        is_serve = (not shots and serve_frame is not None
                    and abs(h["frame"] - serve_frame) <= 14)
        shots.append({"frame": h["frame"], "vcy_after": h["vcy_after"],
                      "is_serve": is_serve, "synth": False,
                      "anchor": is_serve})
    if not shots:
        return None

    # landings first (striker-independent): first bounce before the next shot
    for k, sh in enumerate(shots):
        nxt = shots[k + 1]["frame"] if k + 1 < len(shots) else 10 ** 9
        landing = next((b for b in bounces if sh["frame"] < b["frame"] < nxt), None)
        lx = None
        if landing is not None:
            li = np.searchsorted(frames, landing["frame"])
            lx = float(np.median(cxc[max(0, li - 2):li + 3]))
        sh["landing_x"] = round(lx, 2) if lx is not None else None
        sh["landing_y"] = round(landing["court_y"], 1) if landing else None
        if sh["is_serve"] and lx is not None:
            sh["zone"] = serve_zone(lx, s.get("side") == "deuce", server)
        elif lx is not None:
            sh["zone"] = zone(lx)
        else:
            sh["zone"] = "?"

    # per-side contact search: nearest the ball comes to EACH player's box
    # inside the shot's window — feeds both the touch votes and the letter
    for k, sh in enumerate(shots):
        fi = sh["frame"]
        reach = REFINE
        if k > 0:
            reach = min(reach, (fi - shots[k - 1]["frame"]) // 2)
        if k + 1 < len(shots):
            reach = min(reach, (shots[k + 1]["frame"] - fi) // 2)
        for side in ("near", "far"):
            best = None
            for d in range(max(reach, 1) + 1):
                for f2 in ({fi} if d == 0 else {fi - d, fi + d}):
                    if f2 not in ball_exact:
                        continue
                    p = players.get(f2, {}).get(side)
                    if p is None:
                        continue
                    bx, by = ball_exact[f2]
                    dist = box_dist(bx, by, p)
                    if best is None or dist < best[0]:
                        best = (dist, f2, bx, float(p["h"]) * 720,
                                float(p["cx"]) * 1280)
            sh[f"touch_{side}"] = best

    holes = [(int(a), int(b)) for a, b in zip(frames, frames[1:])
             if b - a > HOLE_FRAMES]
    conflicts = assign_strikers(shots, server if server != "?" else "near", holes)

    # the chart can outvote the serve detector's end call; when it does,
    # the detector's deuce/ad side is suspect too, so the serve zone
    # degrades to '?'
    server_used = server
    if shots[0]["is_serve"]:
        server_used = shots[0]["striker"]
        if server != "?" and server_used != server:
            shots[0]["zone"] = "?"

    for sh in shots:
        best = sh.get(f"touch_{sh['striker']}")
        sh["contact_frame"] = best[1] if best else sh["frame"]
        sh["contact_dist_px"] = round(best[0], 1) if best else None
        if best and not sh["synth"] and best[0] <= LETTER_GATE(best[3]):
            dxp = best[2] - best[4]
            right = dxp > 0 if sh["striker"] == "near" else dxp < 0
            forehand = right != LEFTY[sh["striker"]]
            sh["letter"] = "f" if forehand else "b"
        else:
            sh["letter"] = "?"

    # ending v1 — observable evidence only. The last shot's own landing
    # (far-half only, by construction) codes out-deep/-wide; a ball track
    # that dies at the net right after the last hit codes a net error.
    # Winner vs forced vs unforced is HUMAN judgment the pipeline does not
    # attempt: '@' means "error, attribution not judged" and eval compares
    # the ending TYPE only.
    ending = "?"
    last = shots[-1]
    if last.get("landing_y") is not None:
        ly, lx = last["landing_y"], last.get("landing_x")
        deep = ly < -OUT_MARGIN
        wide = lx is not None and not (
            1.372 - OUT_MARGIN <= lx <= W_C - 1.372 + OUT_MARGIN)
        ending = ("x@" if deep and wide else "d@" if deep
                  else "w@" if wide else "*")   # in, and nothing came back
    else:
        li = int(np.searchsorted(frames, last["frame"]))
        tail_y, tail_f = cyc[li:], frames[li:]
        if (len(tail_y) >= 3
                and abs(float(tail_y[-1]) - NET_Y) < NET_ZONE_M
                and tail_f[-1] - last["frame"] <= 1.2 * fps):
            ending = "n@"

    mcp = ""
    for sh in shots:
        mcp += f"s{sh['zone']}" if sh["is_serve"] else f"{sh['letter']}{sh['zone']}"
    mcp += ending

    return {"clip": stem, "fps": fps, "server": server,
            "server_used": server_used, "side": s.get("side", ""),
            "n_hits": len(hits), "n_bounces": len(bounces),
            "conflicts": conflicts, "n_holes": len(holes),
            "ending": ending, "mcp": mcp, "shots": shots}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    args = parser.parse_args()

    Hm = np.load(ROOT / "outputs/t2/H_img_to_court.npy")
    serves = {r["clip"]: r for r in csv.DictReader(open(OUT_BASE / "serves.csv"))}
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    fields = ["shot", "frame", "contact_frame", "contact_dist_px", "is_serve",
              "synth", "striker", "letter", "zone", "landing_x", "landing_y",
              "vcy_after"]
    rows = []
    for stem in args.clips:
        r = chart_clip(stem, Hm, serves)
        if r is None:
            print(f"{stem}: track too thin or no shots, skipped")
            continue
        with open(CHART_DIR / f"chart2_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for k, sh in enumerate(r["shots"]):
                wr.writerow({"shot": k + 1, **{k2: sh.get(k2) for k2 in fields[1:]}})
        rows.append({k: r[k] for k in
                     ("clip", "server", "server_used", "side", "n_hits",
                      "n_bounces", "n_holes", "conflicts", "ending", "mcp")})
        strikers = "".join(sh["striker"][0].upper() for sh in r["shots"])
        flip = (" SERVER-OVERRIDE" if r["server"] not in ("?", r["server_used"])
                else "")
        print(f"{r['clip']}: server={r['server']}->({r['server_used']}) "
              f"strikers={strikers} holes={r['n_holes']} "
              f"conflicts={r['conflicts']}{flip}  MCP: {r['mcp']}")

    with open(CHART_DIR / "match_chart_v2.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {CHART_DIR / 'match_chart_v2.csv'} ({len(rows)} points)")


if __name__ == "__main__":
    main()
