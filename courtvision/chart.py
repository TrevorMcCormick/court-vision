"""The chart assembler — WASB ball track + player boxes + serve call ->
a per-point shot chart and an MCP-style string.

This is the consolidation of the four t*w chart twins (2026-07-10):
one loop, per-match variation moved into MatchConfig (paths, lefty,
clip offsets, and the sanctioned staging gates — see config.py). The
logic is the twins' verbatim; the event detector is v5 (the crossing
skeleton, courtvision.events), directions are the receiver-mirrored
signal-ladder estimator (courtvision.directions), letters and endings
their own modules, player boxes pass the hygiene of courtvision.boxes.

History worth keeping at the assembler level:

  striker   strict alternation anchored on the serve. The first idea —
            vote by ball direction after the hit (vcy sign) — died on
            contact with the data: the homography assumes the ball is ON
            the ground plane, so an airborne ball's "court velocity" is
            dominated by its vertical motion in image space; post-hit vcy
            reads negative for BOTH ends. Physics votes withdrawn.
  misses    alternation over DETECTED hits breaks when a hit hides in a
            ball-track hole. Holes are observable, so: at each inter-hit
            interval containing a hole, consider a parity flip,
            adjudicated by far-half landing votes (collapse bounces are
            far-half-only by construction).
  contact   refined to the frame the ball is nearest the ASSIGNED
            striker's box — proximity is fine once the striker is
            already known.
  serve     a confident serve call with no detected serve contact is
            synthesized at the gated serve frame so it anchors
            alternation; its landing codes 4/5/6 via service-box thirds.

Landing-vote conflicts with the final chain are counted and recorded,
not hidden: they are the residual missed-hit / bogus-bounce signal.

Usage:
    uv run python -m courtvision chart t3 [clips...]
"""

import csv

import cv2
import numpy as np

from . import boxes, directions, endings, events, letters
from .court import W_C, L_C, NET_Y, CENTER_X, SINGLES_MARGIN, moving_average

SMOOTH = 3
SWING_NEAR_30 = 6.0
REFINE = 14                   # frames each side of the cusp to hunt contact
HOLE_FRAMES = 8               # ball-track gap that can hide a missed hit
SERVE_WIN = 14                # frames; serve-contact snap window


def serve_zone(bx, deuce, server):
    """Thirds of the receiving service box toward the center line: 4 wide,
    5 body, 6 T. Coarse, and honest about it."""
    if server == "near":
        left = deuce
    else:
        left = not deuce
    x0, x1 = (0 + SINGLES_MARGIN, CENTER_X) if left else (CENTER_X, W_C - SINGLES_MARGIN)
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


def assign_strikers(shots, server, holes, lock=None):
    """Strict alternation, with a parity flip allowed at each inter-hit
    interval containing a ball-track hole (a hidden hit keeps true
    alternation but repeats the striker over DETECTED shots). The chain is
    chosen by weighted votes, strongest first:
      touch  (3)  the ball reached exactly one player's box in the
                  contact window — the racquet adjudicates
      serve  (2)  the gated serve call — a PRIOR, not an anchor; a
                  wrong-end serve call must be outvotable by the ball
      landing(1)  a far-half bounce after shot k says NEAR struck it
                  (collapse bounces are far-half-only by construction)
    lock: when a confident serve call anchors shot 0, the chain's first
    striker is LOCKED to it and only the flip slots stay in play —
    measured on the t3/t4 staging pair, the serve detector beats the
    touch votes (t4: all 16 chart overrides of a confident serve flipped
    a CORRECT end to wrong; t3's ball-launch end read 48/56 against
    parity truth while the boxes chase clay shadows). The lock is a
    staging gate (config.staging.lock_serve); t1/t2 chart unlocked."""
    other = {"near": "far", "far": "near"}

    votes = []          # per shot: list of (side, weight)
    for sh in shots:
        v = []
        near_d, far_d = sh.get("touch_near"), sh.get("touch_far")
        near_ok = near_d is not None and near_d[0] <= letters.LETTER_GATE(near_d[3])
        far_ok = far_d is not None and far_d[0] <= letters.LETTER_GATE(far_d[3])
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
    for first in (("near", "far") if lock is None else (lock,)):
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


def chart_clip(cfg, stem, Hm, serves, offsets):
    """Chart one clip -> result dict, or None (track too thin / no shots)."""
    clip = cfg.clip_path(stem)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ball = list(csv.DictReader(open(cfg.ball_dir / f"ball_{stem}.csv")))
    if len(ball) < 20:
        return None
    frames = np.array([int(r["frame"]) for r in ball])
    xs = np.array([float(r["x_stab"]) for r in ball])
    ys = np.array([float(r["y_stab"]) for r in ball])
    # wandering-camera correction: clip-stabilized coords -> fit-camera
    # coords via the probe-measured clip offset, BEFORE the projection
    odx, ody = offsets.get(stem, (0.0, 0.0))
    pts = np.stack([xs - odx, ys - ody], axis=1).reshape(-1, 1, 2).astype(np.float32)
    court = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    cyc = court[:, 1]
    cxc = court[:, 0]

    s = serves.get(stem, {})
    server = s.get("server", "?")
    serve_frame = int(s["serve_frame"]) if server != "?" else None

    evs, serve_frame, v5info = events.detect_events(
        frames, ys, cyc, fps, serve_frame,
        server if server in ("near", "far") else None)
    if serve_frame is None:
        server = "?"        # spine-refuted serve call (see events module)

    # box hygiene (courtvision.boxes): court-half plausibility, teleport
    # rejection, short-gap interp — the audit-measured letter sink lives
    # in the raw boxes
    players = boxes.load(
        cfg.players_dir / f"players_{stem}.csv", Hm,
        offsets.get(stem, (0.0, 0.0)))

    ball_exact = dict(zip(frames.tolist(), zip(xs.tolist(), ys.tolist())))

    if serve_frame is None:
        first_strike = next((e for e in evs if e["kind"] == "hit"), None)
        if first_strike:
            evs = [e for e in evs if e["frame"] >= first_strike["frame"]]

    hits = [e for e in evs if e["kind"] == "hit"]
    bounces = [e for e in evs if e["kind"] == "bounce"]

    shots = []
    if serve_frame is not None and (not hits or hits[0]["frame"] > serve_frame + SERVE_WIN):
        # detector missed serve contact; the gated serve call stands in
        shots.append({"frame": serve_frame, "vcy_after": None,
                      "is_serve": True, "synth": True, "anchor": True})
    for k, h in enumerate(hits):
        is_serve = (not shots and serve_frame is not None
                    and abs(h["frame"] - serve_frame) <= SERVE_WIN)
        shots.append({"frame": h["frame"], "vcy_after": h["vcy_after"],
                      "is_serve": is_serve, "synth": False,
                      "anchor": is_serve})
    if not shots:
        return None

    # landings first (striker-independent): first bounce before the next shot
    for k, sh in enumerate(shots):
        nxt = shots[k + 1]["frame"] if k + 1 < len(shots) else 10 ** 9
        landing = next((b for b in bounces if sh["frame"] < b["frame"] < nxt), None)
        if landing is None and sh["is_serve"]:
            # near-half serve-landing fill: a far server's serve bounces
            # in the NEAR half, where the collapse detector is blind by
            # construction. A ball flying INTO the camera never reverses
            # image-y at the bounce — it decelerates: the bounce is the
            # first big descending-velocity KINK in the near half, and
            # the projection is honest exactly there (ending-fill
            # physics).
            iy_sv = moving_average(ys, SMOOTH)
            viy_sv = np.gradient(iy_sv, frames.astype(float))
            swing_sv = SWING_NEAR_30 * 30.0 / fps
            for i in range(2, len(frames) - 2):
                if not sh["frame"] + 2 < frames[i] < nxt:
                    continue
                if frames[i + 1] - frames[i - 1] > 6:
                    continue      # kink straddling a track hole
                if (viy_sv[i - 1] > 0
                        and viy_sv[i - 1] - viy_sv[i + 1] >= swing_sv
                        and NET_Y + 1 < cyc[i] < L_C):
                    landing = {"frame": int(frames[i]),
                               "court_y": float(cyc[i])}
                    break
        lx = None
        if landing is not None:
            li = np.searchsorted(frames, landing["frame"])
            lx = float(np.median(cxc[max(0, li - 2):li + 3]))
        sh["landing_x"] = round(lx, 2) if lx is not None else None
        sh["landing_y"] = round(landing["court_y"], 1) if landing else None
        if sh["is_serve"]:
            if cfg.staging.serve_zone_requires_side:
                # deuce/ad can be uncommitted (ball-called serves whose
                # stance was unreadable); no side, no zone claim
                sh["zone"] = (serve_zone(lx, s.get("side") == "deuce", server)
                              if lx is not None
                              and s.get("side") in ("deuce", "ad") else "?")
            else:
                sh["zone"] = (serve_zone(lx, s.get("side") == "deuce", server)
                              if lx is not None else "?")
        # rally-shot directions come from directions.annotate() below

    h_typ = letters.typical_heights(players)

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
    lock = (server if (cfg.staging.lock_serve and server != "?"
                       and shots[0].get("anchor")) else None)
    conflicts = assign_strikers(shots, server if server != "?" else "near",
                                holes, lock)

    # the chart can outvote the serve detector's end call; when it does,
    # the detector's deuce/ad side is suspect too, so the serve zone
    # degrades to '?'
    server_used = server
    if shots[0]["is_serve"]:
        server_used = shots[0]["striker"]
        if server != "?" and server_used != server:
            shots[0]["zone"] = "?"

    letters.commit(shots, h_typ, cfg.lefty)

    # rally-shot direction digits (MCP 1/2/3), BOTH halves: the
    # receiver-mirrored mapping + the measured signal ladder (near-half
    # landing > receiver contact > crossing+slope > far-half landing)
    directions.annotate(shots, frames, cyc, cxc, fps)

    ending = endings.infer(shots, frames, ys, cyc, cxc, fps,
                           near_fill=cfg.staging.near_ending_fill)

    mcp = ""
    for sh in shots:
        mcp += f"s{sh['zone']}" if sh["is_serve"] else f"{sh['letter']}{sh['zone']}"
    mcp += ending

    return {"clip": stem, "fps": fps, "server": server,
            "server_used": server_used, "side": s.get("side", ""),
            "n_hits": len(hits), "n_bounces": len(bounces),
            "conflicts": conflicts, "n_holes": len(holes),
            "n_coda": v5info["n_dropped"], "coda_why": v5info["why"],
            "ending": ending, "mcp": mcp, "shots": shots,
            "crossings": v5info.get("crossings", [])}


SHOT_FIELDS = ["shot", "frame", "contact_frame", "contact_dist_px", "is_serve",
               "synth", "striker", "letter", "zone", "landing_x", "landing_y",
               "vcy_after"]


def chart_match(cfg, stems=None, charts_dir=None, quiet=False):
    """Chart a set of clips (default: every clip with a ball track) and
    write chart2_<clip>.csv per point plus match_chart_v2.csv. Returns
    the list of result dicts."""
    if stems is None:
        stems = cfg.ball_stems()
    charts_dir = charts_dir or cfg.charts_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    offsets = cfg.load_offsets()
    Hm = np.load(cfg.homography)
    serves = cfg.load_serves()

    row_keys = ["clip", "server", "server_used", "side", "n_hits",
                "n_bounces", "n_holes", "conflicts"]
    if cfg.staging.coda_report:
        row_keys += ["n_coda", "coda_why"]
    row_keys += ["ending", "mcp"]

    rows, results = [], []
    for stem in stems:
        r = chart_clip(cfg, stem, Hm, serves, offsets)
        if r is None:
            if not quiet:
                print(f"{stem}: track too thin or no shots, skipped")
            continue
        with open(charts_dir / f"chart2_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=SHOT_FIELDS)
            wr.writeheader()
            for k, sh in enumerate(r["shots"]):
                wr.writerow({"shot": k + 1,
                             **{k2: sh.get(k2) for k2 in SHOT_FIELDS[1:]}})
        rows.append({k: r[k] for k in row_keys})
        results.append(r)
        if not quiet:
            strikers = "".join(sh["striker"][0].upper() for sh in r["shots"])
            flip = (" SERVER-OVERRIDE"
                    if r["server"] not in ("?", r["server_used"]) else "")
            coda = ""
            if cfg.staging.coda_report and r["n_coda"]:
                coda = f" coda-{r['n_coda']}[{r['coda_why']}]"
            print(f"{r['clip']}: server={r['server']}->({r['server_used']}) "
                  f"strikers={strikers} holes={r['n_holes']} "
                  f"conflicts={r['conflicts']}{flip}{coda}  MCP: {r['mcp']}")

    with open(charts_dir / "match_chart_v2.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    if not quiet:
        print(f"-> {charts_dir / 'match_chart_v2.csv'} ({len(rows)} points)")
    return results
