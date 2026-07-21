"""Proof-of-concept: does a model TRAINED on the aligned MCP corpus beat
the hand-written rules on the two weakest chart components?

  serve placement   MCP serve digit 4/5/6  (wide / body / T)
  ending type       MCP winner / net / wide / deep  (* / n / w / d)

HONESTY (the whole point):
  * Leave-one-match-out. For each of the 7 matches, the forest trains on
    the other 6 and predicts the held-out match. A match never appears in
    its own training set.
  * The learned model is compared to the SHIPPED hand-tuned heuristic on
    the IDENTICAL held-out points.
  * Labels are the aligned MCP truth (courtvision.evaluate's matched
    records -> parse_mcp / mcp_ending_type). No feature is the label or
    the heuristic being replaced:
      - serve features use the serve's raw ball GEOMETRY (court landing
        x/y, net-crossing x, contact position, toss/launch) but NOT our
        emitted zone digit;
      - ending features use the last shot's raw KINEMATICS but NOT our
        emitted ending token.
  * Baseline scored with refusal = wrong (a '?' is not correct), the same
    rule the confidence layer uses. This is the fair apples-to-apples set:
    both systems face every labelled held-out point and the model is never
    allowed to abstain. A committed-only baseline is reported alongside.

Run:
    uv run python experiments/learn_components.py
Writes outputs/diag/learn_components.md
"""

import csv
import datetime

import cv2
import numpy as np

from courtvision import config, evaluate
from courtvision.config import ROOT
from courtvision.learn import RandomForest
from courtvision.mcp import mcp_ending_type, our_ending_type

SEED = 0


# --------------------------------------------------------------------------
# per-match artifact loading + court projection (mirrors confidence.point_signals)
# --------------------------------------------------------------------------

def _court_track(cfg, clip, Hm, offsets):
    ball = list(csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv")))
    if not ball:
        return None
    frames = np.array([int(r["frame"]) for r in ball])
    xs = np.array([float(r["x_stab"]) for r in ball])
    ys = np.array([float(r["y_stab"]) for r in ball])
    odx, ody = offsets.get(clip, (0.0, 0.0))
    pts = np.stack([xs - odx, ys - ody], axis=1).reshape(-1, 1, 2).astype(np.float32)
    court = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    return {"frames": frames, "ys_img": ys, "cx": court[:, 0], "cy": court[:, 1]}


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _chart_rows(cfg, clip):
    return list(csv.DictReader(open(cfg.charts_dir / f"chart2_{clip}.csv")))


# --------------------------------------------------------------------------
# feature extraction — SERVE PLACEMENT
# --------------------------------------------------------------------------

SERVE_FEATURES = [
    "land_x", "land_y", "abs_land_x", "has_land", "contact_dist",
    "server_x_m", "abs_server_x", "margin_m", "serve_s", "launch_cy",
    "toss_h_norm", "side_deuce", "side_ad", "src_ball", "server_near",
    "netcross_x", "abs_netcross_x", "has_netcross",
]


def serve_features(rows, srv_row, track):
    srv = next((r for r in rows if r["is_serve"] == "True"), None)
    land_x = _f(srv["landing_x"]) if srv and srv["landing_x"] else 0.0
    land_y = _f(srv["landing_y"]) if srv and srv["landing_y"] else 0.0
    has_land = 1.0 if (srv and srv["landing_x"]) else 0.0
    contact_dist = _f(srv["contact_dist_px"]) if srv and srv["contact_dist_px"] else 0.0

    server_x = _f(srv_row.get("server_x_m")) if srv_row else 0.0
    margin = _f(srv_row.get("margin_m")) if srv_row else 0.0
    serve_s = _f(srv_row.get("serve_s")) if srv_row else 0.0
    launch_cy = _f(srv_row.get("launch_cy")) if srv_row else 0.0
    toss_h = _f(srv_row.get("toss_h_norm")) if srv_row else 0.0
    side = (srv_row or {}).get("side", "")
    src_ball = 1.0 if (srv_row or {}).get("src") == "ball" else 0.0
    server = (srv_row or {}).get("server", "")
    server_near = 1.0 if server == "near" else 0.0

    # net-crossing x from the serve's ball flight: first sign change in cy
    nx, has_nx = 0.0, 0.0
    if track is not None:
        cy, cx = track["cy"], track["cx"]
        for i in range(len(cy) - 1):
            if cy[i] == 0 or (cy[i] < 0) != (cy[i + 1] < 0):
                denom = (cy[i + 1] - cy[i]) or 1.0
                frac = -cy[i] / denom
                nx = float(cx[i] + frac * (cx[i + 1] - cx[i]))
                has_nx = 1.0
                break

    return [land_x, land_y, abs(land_x), has_land, contact_dist,
            server_x, abs(server_x), margin, serve_s, launch_cy,
            toss_h, 1.0 if side == "deuce" else 0.0,
            1.0 if side == "ad" else 0.0, src_ball, server_near,
            nx, abs(nx), has_nx]


# --------------------------------------------------------------------------
# feature extraction — ENDING TYPE
# --------------------------------------------------------------------------

END_FEATURES = [
    "n_shots", "last_land_x", "last_land_y", "last_abs_land_x",
    "last_vcy_after", "last_striker_near", "has_last_land",
    "final_cx", "final_cy", "final_abs_cx", "final_abs_cy",
    "vx", "vy", "speed", "max_abs_cy", "max_abs_cx",
    "tail_min_abs_cy", "img_fall_dy", "n_track",
]


def ending_features(rows, track):
    n_shots = float(len(rows))
    last = rows[-1] if rows else None
    lx = _f(last["landing_x"]) if last and last["landing_x"] else 0.0
    ly = _f(last["landing_y"]) if last and last["landing_y"] else 0.0
    has_ll = 1.0 if (last and last["landing_x"]) else 0.0
    vcy = _f(last["vcy_after"]) if last and last["vcy_after"] else 0.0
    last_near = 1.0 if (last and last["striker"] == "near") else 0.0

    fcx = fcy = vx = vy = speed = maxcy = maxcx = tail_min = fall = 0.0
    n_track = 0.0
    if track is not None and len(track["cy"]) > 0:
        cy, cx, ys = track["cy"], track["cx"], track["ys_img"]
        n_track = float(len(cy))
        fcx, fcy = float(cx[-1]), float(cy[-1])
        maxcy = float(np.max(np.abs(cy)))
        maxcx = float(np.max(np.abs(cx)))
        k = min(5, len(cy) - 1)
        if k >= 1:
            vx = float(cx[-1] - cx[-1 - k]) / k
            vy = float(cy[-1] - cy[-1 - k]) / k
            speed = float(np.hypot(vx, vy))
            fall = float(ys[-1] - ys[-1 - k]) / k       # image px/frame, +down
        tail = cy[max(0, int(len(cy) * 0.6)):]
        tail_min = float(np.min(np.abs(tail))) if len(tail) else 0.0

    return [n_shots, lx, ly, abs(lx), vcy, last_near, has_ll,
            fcx, fcy, abs(fcx), abs(fcy), vx, vy, speed,
            maxcy, maxcx, tail_min, fall, n_track]


# --------------------------------------------------------------------------
# assemble the labelled dataset across all matches
# --------------------------------------------------------------------------

def build_dataset():
    serve_rows, end_rows = [], []
    for mid in config.match_ids():
        cfg = config.load(mid)
        Hm = np.load(cfg.homography)
        offsets = cfg.load_offsets()
        serves = cfg.load_serves()
        mchart = {r["clip"]: r for r in
                  csv.DictReader(open(cfg.charts_dir / "match_chart_v2.csv"))}
        _, records = evaluate.evaluate(cfg, verbose=False)
        for rec in records:
            clip = rec["clip"]
            played = rec["played"]
            rows = _chart_rows(cfg, clip)
            track = _court_track(cfg, clip, Hm, offsets)
            srv_row = serves.get(clip, {})
            mc = mchart.get(clip, {})

            # ---- serve placement ----
            s_lab = played[0] if played and played[0] in "456" else None
            if s_lab is not None:
                srv = next((r for r in rows if r["is_serve"] == "True"), None)
                base = srv["zone"] if srv and srv["zone"] in ("4", "5", "6") else "?"
                serve_rows.append({
                    "match": mid, "clip": clip,
                    "x": serve_features(rows, srv_row, track),
                    "y": s_lab, "baseline": base,
                    "committed": base != "?"})

            # ---- ending type (4-class: * n w d; drop rare 'x') ----
            e_lab = mcp_ending_type(played)
            if e_lab in ("*", "n", "w", "d"):
                base_e = our_ending_type(mc.get("ending", "?"))
                if base_e not in ("*", "n", "w", "d"):
                    base_e = "?"
                end_rows.append({
                    "match": mid, "clip": clip,
                    "x": ending_features(rows, track),
                    "y": e_lab, "baseline": base_e,
                    "committed": base_e != "?"})
    return serve_rows, end_rows


# --------------------------------------------------------------------------
# leave-one-match-out evaluation
# --------------------------------------------------------------------------

def lomo(rows, n_trees=200, max_depth=6, min_leaf=4):
    """Returns per-row model prediction + max-class probability, filled by
    a forest that never trained on that row's match."""
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    match_of = np.array([r["match"] for r in rows])
    pred = np.empty(len(rows), dtype=object)
    conf = np.zeros(len(rows))
    for held in sorted(set(match_of)):
        tr, te = match_of != held, match_of == held
        rf = RandomForest(n_trees=n_trees, max_depth=max_depth,
                          min_leaf=min_leaf, seed=SEED).fit(X[tr], y[tr])
        proba = rf.predict_proba(X[te])
        pred[te] = rf.classes_[proba.argmax(axis=1)]
        conf[te] = proba.max(axis=1)
    return X, y, match_of, pred, conf


def per_match_table(y, match_of, model_ok, base_ok):
    lines = []
    mids = sorted(set(match_of))
    for mid in mids + ["POOLED"]:
        m = np.ones(len(y), bool) if mid == "POOLED" else match_of == mid
        n = int(m.sum())
        ma = model_ok[m].mean() if n else 0.0
        ba = base_ok[m].mean() if n else 0.0
        d = ma - ba
        star = "  <-- learned wins" if d > 0 else ("  (tie)" if d == 0 else "")
        lines.append(f"  {mid:8} n={n:4d}   learned {ma:6.1%} ({int(model_ok[m].sum()):3d}/{n})"
                     f"   baseline {ba:6.1%} ({int(base_ok[m].sum()):3d}/{n})"
                     f"   Δ {d:+6.1%}{star}")
    return "\n".join(lines)


def coverage_curve(y, pred, conf, ks=(1.0, 0.75, 0.5, 0.25)):
    ok = (pred == y)
    order = np.argsort(-conf)
    n = len(y)
    lines = []
    for k in ks:
        top = order[:max(1, int(round(n * k)))]
        acc = ok[top].mean()
        lines.append(f"    keep top {int(k*100):3d}%  (n={len(top):4d})   accuracy {acc:6.1%}"
                     f"   min-conf {conf[top].min():.2f}")
    return "\n".join(lines)


def confusion(y, pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y, pred):
        M[idx[t], idx[p]] += 1
    hdr = "        " + "".join(f"{c:>6}" for c in classes) + "   (rows=truth)"
    body = [hdr]
    for c in classes:
        body.append(f"  {c:>5} " + "".join(f"{M[idx[c], idx[cc]]:6d}" for cc in classes))
    return "\n".join(body)


def run():
    serve_rows, end_rows = build_dataset()
    out = []
    out.append("# Learned vs hand-tuned — serve zone & ending type\n")
    out.append(f"_Generated {datetime.date.today().isoformat()} — "
               f"experiments/learn_components.py, deterministic (seed={SEED})._\n")
    out.append("Leave-one-match-out over 7 matches. The forest trains on 6, "
               "predicts the held-out 7th; a match never trains on itself. "
               "Learned and hand-tuned heuristic are scored on the IDENTICAL "
               "held-out labelled points. Baseline refusal ('?') counts as "
               "wrong (the confidence layer's rule); model may not abstain.\n")

    for name, rows, classes in [
            ("SERVE PLACEMENT (zone 4=wide / 5=body / 6=T)", serve_rows,
             ["4", "5", "6"]),
            ("ENDING TYPE (*=winner / n=net / w=wide / d=deep)", end_rows,
             ["*", "n", "w", "d"])]:
        X, y, match_of, pred, conf = lomo(rows)
        model_ok = (pred == y).astype(float)
        base = np.array([r["baseline"] for r in rows])
        committed = np.array([r["committed"] for r in rows])
        base_ok = (base == y).astype(float)

        n = len(y)
        maj = max((y == c).mean() for c in classes)
        pooled_m = model_ok.mean()
        pooled_b = base_ok.mean()
        comm_rate = committed.mean()
        base_ok_comm = base_ok[committed].mean() if committed.any() else 0.0

        out.append(f"\n## {name}\n")
        out.append(f"labelled held-out points: n = {n}")
        out.append(f"class balance: " +
                   ", ".join(f"{c}={int((y==c).sum())}" for c in classes) +
                   f"  (majority-class floor {maj:.1%})")
        out.append(f"\nPOOLED LOMO accuracy:")
        out.append(f"  learned forest         {pooled_m:6.1%}  ({int(model_ok.sum())}/{n})")
        out.append(f"  hand-tuned (refuse=wrong){pooled_b:6.1%}  ({int(base_ok.sum())}/{n})"
                   f"   <- apples-to-apples baseline")
        out.append(f"  hand-tuned committed-only {base_ok_comm:6.1%}  "
                   f"({int(base_ok[committed].sum())}/{int(committed.sum())})   "
                   f"[heuristic committed on {comm_rate:.0%} of points]")
        out.append(f"  Δ learned − baseline    {pooled_m - pooled_b:+6.1%}\n")
        out.append("per held-out match:")
        out.append(per_match_table(y, match_of, model_ok, base_ok))
        out.append("\nconfidence → coverage (auto-file the confident, ask about the rest):")
        out.append(coverage_curve(y, pred, conf))
        out.append("\nlearned-model confusion (LOMO pooled):")
        out.append(confusion(y, pred, classes))
        out.append("")

        # stash for return payload
        if name.startswith("SERVE"):
            serve_summary = (n, pooled_m, pooled_b, base_ok_comm, comm_rate,
                             int(model_ok.sum()), int(base_ok.sum()), maj)
        else:
            end_summary = (n, pooled_m, pooled_b, base_ok_comm, comm_rate,
                           int(model_ok.sum()), int(base_ok.sum()), maj)

    out.append("\n## Feature lists (leakage audit)\n")
    out.append("serve: " + ", ".join(SERVE_FEATURES))
    out.append("ending: " + ", ".join(END_FEATURES))
    out.append("\nNeither our emitted serve zone digit nor our emitted ending "
               "token appears as a feature — only raw ball geometry/kinematics "
               "and staging signals. Labels come from the aligned MCP chart.\n")

    report = "\n".join(out)
    dest = ROOT / "outputs" / "diag" / "learn_components.md"
    dest.write_text(report)
    print(report)
    print(f"\n-> {dest}")
    return serve_summary, end_summary


if __name__ == "__main__":
    run()
