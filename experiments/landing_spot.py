"""Landing-spot extrapolation v1 — the boundary race (blueprint roadmap #1).

HYPOTHESIS. Wide-vs-deep ending recall is 0% because WASB loses the ball
before the out-of-bounds bounce, and both the shipped rules and the corpus
random forest only look at where the ball WAS (whole-clip aggregates /
last recorded landing). The 2025 literature (TT3D, "Where Is The Ball",
CVPR-W) says the landing spot should be INFERRED from the observed flight
— nobody photographs the ball touching the ground, not even Hawk-Eye.

METHOD (v1, pure geometry, zero training, zero new constants tuned).
For each benchmark point, isolate the FINAL FLIGHT SEGMENT: the contiguous
ball-track run after the last shot's contact, capped at flight time. Fit
straight lines cx(t), cy(t) to the segment tail in court coordinates and
race two crossings:

    sideline crossed first  -> wide (w)
    baseline crossed first  -> deep (d)
    both within ~2 frames   -> both (x)
    neither within the cap  -> abstain (?)

Known physics bias, stated up front: the court projection maps an AIRBORNE
ball to its ground shadow, which exaggerates DEPTH (motion along the
camera axis) far more than WIDTH. So the baseline crossing fires early
(bias toward 'd'); the sideline reading is comparatively trustworthy.
v1 accepts that bias and measures it rather than tuning around it.

SCOPE / HONESTY.
- Graded on the same population as experiments/learn_components.py
  (evaluate.evaluate records; truth = mcp_ending_type(played)).
- Primary question: among points that TRULY ended out (w/d/x), can the
  flight segment name the right error? Baseline to beat: 0% (both the
  shipped rules and the LOMO forest score 0/63 wide, 0/86 deep).
- Secondary (integration risk): how often does the racer call 'out' flavors
  on points that truly ended '*' (winner) or 'n' (net)? Reported, not
  optimized. v1 is a diagnosis, not a shipped ending detector.
- Constants below reuse existing court geometry (court.py, endings.py
  OUT_MARGIN); the only new numbers are the segment/extrapolation windows,
  chosen a priori from tennis flight time, not fitted to truth.

Run:  uv run python experiments/landing_spot.py
"""

import csv

import cv2
import numpy as np

from courtvision import config, evaluate
from courtvision.config import ROOT
from courtvision.court import W_C, L_C, NET_Y, SINGLES_MARGIN
from courtvision.mcp import mcp_ending_type

FPS = 30.0                 # every staged reel is normalised to 30fps (ingest)
OUT_MARGIN = 0.25          # meters of slack before 'out' — endings.py value
FLIGHT_CAP_S = 2.0         # max segment length after final contact
EXTRAP_CAP_S = 0.7         # how far past track death the fit may be trusted
TAIL_K = 8                 # samples in the line fit
HOLE_FRAMES = 6            # a gap this big ends the contiguous segment
MAX_M_PER_FRAME = 3.0      # court meters/frame beyond any real ball: a
                           # teleport ends the segment (physics, not tuning
                           # — ~90 m/s; boxes.py has the same gate for players)
BOTH_WIN_FR = 2.0          # crossings this close together -> 'x'

SIDE_L = SINGLES_MARGIN            # singles sidelines in court-x
SIDE_R = W_C - SINGLES_MARGIN


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
    return {"frames": frames, "cx": court[:, 0], "cy": court[:, 1]}


def final_flight_segment(track, contact_frame):
    """Contiguous track run after the final contact, flight-time capped."""
    f, cx, cy = track["frames"], track["cx"], track["cy"]
    m = (f > contact_frame) & (f <= contact_frame + FLIGHT_CAP_S * FPS)
    if not m.any():
        return None
    f, cx, cy = f[m], cx[m], cy[m]
    # cut at the first hole: a fit across a hole is a fit across a fiction
    gaps = np.where(np.diff(f) > HOLE_FRAMES)[0]
    end = gaps[0] + 1 if len(gaps) else len(f)
    f, cx, cy = f[:end].astype(float), cx[:end], cy[:end]
    # cut at the first teleport: balls don't move 3 m per frame — a jump
    # like that is the tracker latching onto something else (found by eye
    # in the cv-18 hero render: t6_point_110 jumps 37 m in 3 frames)
    if len(f) > 1:
        step = np.hypot(np.diff(cx), np.diff(cy)) / np.diff(f)
        tele = np.where(step > MAX_M_PER_FRAME)[0]
        if len(tele):
            f, cx, cy = f[:tele[0] + 1], cx[:tele[0] + 1], cy[:tele[0] + 1]
    if len(f) == 0:
        return None
    return f, cx, cy


def infer_target_far(y0, by):
    """Which half is the final shot heading for? Read the flight itself —
    the chart's striker column is an error-prone stage this module should
    not inherit (found: a point whose 'far striker' ball died 2 m beyond
    the FAR baseline). Already-out position wins; else the y-slope; else
    which half the ball is in."""
    if y0 < 0:
        return True
    if y0 > L_C:
        return False
    if abs(by) > 1e-3:
        return by < 0
    return y0 < NET_Y


def race_boundaries(f, cx, cy):
    """Race the flight line to the sideline vs the baseline.

    Returns (call, t_side, t_base): call in {'w','d','x','?'} with times in
    frames past the last observed sample (0 = already out at track death).
    The target half (far: y=0 baseline; near: y=L_C) is inferred from the
    flight itself via infer_target_far.
    """
    if len(f) < 4:
        return "?", None, None
    k = min(TAIL_K, len(f))
    t, x, y = f[-k:], cx[-k:], cy[-k:]
    bx = np.polyfit(t, x, 1)[0]            # court m / frame
    by = np.polyfit(t, y, 1)[0]
    x0, y0 = x[-1], y[-1]
    cap = EXTRAP_CAP_S * FPS

    target_far = infer_target_far(y0, by)
    base_y = -OUT_MARGIN if target_far else L_C + OUT_MARGIN
    to_base = base_y - y0
    if target_far:
        t_base = 0.0 if y0 < base_y else (to_base / by if by < 0 else np.inf)
    else:
        t_base = 0.0 if y0 > base_y else (to_base / by if by > 0 else np.inf)

    side_x = (SIDE_R + OUT_MARGIN) if bx > 0 else (SIDE_L - OUT_MARGIN)
    already_wide = not (SIDE_L - OUT_MARGIN <= x0 <= SIDE_R + OUT_MARGIN)
    if already_wide:
        t_side = 0.0
    elif abs(bx) < 1e-9:
        t_side = np.inf
    else:
        t_side = (side_x - x0) / bx
        if t_side < 0:
            t_side = np.inf

    s_in = t_side <= cap
    b_in = t_base <= cap
    if s_in and b_in and abs(t_side - t_base) <= BOTH_WIN_FR:
        return "x", t_side, t_base
    if s_in and t_side <= t_base:
        return "w", t_side, t_base
    if b_in:
        return "d", t_side, t_base
    if s_in:
        return "w", t_side, t_base
    return "?", t_side, t_base


def build_rows():
    rows = []
    for mid in config.match_ids():
        cfg = config.load(mid)
        Hm = np.load(cfg.homography)
        offsets = cfg.load_offsets()
        _, records = evaluate.evaluate(cfg, verbose=False)
        for rec in records:
            clip, played = rec["clip"], rec["played"]
            truth = mcp_ending_type(played)
            if truth not in ("*", "n", "w", "d", "x"):
                continue
            chart = list(csv.DictReader(open(cfg.charts_dir / f"chart2_{clip}.csv")))
            track = _court_track(cfg, clip, Hm, offsets)
            if not chart or track is None:
                rows.append({"match": mid, "clip": clip, "truth": truth,
                             "call": "?", "why": "no chart/track"})
                continue
            last = chart[-1]
            contact = float(last["contact_frame"] or last["frame"])
            seg = final_flight_segment(track, contact)
            if seg is None:
                rows.append({"match": mid, "clip": clip, "truth": truth,
                             "call": "?", "why": "no post-contact track"})
                continue
            call, t_s, t_b = race_boundaries(*seg)
            rows.append({"match": mid, "clip": clip, "truth": truth,
                         "call": call, "why": "",
                         "n_seg": len(seg[0]), "t_side": t_s, "t_base": t_b})
    return rows


def confusion(rows, truths, calls=("w", "d", "x", "?")):
    lines = ["        " + "".join(f"{c:>6}" for c in calls) + "   (rows=truth)"]
    for tr in truths:
        sub = [r for r in rows if r["truth"] == tr]
        cnt = {c: sum(1 for r in sub if r["call"] == c) for c in calls}
        lines.append(f"  {tr:>5} " + "".join(f"{cnt[c]:6d}" for c in calls)
                     + f"   n={len(sub)}")
    return "\n".join(lines)


RACE_FEATURES = ["race_w", "race_d", "race_x", "race_abstain",
                 "t_side", "t_base", "n_seg"]


def race_feature_map(rows):
    """(match, clip) -> numeric feature vector from the racer's output."""
    def cap(t):
        return 200.0 if t is None or not np.isfinite(t) else min(float(t), 200.0)
    feats = {}
    for r in rows:
        feats[(r["match"], r["clip"])] = [
            1.0 if r["call"] == "w" else 0.0,
            1.0 if r["call"] == "d" else 0.0,
            1.0 if r["call"] == "x" else 0.0,
            1.0 if r["call"] == "?" else 0.0,
            cap(r.get("t_side")), cap(r.get("t_base")),
            float(r.get("n_seg", 0)),
        ]
    return feats


def lomo_with_racer(rows):
    """Stage 2 — apples-to-apples with the pivot experiment: the SAME
    4-class LOMO forest (learn_components), with vs without the racer's
    outputs as extra feature columns. Same seed, same forest params."""
    import learn_components as lc

    _, end_rows = lc.build_dataset()
    feats = race_feature_map(rows)
    aug_rows = []
    for r in end_rows:
        extra = feats.get((r["match"], r["clip"]),
                          [0.0, 0.0, 0.0, 1.0, 200.0, 200.0, 0.0])
        aug_rows.append({**r, "x": list(r["x"]) + extra})

    out = ["\nSTAGE 2 — corpus forest (4-class LOMO, seed 0), with vs"
           " without racer features"]
    for name, rset in (("baseline features (pivot experiment)", end_rows),
                       ("+ racer features", aug_rows)):
        _, y, _, pred, _ = lc.lomo(rset)
        out.append(f"\n  {name}: overall {np.mean(pred == y):.1%}  (n={len(y)})")
        for cls in ("*", "n", "w", "d"):
            m = y == cls
            if m.any():
                out.append(f"    recall {cls}: {np.sum(pred[m] == cls)}"
                           f"/{int(m.sum())} ({np.mean(pred[m] == cls):.0%})")
    return "\n".join(out)


def run():
    rows = build_rows()
    out = ["# Landing-spot extrapolation v1 — the boundary race\n"]

    outs = [r for r in rows if r["truth"] in ("w", "d", "x")]
    out.append(f"POINTS WITH A TRUE OUT-ENDING (w/d/x): n={len(outs)}")
    out.append("baseline to beat: 0% recall on w and d (rules AND forest)\n")
    out.append(confusion(rows, ("w", "d", "x")))
    for cls in ("w", "d"):
        sub = [r for r in rows if r["truth"] == cls]
        hit = sum(1 for r in sub if r["call"] == cls)
        loose = sum(1 for r in sub if r["call"] in ("w", "d", "x"))
        if sub:
            out.append(f"\n  recall {cls}: strict {hit}/{len(sub)}"
                       f" ({hit / len(sub):.0%});"
                       f" called-any-out {loose}/{len(sub)}"
                       f" ({loose / len(sub):.0%})")

    out.append("\nPER MATCH (true out-endings only): strict-correct / n")
    for mid in config.match_ids():
        sub = [r for r in outs if r["match"] == mid]
        if sub:
            hit = sum(1 for r in sub if r["call"] == r["truth"])
            out.append(f"  {mid}: {hit}/{len(sub)}")

    out.append("\nINTEGRATION RISK — behaviour on non-out truths:")
    out.append(confusion(rows, ("*", "n")))
    fired = [r for r in rows if r["truth"] in ("*", "n")
             and r["call"] in ("w", "d", "x")]
    allsn = [r for r in rows if r["truth"] in ("*", "n")]
    if allsn:
        out.append(f"  false out-calls on */n: {len(fired)}/{len(allsn)}"
                   f" ({len(fired) / len(allsn):.0%}) — v1 is a classifier"
                   " for known-out points, not yet an out detector")

    out.append(lomo_with_racer(rows))

    report = "\n".join(out)
    print(report)
    dest = ROOT / "outputs" / "diag" / "landing_spot_report.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report + "\n")
    print(f"\n[saved] {dest}")


if __name__ == "__main__":
    run()
