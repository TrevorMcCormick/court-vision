"""COUNTERFACTUAL: does a body-anchored letter reference beat the
shipped blob-center-x read?

No re-charting. For every ALIGNED rally letter the eval counts (the
shipped chart committed f/b on a length-matched clip), recompute the
letter under four candidate references at the SAME contact frame the
shipped chart used, reading the player-box CSV and ball track already
on disk. Score each ref against MCP truth side (f/b) exactly as
courtvision.evaluate does.

  ref0  blob center-x            shipped baseline; MUST reproduce the
                                 committed strict-letter numbers
  ref1  foot_x                   torso proxy immune to the motion trail
  ref2  neighbor-median cx       median center-x over +-K frames, to
                                 shed the single-frame motion smear
  ref3  blob center-x + abstain  refuse when box width > WIDE_MULT x the
                                 clip's typical width OR |ball_x - cx|
                                 < MIN_MARGIN px (guessing on a smear)

The letter math is lifted verbatim from courtvision.letters.commit:
  right = dxp > 0 (near) / dxp < 0 (far);  forehand = right != lefty;
  letter = 'f' if forehand else 'b'.  dxp = ball_x - reference_x.

Denominator is FIXED to the eval's committed-aligned letter set (shipped
letter in f/b, aligned clip, MCP side f/b) so every ref is scored on the
same shots. ref3 additionally abstains, so it reports right/committed
plus an abstain count.

Thresholds (WIDE_MULT, MIN_MARGIN, K) are tuned on t4 ONLY; t3 is the
held-out transfer / no-harm gate (the shipped ~67/85 must not regress).

Usage:
    uv run python experiments/letter_anchor_cf.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np

from courtvision.config import load as load_cfg
from courtvision.mcp import FH_SIDE, BH_SIDE, parse_mcp
from courtvision import boxes

# --- tunables (tuned on t4 only; t3 held out) ---
WIDE_MULT = 1.8      # ref3 refuses a box wider than this x the clip typical
MIN_MARGIN = 10.0    # ref3 refuses when |ball_x - ref| under this many px
NEIGH_K = 3          # ref2 median half-window (frames)


def letter_from(dxp, side, lefty_side):
    """courtvision.letters.commit, verbatim."""
    right = dxp > 0 if side == "near" else dxp < 0
    forehand = right != lefty_side
    return "f" if forehand else "b"


def clip_typ_width(players, side):
    """Clip-level typical box width (px): median over the side's boxes."""
    ws = [float(r[side]["w"]) * 1280 for r in players.values() if side in r]
    return float(np.median(ws)) if ws else None


def neighbor_median_cx(players, side, cf, k):
    """Median box-center-x (px) over frames [cf-k, cf+k] where the box
    exists — sheds the single-frame motion smear."""
    xs = []
    for f2 in range(cf - k, cf + k + 1):
        r = players.get(f2, {}).get(side)
        if r is not None:
            xs.append(float(r["cx"]) * 1280)
    return float(np.median(xs)) if xs else None


def run_match(t, wide_mult, min_margin, neigh_k):
    cfg = load_cfg(t)
    Hm = np.load(cfg.homography)
    offsets = cfg.load_offsets()
    lefty = cfg.lefty
    mapd = {r["clip"]: r for r in csv.DictReader(open(cfg.eval.mcp_map))}
    match = {r["clip"]: r for r in
             csv.DictReader(open(cfg.charts_dir / "match_chart_v2.csv"))}

    # per ref: [right, total]; ref3 also tracks abstain count
    res = {r: [0, 0] for r in ("ref0", "ref1", "ref2", "ref3")}
    ref3_abstain = 0
    ref0_mismatch = []          # correctness: recomputed ref0 != shipped

    for clip, mc in match.items():
        m = mapd[clip]
        if m["status"] != "matched":
            continue
        _serve_d, strokes, _played = parse_mcp(m["first"], m["second"])
        shots = list(csv.DictReader(open(cfg.charts_dir / f"chart2_{clip}.csv")))
        if len(shots) != 1 + len(strokes):
            continue                      # aligned (length-matched) only

        players = boxes.load(cfg.players_dir / f"players_{clip}.csv", Hm,
                             offsets.get(clip, (0.0, 0.0)))
        ball = {int(r["frame"]): float(r["x_stab"]) for r in
                csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv"))}
        typ_w = {s: clip_typ_width(players, s) for s in ("near", "far")}

        for k, sh in enumerate(shots):
            if sh["is_serve"] == "True" or sh["letter"] in ("?", ""):
                continue
            mcp_idx = k - 1
            if mcp_idx < 0 or mcp_idx >= len(strokes):
                continue
            ch = strokes[mcp_idx]
            mcp_side = ("f" if ch in FH_SIDE else "b" if ch in BH_SIDE else "?")
            if mcp_side == "?":
                continue

            side = sh["striker"]
            cf = int(sh["contact_frame"])
            row = players.get(cf, {}).get(side)
            if row is None or cf not in ball:
                # shipped committed a letter here, so this should not
                # happen — flag loudly rather than silently drop
                ref0_mismatch.append((clip, k, "no-box-or-ball"))
                continue
            bx = ball[cf]
            cx = float(row["cx"]) * 1280

            # ref0 — blob center-x (shipped baseline)
            let0 = letter_from(bx - cx, side, lefty[side])
            if let0 != sh["letter"]:
                ref0_mismatch.append((clip, k, f"{let0}!={sh['letter']}"))
            res["ref0"][0] += let0 == mcp_side
            res["ref0"][1] += 1

            # ref1 — foot_x
            foot = float(row["foot_x"])
            let1 = letter_from(bx - foot, side, lefty[side])
            res["ref1"][0] += let1 == mcp_side
            res["ref1"][1] += 1

            # ref2 — neighbor-median center-x
            nmx = neighbor_median_cx(players, side, cf, neigh_k)
            let2 = letter_from(bx - nmx, side, lefty[side])
            res["ref2"][0] += let2 == mcp_side
            res["ref2"][1] += 1

            # ref3 — blob center-x with abstention on smeared/ambiguous
            wide = typ_w[side] is not None and \
                float(row["w"]) * 1280 > wide_mult * typ_w[side]
            thin = abs(bx - cx) < min_margin
            if wide or thin:
                ref3_abstain += 1
            else:
                res["ref3"][0] += let0 == mcp_side
                res["ref3"][1] += 1

    return res, ref3_abstain, ref0_mismatch


def report(t, wide_mult, min_margin, neigh_k, tuned_note):
    res, abst, mism = run_match(t, wide_mult, min_margin, neigh_k)
    print(f"=== {t} {tuned_note} "
          f"(WIDE_MULT={wide_mult}, MIN_MARGIN={min_margin}, K={neigh_k}) ===")
    if mism:
        print(f"  !!! ref0 correctness FAIL: {len(mism)} shots differ from "
              f"shipped -> {mism[:8]}")
    else:
        print(f"  ref0 correctness OK: recomputed = shipped on all "
              f"{res['ref0'][1]} committed-aligned letters")
    base = res["ref0"][0]
    for r in ("ref0", "ref1", "ref2", "ref3"):
        rt, tot = res[r]
        d = rt - base
        extra = f"   ({abst} abstained)" if r == "ref3" else ""
        acc = f"{100 * rt / tot:.0f}%" if tot else "n/a"
        print(f"  {r}: {rt}/{tot} ({acc})   delta_right vs ref0 {d:+d}{extra}")
    print()
    return res, abst, mism


if __name__ == "__main__":
    # ---- tuning sweep on t4 ONLY ----
    print("##### t4 tuning sweep (diseased match; thresholds tuned here) #####\n")
    for K in (2, 3, 5):
        report("t4", WIDE_MULT, MIN_MARGIN, K, f"[tune K={K}]")
    for wm in (1.5, 1.8, 2.2):
        for mm in (6.0, 10.0, 15.0):
            report("t4", wm, mm, NEIGH_K, f"[tune ref3 wm={wm} mm={mm}]")

    # ---- final report at chosen thresholds ----
    print("##### FINAL (chosen thresholds) #####\n")
    report("t4", WIDE_MULT, MIN_MARGIN, NEIGH_K, "[diseased]")
    report("t3", WIDE_MULT, MIN_MARGIN, NEIGH_K, "[HELD-OUT no-harm gate]")
