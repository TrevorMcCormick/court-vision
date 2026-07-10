"""Point boundaries from the score bug: segment != point, the bug is the ID.

T3/T4 staging ended on the finding that the reel's edit grammar can't be
trusted to segment points: the RG editor fragments long clay rallies
across probe segments (t3 charted 3.9 shots vs MCP 8.9) and the
Wimbledon feed rarely cuts between points, so one segment holds 2-3 of
them (t4 charted 8.1 vs 7.2). Montreal cut once per point and did this
work for free.

THE SCORE BUG IS THE POINT ID. It is a screen-space overlay (camera pan
never moves it), static within a point, and it changes exactly at point
boundaries. Validated on pixels before any of this was written:
  - t3 segs 12-15 (four clips transcribed "30-15 dup"): ONE unbroken
    digit plateau across all four segments AND their gaps -> one live
    16-shot rally the probe fragmented. Same story segs 17+18 (the
    29-shot rally) and 30+31+32.
  - t3 segs 5/6/8 (all "40-AD"): separated by a 40-40 plateau in the
    gaps -> three DIFFERENT deuce-cycle points, not replays. The
    plateau ORDER carries information the value alone cannot.
  - t4 seg 3: a clean sustained digit step mid-segment -> two points in
    one segment; split at the step.
  - metric margin: same-point crops read changed-pixel fraction
    <= 0.005, different-score crops >= 0.033 (brightness-normalized,
    |diff| > 45). Threshold 0.015 sits in a 6x gap.
  - the small mean-diff step between t3 segs 17/18 was background bleed
    at the crop edge, NOT a digit change (crops rendered and read:
    both "2-4 / 40-30") — hence the changed-pixel metric, not mean diff.

Mechanisms, in order:
  presence  NCC of the bug NAMES region vs a reference template. The
            bug disappears (t4 crowd-shot false positives, RG wipes);
            absent frames are never compared. Segments whose frames are
            <30% bug-present are dropped as no-bug false positives.
  plateaus  digit-region crops labeled into runs of constant score.
            A change needs CONFIRM consecutive present frames past
            CHG_T (wipes and compression flicker don't sustain).
            A plateau survives bug absence < ABSENT_CLOSE_S and
            resumes if the returning value still matches.
  split     plateau boundary inside a segment = point boundary; pieces
            shorter than MIN_PIECE_S are dead-time slivers, dropped.
  merge     adjacent pieces with the SAME plateau id and a reel gap
            <= MERGE_GAP_S are one point; the gap becomes a ball-track
            hole the chart loop already handles.
  dead-air  pieces whose median inter-frame motion < DEAD_MED are
            walking-around footage with no play (t4 26b: med 0.09 vs
            every live piece >= 0.26) — dropped.
  replay    REFUTED, and the refutation is the story. Three dup-score
            groups suspected as replays all turned out to be fragments
            or distinct deuce points. The one cadence suspect (t3 seg
            62: isolated duplicate frames at the 1-in-6 rate a 2x
            slow-mo of a 50fps source would leave) was acquitted by
            ball physics: image-space gravity from the existing WASB
            tracks is unimodal across ALL 68 t3 clips (30th-pct |d2y|
            0.7-2.3 px/f^2, seg 62 at 1.78) and all 48 t4 clips — a 2x
            slow-mo clip would sit ~4x low. No probe-passing slow-mo
            replay exists in either reel, so no replay drop fires here;
            dup_frac is still recorded per piece for the day one shows
            up. (Cadence notes: true duplicate frames read motion
            <= 0.13, live frames >= 0.18 even in quiet wide shots; the
            25fps-sourced t4 reel duplicates 1 frame in 6 by resample
            cadence, isolated not run-length; stillness makes runs.)

Writes outputs/<tree>/segments_v2.csv (same schema as segments.csv plus
plateau/src/action columns), a bug_timeline.png receipt, and per-decision
crop receipts in outputs/<tree>/bug_checks/.

Lifted verbatim from experiments/point_boundary.py (frozen outputs;
the per-tree crop windows/eras below are measured per-broadcast
constants and stay in-module rather than in the match YAMLs).

Usage:
    uv run python -m courtvision.boundaries --tree t3
    uv run python -m courtvision.boundaries --tree t4 --no-cache
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CFG = {
    "t3": {
        "video": ROOT / "clips/t3_djokovic_ruud_30fps.mp4",
        # RG bug, bottom-left. names = the DJOKOVIC/RUUD plate (constant
        # all match, presence template). The "//" server marker (x 71-84)
        # and the speed / shot-clock overlays live outside both crops.
        #
        # THE BUG GROWS A COLUMN PER COMPLETED SET (rendered receipts:
        # scratchpad t3_bug_eras.png). games+points sit at x 199-253 in
        # set 1, slide to 226-280 in set 2 and 252-306 in set 3 as the
        # 7-6 / 6-3 columns are added; the set-1 tiebreak banner pushes
        # the whole bug DOWN 14 px. A fixed crop goes blind after set 1
        # (segs 48/49 and 64/65 read as one plateau each: the crop was
        # staring at the frozen "7 6 / 6 3" columns). Era windows are
        # per-match constants, boundaries placed in the long no-segment
        # gaps between sets — only segment frames need the right era.
        "names": (629, 683, 88, 185),
        "eras": [                    # (start_frame, digits_x0, digits_x1)
            (0, 199, 253),           # set 1
            (24200, 199, 253),       # set-1 tiebreak (banner shifts bug DOWN)
            (29000, 226, 280),       # set 2 (+"7-6" column)
            (41000, 252, 306),       # set 3 (+"6-3" column)
        ],
        "digits_y": (629, 683),
        "dys": tuple(range(0, 17, 2)),  # bug drifts down in the tiebreak era
        "ref_frame": 3081,           # known court view, bug up (seg 5)
    },
    "t4": {
        "video": ROOT / "clips/t4_krejcikova_paolini_30fps.mp4",
        # Wimbledon WTA bug, top-left. digits = serve marker + sets +
        # games + points cols. Layout is CONSTANT across sets (sets and
        # games columns exist from point one; receipt t4_bug_eras.png).
        # The ticking Rolex clock (y<25) and the cut-off in-house
        # scoreboard (x<110) are outside both crops.
        "names": (35, 90, 139, 254),
        "eras": [(0, 264, 355)],
        "digits_y": (35, 90),
        "dys": (0,),
        "ref_frame": 2384,           # known court view, bug up (seg 5)
    },
}

PRES_T = 0.80          # names NCC below this = bug absent
CHG_T = 0.015          # changed-pixel fraction: same point <=0.005, new score >=0.033
PIX_T = 45             # per-pixel |diff| that counts as changed
CONFIRM = 5            # present frames a change must sustain
SETTLE = 8             # present frames a new plateau settles before its
                       # ref anchors (median of the last 5): the RG update
                       # animation runs ~12 frames end to end
MIN_PIECE_S = 3.0      # split pieces shorter than this are point-boundary
                       # stubs (previous point's tail / next point's walk-up)
MERGE_GAP_S = 12.0     # max reel gap a same-plateau merge may bridge
ABSENT_CLOSE_S = 15.0  # bug absent longer than this closes the plateau
NO_BUG_FRAC = 0.30     # segment present-fraction below this = no-bug false positive
DUP_T = 0.15           # motion below this = true duplicate (live never dips under 0.18)
DEAD_MED = 0.15        # piece median motion below this = dead air, no play


def chg_frac(a, b):
    """Changed-pixel fraction, min over +-2 px shifts: the tiebreak-era
    bug drifts a few px against the presence template's dy quantization,
    and a real digit change is never a pure translation."""
    a = a.astype(np.float32)
    a = a - a.mean() + 128
    bf = b.astype(np.float32)
    bf = bf - bf.mean() + 128
    best = 1.0
    for sy in (-2, 0, 2):
        for sx in (-2, 0, 2):
            aa = a[2 + sy:a.shape[0] - 2 + sy, 2 + sx:a.shape[1] - 2 + sx]
            bb = bf[2:-2, 2:-2]
            best = min(best, float((np.abs(aa - bb) > PIX_T).mean()))
    return best


def ncc(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-6 else 0.0


def scan(tree, cache=True):
    """One sequential pass over the reel: presence, digit crop, motion."""
    cfg = CFG[tree]
    out = ROOT / "outputs" / tree / "bug_scan.npz"
    if cache and out.exists():
        z = np.load(out)
        return z["presence"], z["digits"], z["motion"], float(z["fps"])

    cap = cv2.VideoCapture(str(cfg["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg["ref_frame"])
    ok, ref = cap.read()
    assert ok
    y0, y1, x0, x1 = cfg["names"]
    ref_names = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gy0, gy1 = cfg["digits_y"]
    eras = cfg["eras"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    presence, digits, motion = [], [], []
    prev_small = None
    i = 0
    era_k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        # presence: best NCC over the dy grid — the tiebreak banner
        # shifts the whole bug down and it DRIFTS a few px within the
        # era, so the digit crop follows the per-frame winning dy
        best_p, best_dy = -1.0, 0
        for d in cfg["dys"]:
            p = ncc(g[y0 + d:y1 + d, x0:x1], ref_names)
            if p > best_p:
                best_p, best_dy = p, d
        presence.append(best_p)
        while era_k + 1 < len(eras) and i >= eras[era_k + 1][0]:
            era_k += 1
        _, ex0, ex1 = eras[era_k]
        digits.append(g[gy0 + best_dy:gy1 + best_dy, ex0:ex1].copy())
        small = cv2.resize(g, (320, 180)).astype(np.int16)
        motion.append(float(np.abs(small - prev_small).mean())
                      if prev_small is not None else -1.0)
        prev_small = small
        i += 1
        if i % 10000 == 0:
            print(f"  scan {i}")
    presence = np.array(presence, np.float32)
    digits = np.stack(digits)
    motion = np.array(motion, np.float32)
    np.savez_compressed(out, presence=presence, digits=digits,
                        motion=motion, fps=fps)
    print(f"-> {out} ({len(presence)} frames)")
    return presence, digits, motion, fps


def label_plateaus(presence, digits, fps):
    """Per-frame plateau id (-1 = bug absent / undecided).

    The ref is FROZEN per plateau (a rolling ref glides through the
    RG update animation: ~12 frames of ~0.01/frame steps, never 5
    consecutive over threshold — pixel trace in the scratchpad), but it
    is anchored on the MEDIAN of a settling window, not on the first
    frame seen: a ref frozen on a fade-in transition frame sits at
    threshold distance from the whole settled plateau and lets noise
    fake a change 100 frames later (the seg-4 false split)."""
    n = len(presence)
    plateau = np.full(n, -1, np.int32)
    cur_id = -1
    ref = None
    collect = None                    # settling frames before ref is anchored
    last_present = -10 ** 9
    pending = []                      # frames of a sustained change run
    for i in range(n):
        if presence[i] < PRES_T:
            pending = []
            continue
        if (i - last_present) / fps > ABSENT_CLOSE_S:
            ref = None                # long absence closes the plateau
            collect = None
        last_present = i
        if ref is None:
            if collect is None:       # a new plateau opens here
                cur_id += 1
                collect = []
            collect.append(i)
            plateau[i] = cur_id
            if len(collect) >= SETTLE:
                ref = np.median(np.stack([digits[j] for j in collect[-5:]]),
                                axis=0).astype(np.uint8)
                collect = None
            continue
        if chg_frac(digits[i], ref) > CHG_T:
            pending.append(i)
            if len(pending) >= CONFIRM:
                cur_id += 1
                for j in pending:
                    plateau[j] = cur_id
                pending = []
                ref = None            # re-anchor on the settled new score
                collect = [i]         # plateau id already opened above
            # undecided frames stay -1 until confirmed
        else:
            for j in pending:                  # flicker, not a change
                plateau[j] = cur_id
            pending = []
            plateau[i] = cur_id
    return plateau


def seg_plateau_runs(plateau, a, b, fps):
    """Runs of constant plateau id over PRESENT frames in [a, b]."""
    runs = []
    for i in range(a, b + 1):
        p = int(plateau[i])
        if p < 0:
            continue
        if runs and runs[-1][0] == p:
            runs[-1][2] = i
        else:
            runs.append([p, i, i])
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, choices=("t3", "t4"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    tree = args.tree
    out_dir = ROOT / "outputs" / tree
    checks = out_dir / "bug_checks"
    checks.mkdir(parents=True, exist_ok=True)
    for old in checks.glob(f"{tree}_*.png"):    # receipts of the current run only
        old.unlink()

    presence, digits, motion, fps = scan(tree, cache=not args.no_cache)
    plateau = label_plateaus(presence, digits, fps)
    print(f"{tree}: {plateau.max() + 1} plateaus over {len(presence)} frames")

    segs = [(int(r["seg"]), int(r["start_frame"]), int(r["end_frame"]))
            for r in csv.DictReader(open(out_dir / "segments.csv"))]

    # ---- split: pieces of constant plateau within each segment ----
    pieces = []          # {a, b, plateau, src, note}
    for k, a, b in segs:
        pres_frac = float(np.mean(presence[a:b + 1] >= PRES_T))
        if pres_frac < NO_BUG_FRAC:
            pieces.append({"a": a, "b": b, "plateau": -1, "src": f"{k}",
                           "action": "dropped_no_bug",
                           "note": f"bug present {pres_frac:.0%}"})
            continue
        runs = seg_plateau_runs(plateau, a, b, fps)
        # absorb micro-runs (< CONFIRM frames of present data) into neighbors
        runs = [r for r in runs if r[2] - r[1] + 1 >= CONFIRM]
        if not runs:
            pieces.append({"a": a, "b": b, "plateau": -1, "src": f"{k}",
                           "action": "dropped_no_bug", "note": "no stable plateau"})
            continue
        # piece boundaries: segment edges + midpoints of plateau handovers
        cuts = [a]
        for r_prev, r_next in zip(runs, runs[1:]):
            cuts.append((r_prev[2] + r_next[1]) // 2 + 1)
        cuts.append(b + 1)
        for ci, (p0, p1) in enumerate(zip(cuts[:-1], cuts[1:])):
            pid = runs[ci][0]
            src = f"{k}" if len(runs) == 1 else f"{k}{chr(97 + ci)}"
            dur = (p1 - p0) / fps
            if dur < MIN_PIECE_S:
                pieces.append({"a": p0, "b": p1 - 1, "plateau": pid,
                               "src": src, "action": "dropped_sliver",
                               "note": f"{dur:.1f}s"})
                continue
            pieces.append({"a": p0, "b": p1 - 1, "plateau": pid, "src": src,
                           "action": "split" if len(runs) > 1 else "keep",
                           "note": ""})

    # ---- merge: same plateau id across a small reel gap ----
    merged = []
    for pc in pieces:
        if pc["action"].startswith("dropped"):
            merged.append(pc)
            continue
        prev = next((m for m in reversed(merged)
                     if not m["action"].startswith("dropped")), None)
        if (prev is not None and prev["plateau"] == pc["plateau"]
                and pc["plateau"] >= 0
                and (pc["a"] - prev["b"]) / fps <= MERGE_GAP_S):
            prev["b"] = pc["b"]
            prev["src"] += f"+{pc['src']}"
            prev["action"] = "merged"
            continue
        merged.append(pc)

    # ---- dead-air drop + dup_frac bookkeeping (replay: see docstring) ----
    for pc in merged:
        if pc["action"].startswith("dropped"):
            continue
        m = motion[pc["a"] + 1:pc["b"] + 1]
        m = m[m >= 0]
        pc["dup_frac"] = float(np.mean(m < DUP_T)) if len(m) else 0.0
        med = float(np.median(m)) if len(m) else 0.0
        if med < DEAD_MED:
            pc["action"] = "dropped_dead_air"
            pc["note"] = f"median motion {med:.2f}"

    # ---- receipts: crop pairs for every merge and split ----
    for pc in merged:
        if pc["action"] not in ("merged", "split"):
            continue
        fa = pc["a"] + 5
        fb = pc["b"] - 5
        tile = []
        for lbl, f in (("start", fa), ("end", fb)):
            c = cv2.resize(digits[f], None, fx=4, fy=4,
                           interpolation=cv2.INTER_NEAREST)
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
            bar = np.full((20, c.shape[1], 3), 40, np.uint8)
            cv2.putText(bar, f"{pc['src']} {lbl} f{f} p{pc['plateau']}",
                        (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            tile.append(np.vstack([bar, c]))
        cv2.imwrite(str(checks / f"{tree}_{pc['action']}_{pc['src']}.png"),
                    np.vstack(tile))

    # ---- outputs ----
    keep = [pc for pc in merged if not pc["action"].startswith("dropped")]
    with open(out_dir / "segments_v2.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seg", "start_frame", "end_frame", "start_s", "dur_s",
                     "plateau", "src", "action", "dup_frac"])
        for k2, pc in enumerate(keep, 1):
            wr.writerow([k2, pc["a"], pc["b"], round(pc["a"] / fps, 2),
                         round((pc["b"] - pc["a"] + 1) / fps, 2),
                         pc["plateau"], pc["src"], pc["action"],
                         round(pc.get("dup_frac", 0.0), 3)])
    with open(out_dir / "segments_v2_dropped.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["start_frame", "end_frame", "src", "action", "note"])
        for pc in merged:
            if pc["action"].startswith("dropped"):
                wr.writerow([pc["a"], pc["b"], pc["src"], pc["action"],
                             pc["note"]])

    # ---- timeline receipt ----
    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    t = np.arange(len(presence)) / fps
    axes[0].plot(t, presence, lw=0.4)
    axes[0].axhline(PRES_T, color="r", ls=":")
    axes[0].set_ylabel("bug presence NCC")
    pl = plateau.astype(float)
    pl[pl < 0] = np.nan
    axes[1].plot(t, pl, lw=0.6)
    axes[1].set_ylabel("plateau id (point)")
    for _, a, b in segs:
        axes[2].axvspan(a / fps, b / fps, ymin=0.55, ymax=0.95,
                        color="#888888")
    for pc in keep:
        axes[2].axvspan(pc["a"] / fps, pc["b"] / fps, ymin=0.05, ymax=0.45,
                        color="#2e9e4f")
    for pc in merged:
        if pc["action"].startswith("dropped"):
            axes[2].axvspan(pc["a"] / fps, pc["b"] / fps, ymin=0.05,
                            ymax=0.45, color="#c0392b")
    axes[2].set_yticks([0.25, 0.75])
    axes[2].set_yticklabels(["v2 (green=keep, red=drop)", "v1 segments"])
    axes[2].set_xlabel("reel time (s)")
    fig.suptitle(f"{tree}: score-bug point boundaries — "
                 f"{len(segs)} probe segments -> {len(keep)} points")
    fig.tight_layout()
    fig.savefig(out_dir / "bug_timeline.png", dpi=110)

    n_m = sum(1 for pc in keep if pc["action"] == "merged")
    n_s = sum(1 for pc in keep if pc["action"] == "split")
    n_dn = sum(1 for pc in merged if pc["action"] == "dropped_no_bug")
    n_dr = sum(1 for pc in merged if pc["action"] == "dropped_replay")
    n_dv = sum(1 for pc in merged if pc["action"] == "dropped_sliver")
    print(f"{tree}: {len(segs)} segments -> {len(keep)} points "
          f"({n_m} merged, {n_s} split pieces, {n_dn} no-bug, "
          f"{n_dr} replay, {n_dv} slivers dropped)")
    for pc in keep:
        print(f"  seg {pc['src']:>12} f{pc['a']}-{pc['b']} "
              f"p{pc['plateau']} {pc['action']} dup={pc.get('dup_frac', 0):.2f}")
    print(f"-> {out_dir / 'segments_v2.csv'}, bug_timeline.png, bug_checks/")


if __name__ == "__main__":
    main()
