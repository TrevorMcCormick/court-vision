"""Pose letters v1 — skeletons for the smudges (blueprint roadmap #3).

The forehand/backhand call has been stuck reading which side of a
player-BLOB the ball arrives on (t3 67/85, t4 17/31 committed-aligned;
hand-tuning counterfactually refuted). The research's highest-leverage
cheap swap: pose skeletons (rtmlib, Apache-2.0, auto-downloaded ONNX
weights, CPU-fast). This experiment runs the week-one gate from the
blueprint: (a) VISUAL AUDIT — can the far player's wrists be trusted at
720p? — and (b) a first head-to-head: a wrist-side letter call vs the
shipped blob rule, same shots, same truth.

Features per non-serve shot at its contact frame (all four players in
t3/t4 are right-handed, per config):
  A. wrist-cross: the racquet (right) wrist's side of the torso midline
     (shoulder midpoint), mirrored for the far player (they face the
     camera, so image-right is their left).
  B. ball-side vs torso midline (the blob rule with a pose anchor) —
     the re-anchoring control the counterfactual experiment predicts
     will NOT move (letter_anchor_cf, REFUTED, 2026-07-20).
Grading: MCP truth letters on length-matched aligned clips only (the
eval convention), same denominator for pose calls and the shipped
chart letters.

Run:  PYTHONPATH=.:experiments uv run --with rtmlib --with onnxruntime \
          python experiments/pose_letters.py
"""

import csv

import cv2
import numpy as np

from courtvision import config
from courtvision.config import ROOT
from courtvision.mcp import mcp_point_tokens, parse_mcp

MATCHES = ["t3", "t4"]
FPS = 30.0
W_IMG, H_IMG = 1280, 720
# COCO-17 indices
L_SHO, R_SHO, L_WRI, R_WRI = 5, 6, 9, 10
AUDIT_N = 8                     # overlay stills saved per match

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def load_players(cfg, clip):
    p = cfg.out_dir / "players" / f"players_{clip}.csv"
    if not p.exists():
        return {}
    rows = {}
    for r in csv.DictReader(open(p)):
        rows[(r["player"], int(r["frame"]))] = r
    return rows


def box_at(players, who, f):
    for df in (0, -1, 1, -2, 2):
        r = players.get((who, f + df))
        if r:
            cx, cy = float(r["cx"]) * W_IMG, float(r["cy"]) * H_IMG
            w, h = float(r["w"]) * W_IMG, float(r["h"]) * H_IMG
            return cx, cy, w, h
    return None


def best_pose(kpts, scores, box, who, Hm):
    """Pick the detection standing in the right half of the COURT.

    v1 matched nearest-to-blob-center and the audit stills caught it
    grading a ball kid: rogue blob boxes (the known box disease) drag
    the match to the wrong human. v1.1: project each detection's ankle
    midpoint through the homography; candidates must stand in `who`'s
    half within court bounds (ball kids kneel outside, at the fence, or
    at the net posts); take the highest-confidence candidate. The blob
    box is only a tiebreak hint.
    """
    if kpts is None or len(kpts) == 0:
        return None, None
    L_ANK, R_ANK = 15, 16
    cand = []
    for i in range(len(kpts)):
        feet = (kpts[i][L_ANK] + kpts[i][R_ANK]) / 2
        cm = cv2.perspectiveTransform(
            np.float32([[feet]]), Hm).reshape(2)
        x_ok = -1.5 <= cm[0] <= 12.5
        y_ok = (-4.0 <= cm[1] <= 11.0) if who == "far" else \
               (12.8 <= cm[1] <= 28.0)
        if x_ok and y_ok:
            cand.append(i)
    if not cand:
        return None, None
    if box is not None and len(cand) > 1:
        centers = kpts[cand].mean(axis=1)
        d = np.hypot(centers[:, 0] - box[0], centers[:, 1] - box[1])
        near = [c for c, dd in zip(cand, d) if dd < 2 * max(box[2], box[3], 60)]
        cand = near or cand
    i = max(cand, key=lambda c: float(scores[c].mean()))
    return kpts[i], scores[i]


def draw_pose(img, kp, sc):
    for a, b in SKELETON:
        if sc[a] > 0.3 and sc[b] > 0.3:
            cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     (60, 220, 255), 2, cv2.LINE_AA)
    for j in (L_WRI, R_WRI):
        if sc[j] > 0.3:
            cv2.circle(img, tuple(kp[j].astype(int)), 5, (0, 0, 255), -1)


def run_match(mid, body, out, diag):
    cfg = config.load(mid)
    Hm = np.load(cfg.out_dir / "H_img_to_court.npy")
    mcp = {r["clip"]: r for r in
           csv.DictReader(open(ROOT / "data" / "mcp" / f"{mid}_mcp_map.csv"))
           if r.get("status") == "matched"}
    cap_path = ROOT / "clips" / f"points_{mid}"
    tally = {"A": [0, 0], "B": [0, 0], "chart": [0, 0]}
    far_conf, near_conf = [], []
    audits = 0

    for clip, m in sorted(mcp.items()):
        _, _, played = parse_mcp(m.get("first", ""), m.get("second", ""))
        toks = mcp_point_tokens(played)
        true_letters = [t[0] for t in toks[1:-1]]        # non-serve shots
        ch = cfg.charts_dir / f"chart2_{clip}.csv"
        if not ch.exists():
            continue
        chart = list(csv.DictReader(open(ch)))
        if len(chart) - 1 != len(true_letters):          # length-matched only
            continue
        players = load_players(cfg, clip)
        cap = cv2.VideoCapture(str(cap_path / f"{clip}.mp4"))
        for i, row in enumerate(chart[1:]):
            truth = true_letters[i]
            if truth not in "fb":
                continue
            f = int(float(row["contact_frame"] or row["frame"]))
            who = row["striker"]
            box = box_at(players, who, f)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok or box is None:
                continue
            kpts, scores = body(img)
            kp, sc = best_pose(kpts, scores, box, who, Hm)
            if kp is None:
                continue
            (far_conf if who == "far" else near_conf).append(
                [sc[L_SHO], sc[R_SHO], sc[L_WRI], sc[R_WRI]])

            torso_x = (kp[L_SHO][0] + kp[R_SHO][0]) / 2
            mirror = -1.0 if who == "far" else 1.0       # far faces camera
            # A: racquet-wrist side of the torso
            call_a = "f" if (kp[R_WRI][0] - torso_x) * mirror > 0 else "b"
            tally["A"][0] += call_a == truth
            tally["A"][1] += 1
            # B: ball side of the torso (pose-anchored blob rule)
            bx = None
            bp = cfg.ball_dir / f"ball_{clip}.csv"
            if bp.exists():
                for br in csv.DictReader(open(bp)):
                    if int(br["frame"]) == f:
                        bx = float(br["x_stab"])
                        break
            if bx is not None:
                call_b = "f" if (bx - torso_x) * mirror > 0 else "b"
                tally["B"][0] += call_b == truth
                tally["B"][1] += 1
            # shipped chart letter on the same shot
            if row["letter"] in "fb":
                tally["chart"][0] += row["letter"] == truth
                tally["chart"][1] += 1
            if audits < AUDIT_N and (who == "far" or audits < 2):
                vis = img.copy()
                draw_pose(vis, kp, sc)
                cv2.rectangle(vis, (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)),
                              (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)),
                              (120, 255, 120), 1)
                cv2.imwrite(str(diag / f"pose_{clip}_s{i+2}_{who}.png"), vis)
                audits += 1
        cap.release()

    out.append(f"\n## {mid}")
    for k, label in (("A", "wrist-cross (pose)"), ("B", "ball-side, pose anchor"),
                     ("chart", "shipped blob rule (same shots)")):
        n, d = tally[k]
        out.append(f"  {label}: {n}/{d}"
                   f" ({n/d:.0%})" if d else f"  {label}: n=0")
    for name, arr in (("near", near_conf), ("far", far_conf)):
        if arr:
            a = np.array(arr)
            out.append(f"  {name}-player keypoint confidence "
                       f"(Lsho/Rsho/Lwri/Rwri): "
                       + "/".join(f"{v:.2f}" for v in a.mean(axis=0))
                       + f"  (n={len(arr)} shots)")


def run():
    from rtmlib import Body
    body = Body(mode="balanced", backend="onnxruntime", device="cpu")
    diag = ROOT / "outputs" / "diag"
    diag.mkdir(parents=True, exist_ok=True)
    out = ["# Pose letters v1 — skeletons vs the blob rule",
           "(rtmlib Body, CPU; all four players right-handed; graded on",
           " MCP truth letters, length-matched aligned clips, same shots",
           " for every method)"]
    for mid in MATCHES:
        run_match(mid, body, out, diag)
    report = "\n".join(out)
    print(report)
    (diag / "pose_letters_report.txt").write_text(report + "\n")
    print(f"\n[saved] {diag}/pose_letters_report.txt + pose_*.png audits")


if __name__ == "__main__":
    run()
