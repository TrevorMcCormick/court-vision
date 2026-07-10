"""Direction-signal quality measurement — t3 (TUNING SET) by default.

For every aligned rally pair on length-matched t3 points, compute each
of shot_direction.py's signals independently and score it against the
MCP digit: accuracy AND coverage, plus the pairwise agreement matrix
the precedence/refusal design is read off of. t1/t2/t4 are held out;
run them ONLY as a final report, never to tune.

Usage:
    uv run experiments/dir_signals_dev.py [t3]
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import shot_direction as sd
from mcp_accept import mcp_point_tokens

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"


def load_track(t, stem, Hm, offsets):
    ball = list(csv.DictReader(open(ROOT / "outputs" / t / "ball_wasb" / f"ball_{stem}.csv")))
    frames = np.array([int(r["frame"]) for r in ball])
    xs = np.array([float(r["x_stab"]) for r in ball])
    ys = np.array([float(r["y_stab"]) for r in ball])
    odx, ody = offsets.get(stem, (0.0, 0.0))
    pts = np.stack([xs - odx, ys - ody], axis=1).reshape(-1, 1, 2).astype(np.float32)
    court = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    return frames, court[:, 1], court[:, 0]


def to_shot(row):
    def num(v):
        return None if v in ("", "None", None) else float(v)
    return {"frame": int(row["frame"]),
            "contact_frame": (int(row["contact_frame"])
                              if row["contact_frame"] not in ("", "None") else None),
            "is_serve": row["is_serve"] == "True",
            "synth": row["synth"] == "True",
            "striker": row["striker"],
            "landing_x": num(row["landing_x"]),
            "landing_y": num(row["landing_y"])}


def main():
    t = sys.argv[1] if len(sys.argv) > 1 else "t3"
    Hm = np.load(ROOT / "outputs" / t / "H_img_to_court.npy")
    offsets = {}
    off_csv = ROOT / "outputs" / t / "clip_offsets.csv"
    if off_csv.exists():
        for r in csv.DictReader(open(off_csv)):
            offsets[r["clip"]] = (float(r["dx"]), float(r["dy"]))

    mapd = {r["clip"]: r for r in csv.DictReader(open(DATA / f"{t}_mcp_map.csv"))}
    chart = ROOT / "outputs" / t / "charts_wasb"
    match = {r["clip"]: r for r in csv.DictReader(open(chart / "match_chart_v2.csv"))}

    SIGS = ["land_near", "land_far", "contact", "cross"]
    acc = {s: [0, 0] for s in SIGS}
    cover = {s: 0 for s in SIGS}
    pair = {}
    est_acc = [0, 0]
    est_ref = 0
    n_shots = 0

    for clip, mc in match.items():
        m = mapd[clip]
        if m["status"] != "matched":
            continue
        played = m["second"] if m["second"].strip() else m["first"]
        mcp = mcp_point_tokens(played)
        rows = list(csv.DictReader(open(chart / f"chart2_{clip}.csv")))
        shots = [to_shot(r) for r in rows]
        toks_rows = list(shots)
        if shots and not shots[0]["is_serve"]:
            toks_rows.insert(0, None)
        toks_rows.append(None)
        if len(mcp) != len(toks_rows):
            continue
        frames, cyc, cxc = load_track(t, clip, Hm, offsets)
        fps = 25.0
        for k, (mt, sh) in enumerate(zip(mcp, toks_rows)):
            if sh is None or sh["is_serve"] or len(mt) != 2 or mt[1] not in "123":
                continue
            j = shots.index(sh)
            nxt = shots[j + 1] if j + 1 < len(shots) else None
            n_shots += 1
            got = {}
            land = sd.landing_signal(sh)
            if land is not None:
                key = "land_near" if land[1] == "near" else "land_far"
                got[key] = sd.direction_digit(*land)
            cont = sd.contact_signal(sh, nxt, frames, cxc)
            if cont is not None and cont[1] in ("near", "far"):
                got["contact"] = sd.direction_digit(*cont)
            cross = sd.crossing_signal(sh, nxt, frames, cyc, cxc, fps)
            if cross is not None:
                got["cross"] = sd.direction_digit(*cross)
            for s, d in got.items():
                cover[s] += 1
                acc[s][0] += d == mt[1]
                acc[s][1] += 1
            ks = sorted(got)
            for i in range(len(ks)):
                for jj in range(i + 1, len(ks)):
                    key = (ks[i], ks[jj])
                    ag = got[ks[i]] == got[ks[jj]]
                    st = pair.setdefault(key, [0, 0, 0, 0])  # n, agree, right-if-agree, right-first-if-disagree
                    st[0] += 1
                    st[1] += ag
                    if ag:
                        st[2] += got[ks[i]] == mt[1]
            d, why = sd.estimate(sh, nxt, frames, cyc, cxc, fps)
            if d == "?":
                est_ref += 1
            else:
                est_acc[0] += d == mt[1]
                est_acc[1] += 1

    print(f"=== {t}: direction signals on aligned rally pairs "
          f"(length-matched points), n={n_shots} ===\n")
    print(f"{'signal':12}{'coverage':>14}{'accuracy':>14}")
    for s in SIGS:
        a, b = acc[s]
        pc = f"{100 * cover[s] / n_shots:.0f}%" if n_shots else "-"
        pa = f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"
        print(f"{s:12}{f'{cover[s]}/{n_shots}':>9} {pc:>4}{pa:>16}")
    print("\npairwise (n / agree / right-when-agree):")
    for (s1, s2), (nn, ag, ra, _) in sorted(pair.items()):
        print(f"  {s1:10} vs {s2:10}  n={nn:<4} agree={ag:<4} "
              f"right-when-agree={ra}/{ag}" if ag else
              f"  {s1:10} vs {s2:10}  n={nn:<4} agree=0")
    a, b = est_acc
    print(f"\nestimator (current precedence): committed {b}/{n_shots} "
          f"({100 * b / max(n_shots, 1):.0f}%), right {a}/{b} "
          f"({100 * a / max(b, 1):.0f}%), refused {est_ref}")


if __name__ == "__main__":
    main()
