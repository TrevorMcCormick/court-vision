"""Two-pass score-bug sweep: cheap pixel plateaus, then bounded OCR.

Pass 1 (no OCR, ~0.05 ms/crop): over every 5 fps bug-crop, gate on
name-presence + points-box, and grow plateaus by the changed-pixel metric on
the digits signature. This segments the whole 84-min reel into score
plateaus without paying OCR for replays / the pre-match intro.
Pass 2: OCR ONE representative crop (plateau median) per surviving plateau
(length >= MIN_S). OCR cost is bounded by the number of plateaus (~hundreds),
not by frame count.

Output: experiments/g1_scorebug/sweep_plateaus.csv (chronological points).
"""
import csv, sys, glob, os
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import read_bug as rb
import sweep2 as sw2                       # reuse to_canvas / dig_crop / chg_frac / keyf

CROPS = sw2.CROPS
FPS_SAMPLE = 5
CHG_T = 0.020
MIN_S = 1.0                                 # min plateau seconds to keep


def pass1():
    files = sorted(glob.glob(os.path.join(CROPS, "f*.jpg")))
    plats = []          # {i0,i1,ref_sig,paths:[...]}
    cur = None
    for idx, fp in enumerate(files):
        canv = sw2.to_canvas(cv2.imread(fp))
        live = rb._present(canv) >= 0.08 and rb._points_box(canv) is not None
        if not live:
            if cur and (cur["i1"] - cur["i0"] + 1) >= MIN_S * FPS_SAMPLE:
                plats.append(cur)
            cur = None
            continue
        sig = sw2.dig_crop(canv)
        if cur is None:
            cur = {"i0": idx, "i1": idx, "ref": sig, "mid": fp}
        elif sw2.chg_frac(sig, cur["ref"]) < CHG_T:
            cur["i1"] = idx
        else:
            if (cur["i1"] - cur["i0"] + 1) >= MIN_S * FPS_SAMPLE:
                plats.append(cur)
            cur = {"i0": idx, "i1": idx, "ref": sig, "mid": fp}
    if cur and (cur["i1"] - cur["i0"] + 1) >= MIN_S * FPS_SAMPLE:
        plats.append(cur)
    # set mid path to the plateau midpoint
    for p in plats:
        p["mid_idx"] = (p["i0"] + p["i1"]) // 2
        p["mid"] = files[p["mid_idx"]]
    return plats, files


def pass2(plats, files):
    out = []
    for p in plats:
        # vote over up to 3 crops (25%,50%,75%) for robustness
        idxs = sorted(set([p["i0"] + (p["i1"] - p["i0"]) * q // 4 for q in (1, 2, 3)]))
        reads = [rb.read_frame(sw2.to_canvas(cv2.imread(files[i]))) for i in idxs]
        keys = [sw2.keyf(r) for r in reads if sw2.keyf(r)]
        if not keys:
            continue
        # majority key
        from collections import Counter
        key, _ = Counter(keys).most_common(1)[0]
        f0 = p["i0"] * 6; f1 = p["i1"] * 6
        out.append((f0, f1, p["i1"] - p["i0"] + 1, key))
    return out


if __name__ == "__main__":
    plats, files = pass1()
    print(f"pass1: {len(files)} crops -> {len(plats)} raw plateaus (>= {MIN_S}s)")
    reads = pass2(plats, files)
    outp = Path(__file__).parent / "sweep_plateaus.csv"
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "f0", "f1", "t0", "t1", "n", "setA", "setB", "gmA", "gmB", "ptsA", "ptsB", "server"])
        for j, (f0, f1, n, key) in enumerate(reads):
            w.writerow([j, f0, f1, f"{f0/30:.1f}", f"{f1/30:.1f}", n, *key])
    print(f"pass2: {len(reads)} plateaus with a valid score read -> {outp}")
