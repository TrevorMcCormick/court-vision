"""Full-match score-bug sweep -> plateaus (points) with scores.

Cheap two-pass fused: step through the 30fps reel, gate on name-region
presence + a changed-pixel test over the digits region (0.05 ms/frame),
and only pay the ~270 ms OCR when the digits crop actually changes. The
result is a chronological list of score PLATEAUS = candidate points, each
with (set,games,points,server) read by machine. This automates the two
by-eye staging knobs at once: point boundaries (normally boundaries.py's
per-broadcast CFG) and the score transcription (normally the by-eye
alignment CSV).

Writes experiments/g1_scorebug/sweep_raw.csv (per-sample) and
sweep_plateaus.csv (collapsed).
"""
import csv, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import read_bug as rb

VIDEO = "clips/g1_swiatek_paolini_30fps.mp4"
STEP = 6                      # sample every 6 frames (0.2 s)
DIG_Y = (628, 690)
DIG_X = (188, 296)            # games+points region, both eras
CHG_T = 0.020                 # changed-pixel fraction that flags a new score
PIX_T = 40
CONFIRM = 3                   # samples a new plateau must sustain
MIN_PLAT = 5                  # min samples (=1.0 s) for a real plateau


def dig_crop(frame):
    c = cv2.cvtColor(frame[DIG_Y[0]:DIG_Y[1], DIG_X[0]:DIG_X[1]], cv2.COLOR_BGR2GRAY)
    return cv2.resize(c, (108, 62))


def chg_frac(a, b):
    a = a.astype(np.int16); b = b.astype(np.int16)
    return (np.abs(a - b) > PIX_T).mean()


def sweep():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    last_ocr_crop = None
    last_res = None
    i = 0
    while i < n:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0:
            ok, fr = cap.retrieve()
            if ok:
                pres = rb._present(fr)
                if pres < 0.08:
                    rows.append((i, i / fps, 0, None))
                else:
                    crop = dig_crop(fr)
                    if last_ocr_crop is not None and chg_frac(crop, last_ocr_crop) < CHG_T:
                        res = last_res
                    else:
                        res = rb.read_frame(fr)
                        last_ocr_crop = crop
                        last_res = res
                    rows.append((i, i / fps, 1, res))
        i += 1
    cap.release()
    return rows, fps


def keyf(res):
    if not res or res.get("era") is None:
        return None
    return (res.get("setA", ""), res.get("setB", ""), res.get("gmA", ""),
            res.get("gmB", ""), res.get("ptsA", ""), res.get("ptsB", ""),
            res.get("server"))


def collapse(rows, fps):
    """Collapse consecutive equal score-keys into plateaus, tolerating brief
    bug-absence and single-sample OCR flickers (CONFIRM)."""
    plats = []
    cur = None          # {key, f0, f1, samples}
    pending = None; pend_n = 0
    for (i, t, pres, res) in rows:
        k = keyf(res) if pres else None
        if k is None:
            continue
        if cur and k == cur["key"]:
            cur["f1"] = i; cur["n"] += 1; pending = None; pend_n = 0
            continue
        # different key
        if pending == k:
            pend_n += 1
        else:
            pending = k; pend_n = 1
        if pend_n >= CONFIRM:
            if cur and cur["n"] >= MIN_PLAT:
                plats.append(cur)
            cur = {"key": k, "f0": i - (pend_n - 1) * STEP, "f1": i, "n": pend_n}
            pending = None; pend_n = 0
    if cur and cur["n"] >= MIN_PLAT:
        plats.append(cur)
    return plats


if __name__ == "__main__":
    rows, fps = sweep()
    outdir = Path(__file__).parent
    with open(outdir / "sweep_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "present", "key"])
        for (i, t, pres, res) in rows:
            w.writerow([i, f"{t:.2f}", pres, keyf(res) if pres else ""])
    plats = collapse(rows, fps)
    with open(outdir / "sweep_plateaus.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "f0", "f1", "t0", "t1", "n", "setA", "setB",
                    "gmA", "gmB", "ptsA", "ptsB", "server"])
        for j, p in enumerate(plats):
            k = p["key"]
            w.writerow([j, p["f0"], p["f1"], f"{p['f0']/fps:.1f}", f"{p['f1']/fps:.1f}",
                        p["n"], *k])
    print(f"samples={len(rows)} present={sum(r[2] for r in rows)} plateaus={len(plats)}")
