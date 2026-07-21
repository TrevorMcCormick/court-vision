"""Fast score-bug sweep over ffmpeg-extracted bug crops (5 fps).

Crops are 320x80 at frame origin (x=0, y=624); crop index i -> full-frame
number i*6 (5 fps from a 30 fps reel). Each crop is pasted into a small
canvas so the absolute-coord read_bug functions work unchanged. Cheap gates
(name presence, points-box) run every crop (~0.05 ms); the ~270 ms OCR fires
only when the digits region changes vs the last OCR'd crop. Output: a
chronological list of score PLATEAUS (candidate points).
"""
import csv, sys, glob, os
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import read_bug as rb

CROPS = "/private/tmp/claude-502/-Users-trevor-mccormick-Documents-trmccormick/62cba661-1dd3-4489-9233-dd6e600b423d/scratchpad/bug5"
Y0 = 624
FPS_SAMPLE = 5
FPS_VIDEO = 30
DIG_Y = (628, 690); DIG_X = (188, 296)
CHG_T = 0.020; PIX_T = 40
CROP_STEP = 3               # process every 3rd 5fps-crop (~1.7 fps) for speed
CONFIRM = 1
MIN_PLAT = 2


def to_canvas(crop):
    cv = np.zeros((704, 320, 3), np.uint8)
    h = min(crop.shape[0], 704 - Y0)
    cv[Y0:Y0 + h, 0:crop.shape[1]] = crop[:h]
    return cv


def dig_crop(canvas):
    c = cv2.cvtColor(canvas[DIG_Y[0]:DIG_Y[1], DIG_X[0]:DIG_X[1]], cv2.COLOR_BGR2GRAY)
    return cv2.resize(c, (108, 62))


def chg_frac(a, b):
    return (np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIX_T).mean()


def keyf(res):
    if not res or res.get("era") is None:
        return None
    return (res.get("setA", ""), res.get("setB", ""), res.get("gmA", ""),
            res.get("gmB", ""), res.get("ptsA", ""), res.get("ptsB", ""), res.get("server"))


def run():
    files = sorted(glob.glob(os.path.join(CROPS, "f*.jpg")))[::CROP_STEP]
    rows = []
    last_crop = None; last_res = None; n_ocr = 0
    for idx, fp in enumerate(files):
        frame_no = idx * CROP_STEP * (FPS_VIDEO // FPS_SAMPLE)
        crop = cv2.imread(fp)
        canv = to_canvas(crop)
        if rb._present(canv) < 0.08 or rb._points_box(canv) is None:
            rows.append((frame_no, 0, None)); continue
        dc = dig_crop(canv)
        if last_crop is not None and chg_frac(dc, last_crop) < CHG_T:
            res = last_res
        else:
            res = rb.read_frame(canv); last_crop = dc; last_res = res; n_ocr += 1
        rows.append((frame_no, 1, res))
    return rows, n_ocr


def collapse(rows):
    plats = []; cur = None; pending = None; pend_n = 0
    for (fno, pres, res) in rows:
        k = keyf(res) if pres else None
        if k is None:
            continue
        if cur and k == cur["key"]:
            cur["f1"] = fno; cur["n"] += 1; pending = None; pend_n = 0; continue
        if pending == k:
            pend_n += 1
        else:
            pending = k; pend_n = 1
        if pend_n >= CONFIRM:
            if cur and cur["n"] >= MIN_PLAT:
                plats.append(cur)
            cur = {"key": k, "f0": fno - (pend_n - 1) * CROP_STEP * 6, "f1": fno, "n": pend_n}
            pending = None; pend_n = 0
    if cur and cur["n"] >= MIN_PLAT:
        plats.append(cur)
    return plats


if __name__ == "__main__":
    rows, n_ocr = run()
    out = Path(__file__).parent
    with open(out / "sweep_raw.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "present", "key"])
        for (fno, pres, res) in rows:
            w.writerow([fno, pres, keyf(res) if pres else ""])
    plats = collapse(rows)
    with open(out / "sweep_plateaus.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "f0", "f1", "t0", "t1", "n", "setA", "setB", "gmA", "gmB", "ptsA", "ptsB", "server"])
        for j, p in enumerate(plats):
            w.writerow([j, p["f0"], p["f1"], f"{p['f0']/30:.1f}", f"{p['f1']/30:.1f}", p["n"], *p["key"]])
    print(f"crops={len(rows)} present={sum(r[1] for r in rows)} ocr_calls={n_ocr} plateaus={len(plats)}")
