"""Automatic score-bug reader for the RG-2024 WTA bug (g1 = Swiatek-Paolini).

The bug sits bottom-left, two rows (player-1 SWIATEK top, PAOLINI bottom).
Columns, left to right:  [serve "//"] NAME [completed-set cols] GAMES POINTS.
  - POINTS is dark text on a solid LIGHT (cream/grey) cell  -> rightmost anchor
  - GAMES is white text on a lighter-green highlighted cell
  - completed-set columns appear left of GAMES once a set is won (set 2+)
  - the "//" serve marker sits at the far left of the SERVING player's row

Read strategy: anchor on the LIGHT points box (found dynamically by column
fill), then derive the games / completed-set cells by fixed offsets left of
it. Per row, OCR each cell with a whitelisted tesseract. Points cell inverted
(dark-on-light); games/set cells thresholded bright (white-on-green).
Serve marker read by white density per row. Presence gated on name-region
white density.

This is the "MANUAL 3/3" knob (by-eye score-bug transcription) attacked
by machine. Additive experiment; touches no shipped code.
"""
import cv2, numpy as np, pytesseract, re

BUG_Y = (628, 690)        # full bug band
ROW_TOP = (631, 658)      # SWIATEK row y
ROW_BOT = (658, 687)      # PAOLINI row y
NAME_X  = (86, 190)       # presence region
SERVE_X = (58, 92)        # "//" marker region
PTS_W = 40                # points-box width (px)
GM_DX = (-30, 2)          # games cell offset from points-box left edge
SET_DX = (-58, -28)       # completed-set cell offset


def _thr(cell, invert):
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    _, t = cv2.threshold(g, 0, 255, flag)
    t = cv2.copyMakeBorder(t, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
    return 255 - t                    # black text on white for tesseract


def _ocr(cell, whitelist, invert, psm=8):
    if cell.size == 0:
        return ""
    t = _thr(cell, invert)
    # reject near-empty cells (blank points box)
    if (t < 128).mean() < 0.008:
        return ""
    cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(t, config=cfg).strip()


def _present(frame):
    reg = frame[BUG_Y[0]:BUG_Y[1], NAME_X[0]:NAME_X[1]]
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    return (g > 170).mean()


def _points_box(frame):
    """Rightmost solid-light vertical block in x in [206,298]. Returns (L,R) or None."""
    band = frame[BUG_Y[0]:BUG_Y[1], 206:300]
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    frac = ((g > 135) & (hsv[:, :, 1] < 85)).mean(axis=0)
    cols = np.where(frac > 0.5)[0]
    if len(cols) < 6:
        return None
    # take the widest contiguous run
    runs, s = [], cols[0]
    for i in range(1, len(cols)):
        if cols[i] != cols[i-1] + 1:
            runs.append((s, cols[i-1])); s = cols[i]
    runs.append((s, cols[-1]))
    L, R = max(runs, key=lambda r: r[1]-r[0])
    return (206 + L, 206 + R)


def _clean_pts(s):
    s = s.upper().replace(" ", "").replace("O", "0")
    if "A" in s or "D" in s:
        return "AD"
    m = re.sub(r"[^0-9]", "", s)
    if m in ("0", "15", "30", "40"): return m
    fix = {"1": "15", "5": "15", "3": "30", "4": "40", "45": "40", "48": "40",
           "150": "15", "300": "30", "400": "40", "10": "40", "00": "0"}
    return fix.get(m, m)


def _clean_dig(s):
    m = re.sub(r"[^0-9]", "", s.upper().replace("O", "0"))
    return m[:1] if m else ""


def read_frame(frame):
    """Return dict or None. A=Swiatek(top), B=Paolini(bottom)."""
    if _present(frame) < 0.08:
        return None
    box = _points_box(frame)
    if box is None:
        return {"present": True, "era": None, "note": "no-points-box"}
    pL, pR = box
    era = 1 if pL < 242 else 2
    px = (pL - 3, pL + PTS_W)
    gx = (pL + GM_DX[0], pL + GM_DX[1])
    sx = (pL + SET_DX[0], pL + SET_DX[1])
    out = {"present": True, "era": era, "pL": pL}
    for tag, (y0, y1) in (("A", ROW_TOP), ("B", ROW_BOT)):
        row = frame[y0:y1]
        out[f"gm{tag}"] = _clean_dig(_ocr(row[:, gx[0]:gx[1]], "01234567", False, 10))
        praw = _ocr(row[:, px[0]:px[1]], "0123456ADO", True, 8) or \
               _ocr(row[:, px[0]:px[1]], "0123456ADO", True, 7)
        out[f"pts{tag}"] = _clean_pts(praw)
        out[f"set{tag}"] = _clean_dig(_ocr(row[:, sx[0]:sx[1]], "01234567", False, 10)) if era == 2 else ""
    dtop = (cv2.cvtColor(frame[ROW_TOP[0]:ROW_TOP[1], SERVE_X[0]:SERVE_X[1]], cv2.COLOR_BGR2GRAY) > 170).mean()
    dbot = (cv2.cvtColor(frame[ROW_BOT[0]:ROW_BOT[1], SERVE_X[0]:SERVE_X[1]], cv2.COLOR_BGR2GRAY) > 170).mean()
    out["server"] = 1 if (dtop > dbot and dtop > 0.06) else (2 if dbot > 0.06 else None)
    return out


if __name__ == "__main__":
    truth = {
        "t1200": "set1 gm1-0 ptsAD-blank svr2",
        "t1800": "set1 gm2-2 pts blank(0-0) svr1",
        "t2400": "set1 gm4-2 pts15-0 svr1",
        "t3600": "set2 set6-2 gm3-0 pts0-15 svr2",
        "t4000": "set2 set6-2 gm4-0 pts40-30 svr1",
    }
    base = "/private/tmp/claude-502/-Users-trevor-mccormick-Documents-trmccormick/62cba661-1dd3-4489-9233-dd6e600b423d/scratchpad/g1f"
    for k, v in truth.items():
        r = read_frame(cv2.imread(f"{base}/{k}.png"))
        print(f"{k}: TRUTH {v}\n      READ  {r}")
