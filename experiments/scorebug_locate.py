"""Auto-locate the score bug on ANY broadcast — kill staging knob #2's
by-eye crop. (blueprint: the last manual staging knob.)

Insight: the score bug is the one large graphic that PERSISTS across shot
cuts. Court view, replay, crowd, close-up — all change frame to frame; the
overlay stays put. So sample diverse frames, take per-pixel temporal
stability, and the bug is the biggest stable, text-bearing rectangle near a
frame edge. Zero hand-measured coordinates.

Digits change (0->15->30->40) so they flicker (moderate variance) INSIDE an
otherwise-stable box — a useful secondary signature, not required here.

    PYTHONPATH=. uv run python experiments/scorebug_locate.py <video.mp4> <tag>
Writes outputs/diag/scorebug_<tag>.png (std heat + boxes) and prints OCR.
"""

import sys

import cv2
import numpy as np
import pytesseract

from courtvision.config import ROOT


def sample_frames(video, n=48):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    return frames


def locate(frames):
    """Return ranked candidate bug boxes [(score, x0,y0,x1,y1), ...]."""
    H, W = frames[0].shape[:2]
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                     for f in frames])
    std = gray.std(axis=0)                       # per-pixel temporal std
    stable = (std < 14).astype(np.uint8) * 255   # persistent overlay pixels
    # a bug persists across SHOT CUTS, so frames must span diverse shots;
    # on a fixed-camera clip the static stands are stable too and drown it
    # out (see the RG failure) — the caller samples far-apart windows.
    # join the box, drop specks
    stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    stable = cv2.morphologyEx(stable, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # text-ness: edges of the median frame (a bug is full of glyph edges)
    med = np.median(gray, axis=0).astype(np.uint8)
    edges = cv2.Canny(med, 60, 160)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(stable)
    cands = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 800 or w < 60 or h < 12:
            continue
        ar = w / h
        if not (1.3 < ar < 14):                  # score bugs are wide bars
            continue
        edge_frac = edges[y:y+h, x:x+w].mean() / 255.0
        if edge_frac < 0.02:                     # empty box, no text
            continue
        # prefer lower third, wide, text-dense, sizeable
        low = 1.0 if y > H * 0.55 else (0.4 if y > H * 0.45 else 0.1)
        score = area * low * min(edge_frac, 0.25) * min(ar / 4, 1.5)
        cands.append((score, x, y, x + w, y + h, edge_frac))
    cands.sort(reverse=True)
    return std, cands


def merge_bug(cands, shape):
    """Fuse the stable text boxes of ONE corner into a single bug bbox, then
    expand right to swallow the flickering digit cells (which are excluded
    from the stable mask precisely because the score changes)."""
    H, W = shape
    _, bx0, by0, bx1, by1, _ = cands[0]
    xs0, ys0, xs1, ys1 = bx0, by0, bx1, by1
    for c in cands[1:]:
        _, x0, y0, x1, y1, _ = c
        if abs((y0 + y1) / 2 - (by0 + by1) / 2) < 70 and abs(x0 - bx0) < 220:
            xs0, ys0 = min(xs0, x0), min(ys0, y0)
            xs1, ys1 = max(xs1, x1), max(ys1, y1)
    w = xs1 - xs0
    xs1 = min(W, xs1 + int(w * 1.25))       # digits live right of the names
    ys0 = max(0, ys0 - 26)                   # header/serve tab above
    ys1 = min(H, ys1 + 6)
    return (xs0, ys0, xs1, ys1)


def read_box(frame, box):
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    txt = {}
    for name, inv in (("bright", False), ("dark", True)):
        flag = (cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
        _, t = cv2.threshold(g, 0, 255, flag)
        txt[name] = pytesseract.image_to_string(255 - t, config="--psm 6").strip()
    return txt


def main(video, tag):
    frames = sample_frames(video)
    std, cands = locate(frames)
    vis = frames[len(frames) // 2].copy()
    heat = cv2.applyColorMap((255 - np.clip(std, 0, 60) / 60 * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    vis = cv2.addWeighted(vis, 0.6, heat, 0.4, 0)
    for i, c in enumerate(cands[:4]):
        _, x0, y0, x1, y1, ef = c
        col = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2 if i == 0 else 1)
        print(f"  cand{i}: box=({x0},{y0},{x1},{y1}) edge={ef:.2f} score={c[0]:.0f}")
    if cands:
        bug = merge_bug(cands, frames[0].shape[:2])
        cv2.rectangle(vis, (bug[0], bug[1]), (bug[2], bug[3]), (0, 0, 255), 2)
    out = ROOT / "outputs" / "diag" / f"scorebug_{tag}.png"
    cv2.imwrite(str(out), vis)
    print(f"{tag}: {len(cands)} candidates -> {out}")
    if cands:
        print(f"  MERGED bug box (red) = {bug}")
        txt = read_box(frames[len(frames) // 2], bug)
        print(f"  OCR bright: {txt['bright']!r}")
        print(f"  OCR dark:   {txt['dark']!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "new")
