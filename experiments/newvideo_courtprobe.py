"""Zero-config court fit on brand-new downloaded video (scaling proof).

The staging blocker #1 (ingest.py) was the by-eye court knob: HSV hull band,
static fit-window frame, and 4 manual corners. The neural detector needs
none of it — it fits any court-view frame directly. And sampling frames by
keypoint count AUTO-FINDS a court-view frame, so the fit-window knob dies too.

Give it a video; it samples frames, runs the net on each, and fits the court
from the frame with the most keypoints. Overlay -> outputs/diag/newcourt_<tag>.png

    PYTHONPATH=. uv run python experiments/newvideo_courtprobe.py <video.mp4> <tag>
"""

import sys

import cv2
import numpy as np

from courtvision import courtfit_auto
from courtvision.config import ROOT
from courtvision.fitcourt import MODEL_LINES


def probe(video, tag, n_samples=24):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    model = courtfit_auto._load_model()
    ref = courtfit_auto._court_reference()
    best = None
    print(f"{tag}: {total} frames, sampling {n_samples}")
    for i in range(n_samples):
        fi = int(total * (i + 0.5) / n_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        pts = courtfit_auto.detect_keypoints(model, frame)
        n = sum(1 for p in pts if p[0] is not None)
        H_ref2img, xval = courtfit_auto.fit_homography(pts, ref)
        tag_ok = "court-view" if n >= 12 else ""
        print(f"  frame {fi:>7}: {n:2}/14 kps  xval={xval if xval else '—':>5}"
              f"  {tag_ok}")
        if H_ref2img is not None and (best is None or n > best[0]
                                      or (n == best[0] and xval < best[1])):
            H_img2m = courtfit_auto.S_REF2M @ np.linalg.inv(H_ref2img)
            best = (n, xval, np.linalg.inv(H_img2m), frame.copy(), fi)
    if best is None:
        print(f"{tag}: NO court-view frame found in {n_samples} samples")
        return None
    n, xval, H_c2i, frame, fi = best
    vis = frame.copy()
    for p, q in MODEL_LINES.values():
        proj = cv2.perspectiveTransform(
            np.float32([p, q]).reshape(-1, 1, 2), H_c2i).reshape(-1, 2)
        cv2.line(vis, tuple(proj[0].astype(int)), tuple(proj[1].astype(int)),
                 (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"{tag}: zero-config neural court, {n}/14 kps @f{fi}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out = ROOT / "outputs" / "diag" / f"newcourt_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"{tag}: BEST {n}/14 kps (xval {xval:.1f}px) @frame {fi} -> {out}")
    return {"tag": tag, "n_kps": n, "xval": float(xval), "frame": fi}


if __name__ == "__main__":
    probe(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "new")
