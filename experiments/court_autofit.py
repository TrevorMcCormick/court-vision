"""Court auto-fit v1 — the court that fits itself (blueprint roadmap #2).

Run the public 14-keypoint TennisCourtDetector (third_party/, pretrained
on 8,841 broadcast images across hard/clay/grass; no license — research
use only, see blueprint risk register) on each match's fit-window plate
(the stabilized median frame the hand homography was built on), solve
the homography from its keypoints via the author's config search
(reimplemented numpy-only), convert into this repo's image->meters
convention, and grade against the 8 HAND-FIT homographies as ground
truth.

The money test: g1's clay plate, where the classical auto-fit failed at
74 px rms and a human had to click corners.

Deltas are measured where it matters: project 6 court landmarks (4
doubles corners + 2 net posts) through both fits and compare in pixels.
Overlay receipts per match: outputs/diag/autofit_<id>.png (net fit
orange, hand fit green).

Run:  PYTHONPATH=.:experiments uv run --with scipy python experiments/court_autofit.py
"""

import sys

import cv2
import numpy as np
import torch

from courtvision import config
from courtvision.config import ROOT
from courtvision.court import W_C, L_C, NET_Y

sys.path.insert(0, str(ROOT / "third_party" / "TennisCourtDetector"))
from tracknet import BallTrackerNet            # noqa: E402
from postprocess import postprocess            # noqa: E402
from court_reference import CourtReference     # noqa: E402

WEIGHTS = ROOT / "third_party" / "weights" / "court_det_model.pt"
MATCHES = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "g1"]
NET_W, NET_H = 640, 360

# their reference canvas -> meters (doubles corners at these ref pixels)
REF_X0, REF_X1 = 286.0, 1379.0
REF_Y0, REF_Y1 = 561.0, 2935.0
S_REF2M = np.array([[W_C / (REF_X1 - REF_X0), 0, -REF_X0 * W_C / (REF_X1 - REF_X0)],
                    [0, L_C / (REF_Y1 - REF_Y0), -REF_Y0 * L_C / (REF_Y1 - REF_Y0)],
                    [0, 0, 1.0]])

LANDMARKS_M = np.array([
    [0, 0], [W_C, 0], [0, L_C], [W_C, L_C],      # doubles corners
    [0, NET_Y], [W_C, NET_Y],                    # net posts
], np.float32)


def detect_keypoints(model, img):
    inp = cv2.resize(img, (NET_W, NET_H)).astype(np.float32) / 255.0
    with torch.no_grad():
        out = model(torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).float())[0]
    pred = torch.sigmoid(out).numpy()
    # postprocess() already upscales heatmap coords by 2 (assumes 720p);
    # scale only if the plate isn't 1280x720
    sx, sy = img.shape[1] / (2 * NET_W), img.shape[0] / (2 * NET_H)
    pts = []
    for k in range(14):
        hm = (pred[k] * 255).astype(np.uint8)
        x, y = postprocess(hm, low_thresh=170, max_radius=25)
        pts.append((x * sx, y * sy) if x is not None else (None, None))
    return pts


def fit_homography(pts, ref):
    """The author's config search, numpy-only: try 12 four-point court
    configurations, keep the one whose fit best predicts the UNUSED
    detected keypoints (cross-validation, not self-consistency)."""
    best, best_err = None, np.inf
    refer = np.array(ref.key_points, np.float32).reshape(-1, 1, 2)
    conf_inds = {i: [ref.key_points.index(p) for p in ref.court_conf[i]]
                 for i in ref.court_conf}
    for ci, inds in conf_inds.items():
        quad = [pts[i] for i in inds]
        if any(p[0] is None for p in quad):
            continue
        src = np.float32(ref.court_conf[ci])
        H, _ = cv2.findHomography(src, np.float32(quad), method=0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(refer, H).reshape(-1, 2)
        errs = [np.hypot(*(np.array(pts[i]) - proj[i]))
                for i in range(12) if i not in inds and pts[i][0] is not None]
        if errs and np.mean(errs) < best_err:
            best_err, best = np.mean(errs), H
    return best, best_err


def draw_court(img, H_c2i, color):
    lines = [((0, 0), (W_C, 0)), ((0, L_C), (W_C, L_C)),
             ((0, 0), (0, L_C)), ((W_C, W_C and 0), (W_C, L_C)),
             ((0, NET_Y), (W_C, NET_Y))]
    for a, b in lines:
        p = cv2.perspectiveTransform(
            np.float32([a, b]).reshape(-1, 1, 2), H_c2i).reshape(-1, 2)
        cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)),
                 color, 2, cv2.LINE_AA)


def run():
    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    ref = CourtReference()
    out = ["# Court auto-fit v1 — pretrained 14-keypoint net vs the hand fits",
           "(deltas in px at 6 landmarks: 4 doubles corners + 2 net posts;",
           " hand-fit homographies are ground truth; g1 is the clay plate",
           " where classical auto-fit failed at 74 px rms)\n"]
    diag = ROOT / "outputs" / "diag"
    diag.mkdir(parents=True, exist_ok=True)

    for mid in MATCHES:
        cfg = config.load(mid)
        plate = cv2.imread(str(cfg.out_dir / "plate_fit.png"))
        if plate is None:
            out.append(f"{mid}: NO PLATE")
            continue
        pts = detect_keypoints(model, plate)
        n_det = sum(1 for p in pts if p[0] is not None)
        H_ref2img, xval = fit_homography(pts, ref)
        if H_ref2img is None:
            out.append(f"{mid}: kps {n_det}/14 — NO FIT (abstain)")
            continue
        # image -> meters, this repo's convention
        H_img2m = S_REF2M @ np.linalg.inv(H_ref2img)
        H_m2img = np.linalg.inv(H_img2m)
        H_hand_c2i = np.load(cfg.out_dir / "H_court_to_img.npy")

        lm = LANDMARKS_M.reshape(-1, 1, 2)
        p_net = cv2.perspectiveTransform(lm, H_m2img.astype(np.float64)).reshape(-1, 2)
        p_hand = cv2.perspectiveTransform(lm, H_hand_c2i).reshape(-1, 2)
        d = np.hypot(*(p_net - p_hand).T)
        out.append(f"{mid}: kps {n_det}/14, config-xval {xval:.1f}px | "
                   f"landmark delta mean {d.mean():.1f}px, max {d.max():.1f}px"
                   f"{'   <-- the clay plate' if mid == 'g1' else ''}")

        vis = plate.copy()
        draw_court(vis, H_hand_c2i, (120, 255, 120))
        draw_court(vis, H_m2img, (0, 165, 255))
        cv2.imwrite(str(diag / f"autofit_{mid}.png"), vis)

    report = "\n".join(out)
    print(report)
    (diag / "court_autofit_report.txt").write_text(report + "\n")
    print(f"\n[saved] {diag}/court_autofit_report.txt + autofit_<id>.png overlays")


if __name__ == "__main__":
    run()
