"""Neural court auto-fit + the paint referee (blueprint roadmap #2, wired).

Promotes experiments/court_autofit.py into the package. Runs the public
14-keypoint TennisCourtDetector on a match's fit plate, solves the court
homography, and lets fitcourt's line-mask scorer — the "paint referee" —
decide whether the neural court sits on the real painted white lines
better than the human's hand fit. If the neural fit is missing or worse,
ABSTAIN and keep the hand fit. The hand fit is snapshotted once
(H_court_to_img_hand.npy) and stays the trusted baseline the gate always
compares against, so re-running is idempotent.

The heavy pieces (torch + the third_party detector) import lazily: this
module is cheap to import and the chart/eval path never pays for them.
The detector clone + weights live in third_party/ (gitignored,
research-use-only license — see the blueprint risk register), so the
neural fit runs on this machine only; a fresh clone falls back to the
hand corners.
"""

import cv2
import numpy as np

from .config import ROOT
from .court import W_C, L_C, NET_Y

WEIGHTS = ROOT / "third_party" / "weights" / "court_det_model.pt"
DETECTOR = ROOT / "third_party" / "TennisCourtDetector"
NET_W, NET_H = 640, 360

# the detector's reference canvas (doubles corners at these ref pixels) -> meters
REF_X0, REF_X1 = 286.0, 1379.0
REF_Y0, REF_Y1 = 561.0, 2935.0
S_REF2M = np.array([[W_C / (REF_X1 - REF_X0), 0, -REF_X0 * W_C / (REF_X1 - REF_X0)],
                    [0, L_C / (REF_Y1 - REF_Y0), -REF_Y0 * L_C / (REF_Y1 - REF_Y0)],
                    [0, 0, 1.0]])

# 4 doubles corners + 2 net posts, in meters — for the hand-vs-neural delta
LANDMARKS_M = np.array([[0, 0], [W_C, 0], [0, L_C], [W_C, L_C],
                        [0, NET_Y], [W_C, NET_Y]], np.float32)


def _load_model():
    import sys
    import torch
    if str(DETECTOR) not in sys.path:
        sys.path.insert(0, str(DETECTOR))
    from tracknet import BallTrackerNet
    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


def _court_reference():
    import sys
    if str(DETECTOR) not in sys.path:
        sys.path.insert(0, str(DETECTOR))
    from court_reference import CourtReference
    return CourtReference()


def _postprocess(heatmap, low_thresh=170, max_radius=25, scale=2):
    """The one keypoint peak from a heatmap — the detector's postprocess,
    reimplemented with only cv2 (its module pulls in scipy for a refine
    path this pipeline never uses)."""
    _, hm = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(hm, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=2, minRadius=10,
                               maxRadius=max_radius)
    if circles is None:
        return None, None
    return circles[0][0][0] * scale, circles[0][0][1] * scale


def detect_keypoints(model, img):
    import torch
    inp = cv2.resize(img, (NET_W, NET_H)).astype(np.float32) / 255.0
    with torch.no_grad():
        out = model(torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).float())[0]
    pred = torch.sigmoid(out).numpy()
    # _postprocess already upscales by 2 (assumes 720p); scale only if the
    # plate isn't 1280x720
    sx, sy = img.shape[1] / (2 * NET_W), img.shape[0] / (2 * NET_H)
    pts = []
    for k in range(14):
        hm = (pred[k] * 255).astype(np.uint8)
        x, y = _postprocess(hm, low_thresh=170, max_radius=25)
        pts.append((x * sx, y * sy) if x is not None else (None, None))
    return pts


def fit_homography(pts, ref):
    """The detector author's config search, numpy-only: try each 4-point
    court configuration, keep the one whose fit best predicts the UNUSED
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


def neural_court_to_img(plate):
    """(H_court_to_img, n_keypoints, xval_px) for the fit plate, or
    (None, n, None) when the net can't place enough keypoints to fit."""
    model = _load_model()
    ref = _court_reference()
    pts = detect_keypoints(model, plate)
    n = sum(1 for p in pts if p[0] is not None)
    H_ref2img, xval = fit_homography(pts, ref)
    if H_ref2img is None:
        return None, n, None
    H_img2m = S_REF2M @ np.linalg.inv(H_ref2img)
    H_m2img = np.linalg.inv(H_img2m)                 # court(meters) -> image
    return H_m2img.astype(np.float64), n, (float(xval) if xval else None)


def paint_score(H_c2i, line_mask):
    """Mean pixels from the projected model lines to the nearest painted
    white pixel — fitcourt's line-mask scorer, standalone. Lower is better;
    inf if the court projects mostly off-frame. H_c2i is court->image."""
    from .fitcourt import MODEL_LINES
    h, w = line_mask.shape[:2]
    dist = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 5)
    errs = []
    for p, q in MODEL_LINES.values():
        ts = np.linspace(0.05, 0.95, 40)
        pts = np.float32([(p[0] + t * (q[0] - p[0]),
                           p[1] + t * (q[1] - p[1])) for t in ts])
        proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2),
                                        H_c2i.astype(np.float64)).reshape(-1, 2)
        ok = ((proj[:, 0] >= 0) & (proj[:, 0] < w)
              & (proj[:, 1] >= 0) & (proj[:, 1] < h))
        if ok.sum() < 10:
            return float("inf")
        d = dist[proj[ok, 1].astype(int), proj[ok, 0].astype(int)]
        errs.append(float(np.clip(d, 0, 25).mean()))
    return float(np.mean(errs))


def _landmark_delta(a_c2i, b_c2i):
    lm = LANDMARKS_M.reshape(-1, 1, 2)
    pa = cv2.perspectiveTransform(lm, a_c2i.astype(np.float64)).reshape(-1, 2)
    pb = cv2.perspectiveTransform(lm, b_c2i.astype(np.float64)).reshape(-1, 2)
    return float(np.hypot(*(pa - pb).T).mean())


def _overlay(plate, hand_c2i, neural_c2i, chosen, out_dir, mid):
    from .fitcourt import MODEL_LINES
    vis = plate.copy()

    def draw(H, color, thick):
        for p, q in MODEL_LINES.values():
            proj = cv2.perspectiveTransform(
                np.float32([p, q]).reshape(-1, 1, 2),
                H.astype(np.float64)).reshape(-1, 2)
            cv2.line(vis, tuple(proj[0].astype(int)),
                     tuple(proj[1].astype(int)), color, thick, cv2.LINE_AA)

    # hand green, neural orange; the CHOSEN one drawn thick on top
    draw(hand_c2i, (120, 255, 120), 3 if chosen == "hand" else 1)
    if neural_c2i is not None:
        draw(neural_c2i, (0, 165, 255), 3 if chosen == "neural" else 1)
    cv2.putText(vis, f"{mid}: chose {chosen} (green=hand orange=neural)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path = out_dir / f"autofit_gate_{mid}.png"
    cv2.imwrite(str(path), vis)
    return path


def gated_fit(cfg, sane_px=6.0):
    """Run the neural fit, referee it against the hand fit on the painted
    lines, and write the winner to H_*.npy. Abstain (keep hand) when the
    net can't fit or its lines miss the paint. Returns the verdict dict."""
    out = cfg.out_dir
    plate = cv2.imread(str(out / "plate_fit.png"))
    line_mask = cv2.imread(str(out / "white_mask.png"), cv2.IMREAD_GRAYSCALE)
    if plate is None or line_mask is None:
        raise FileNotFoundError(
            f"{cfg.id}: need plate_fit.png + white_mask.png in {out} "
            "(run `fitcourt` once to stage them)")

    # preserve the hand fit once; it stays the trusted baseline forever
    hand_path = out / "H_court_to_img_hand.npy"
    if not hand_path.exists():
        np.save(hand_path, np.load(out / "H_court_to_img.npy"))
    hand_c2i = np.load(hand_path)
    hand_score = paint_score(hand_c2i, line_mask)

    neural_c2i, n_kps, xval = neural_court_to_img(plate)
    verdict = {"match": cfg.id, "n_kps": n_kps, "xval_px": xval,
               "hand_score": round(hand_score, 2), "neural_score": None,
               "landmark_delta_px": None}

    if neural_c2i is None:
        verdict.update(decision="abstain: no neural fit", chosen="hand")
        chosen = hand_c2i
    else:
        neural_score = paint_score(neural_c2i, line_mask)
        verdict["neural_score"] = round(neural_score, 2)
        verdict["landmark_delta_px"] = round(_landmark_delta(neural_c2i, hand_c2i), 1)
        if neural_score < sane_px and neural_score <= hand_score:
            verdict.update(decision="use neural", chosen="neural")
            chosen = neural_c2i
        else:
            why = ("neural above sanity bar" if neural_score >= sane_px
                   else "hand hugs paint better")
            verdict.update(decision=f"keep hand ({why})", chosen="hand")
            chosen = hand_c2i

    np.save(out / "H_court_to_img.npy", chosen)
    np.save(out / "H_img_to_court.npy", np.linalg.inv(chosen))
    verdict["overlay"] = str(_overlay(plate, hand_c2i, neural_c2i,
                                      verdict["chosen"], out, cfg.id))
    return verdict
