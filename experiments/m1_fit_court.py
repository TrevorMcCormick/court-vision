"""M1 run 3: fit the court model and compute the homography.

Takes the Hough segments from run 2, clusters them into court lines, and
fits a homography from exactly four correspondences: the outer corners of
the doubles court (baselines x doubles sidelines). The net tape also shows
up as a horizontal line, but it can never be the topmost or bottommost
cluster, so using extremes sidesteps it entirely.

Validation is held out: the singles sidelines, service lines, and center
line took no part in the fit — reproject the full court model and see if
they land on the white pixels.

Court model (meters): doubles court 10.97 wide x 23.77 long, far baseline
at y=0. Singles sidelines inset 1.372; service lines 6.40 from the net
(net at y=11.885); center service line between the service lines.

Usage:
    uv run experiments/m1_fit_court.py outputs/m1/clean_plate.png outputs/m1/segments.npy
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m1"

# court model in meters, (x, y), far baseline y=0, left doubles sideline x=0
W_COURT = 10.97
L_COURT = 23.77
SINGLES_INSET = 1.372
NET_Y = L_COURT / 2
SVC_FAR_Y = NET_Y - 6.40
SVC_NEAR_Y = NET_Y + 6.40
CENTER_X = W_COURT / 2

MODEL_LINES = {
    "baseline_far": ((0, 0), (W_COURT, 0)),
    "baseline_near": ((0, L_COURT), (W_COURT, L_COURT)),
    "doubles_left": ((0, 0), (0, L_COURT)),
    "doubles_right": ((W_COURT, 0), (W_COURT, L_COURT)),
    "singles_left": ((SINGLES_INSET, 0), (SINGLES_INSET, L_COURT)),
    "singles_right": ((W_COURT - SINGLES_INSET, 0), (W_COURT - SINGLES_INSET, L_COURT)),
    "service_far": ((SINGLES_INSET, SVC_FAR_Y), (W_COURT - SINGLES_INSET, SVC_FAR_Y)),
    "service_near": ((SINGLES_INSET, SVC_NEAR_Y), (W_COURT - SINGLES_INSET, SVC_NEAR_Y)),
    "center_service": ((CENTER_X, SVC_FAR_Y), (CENTER_X, SVC_NEAR_Y)),
    "net": ((0, NET_Y), (W_COURT, NET_Y)),
}

MODEL_CORNERS = np.float32([
    (0, 0), (W_COURT, 0),            # far-left, far-right
    (0, L_COURT), (W_COURT, L_COURT) # near-left, near-right
])


def seg_angle(s):
    x1, y1, x2, y2 = s
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def fit_line(points):
    """Fit ax + by + c = 0 through Nx2 points, |a,b| = 1."""
    pts = np.asarray(points, dtype=np.float64)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean)
    d = vt[0]  # direction
    n = np.array([-d[1], d[0]])
    c = -n @ mean
    return n[0], n[1], c


def line_x_at(line, y):
    a, b, c = line
    return (-c - b * y) / a


def line_y_at(line, x):
    a, b, c = line
    return (-c - a * x) / b


def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    return np.array([(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d])


def cluster_1d(values, tol):
    """Greedy 1-D clustering; returns list of index lists."""
    order = np.argsort(values)
    clusters = [[order[0]]]
    for idx in order[1:]:
        if values[idx] - values[clusters[-1][-1]] <= tol:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plate")
    parser.add_argument("segments")
    args = parser.parse_args()

    img = cv2.imread(args.plate)
    h, w = img.shape[:2]
    segs = np.load(args.segments)

    angles = np.array([seg_angle(s) for s in segs])
    horiz = segs[(angles < 30) | (angles > 150)]
    vert = segs[(angles >= 30) & (angles <= 150)]
    print(f"{len(horiz)} horizontal segments, {len(vert)} vertical")

    # cluster horizontals by y at mid-column, verticals by x at mid-row
    hy = np.array([line_y_at(fit_line([s[:2], s[2:]]), w / 2) for s in horiz])
    vx = np.array([line_x_at(fit_line([s[:2], s[2:]]), h * 0.5) for s in vert])
    h_clusters = cluster_1d(hy, tol=15)
    v_clusters = cluster_1d(vx, tol=25)
    print(f"h clusters at y≈{[round(float(np.mean(hy[c]))) for c in h_clusters]}")
    print(f"v clusters at x≈{[round(float(np.mean(vx[c]))) for c in v_clusters]}")

    def merged_line(seg_group):
        pts = np.concatenate([[s[:2], s[2:]] for s in seg_group])
        return fit_line(pts)

    baseline_far = merged_line(horiz[h_clusters[0]])
    baseline_near = merged_line(horiz[h_clusters[-1]])
    doubles_left = merged_line(vert[v_clusters[0]])
    doubles_right = merged_line(vert[v_clusters[-1]])

    corners_img = np.float32([
        intersect(baseline_far, doubles_left),
        intersect(baseline_far, doubles_right),
        intersect(baseline_near, doubles_left),
        intersect(baseline_near, doubles_right),
    ])
    print("image corners (px):")
    for name, pt in zip(["far-L", "far-R", "near-L", "near-R"], corners_img):
        print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")

    H = cv2.getPerspectiveTransform(MODEL_CORNERS, corners_img)  # court(m) -> image(px)
    np.save(OUT_DIR / "H_court_to_img.npy", H)
    np.save(OUT_DIR / "H_img_to_court.npy", np.linalg.inv(H))

    # reproject the full model onto the frame — held-out lines are the test
    overlay = img.copy()
    for name, (p, q) in MODEL_LINES.items():
        pts = np.float32([p, q]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        color = (0, 255, 255) if "baseline" in name or "doubles" in name else (0, 128, 255)
        cv2.line(overlay, tuple(proj[0].astype(int)), tuple(proj[1].astype(int)), color, 2)
    cv2.imwrite(str(OUT_DIR / "model_reprojection.png"), overlay)

    # quantitative check: mean distance from held-out reprojected lines to
    # nearest white pixel (distance transform of the inverted white mask)
    white = cv2.imread(str(OUT_DIR / "white_mask.png"), cv2.IMREAD_GRAYSCALE)
    dist = cv2.distanceTransform(255 - white, cv2.DIST_L2, 5)
    held_out = ["singles_left", "singles_right", "service_far", "service_near", "center_service"]
    for name in held_out:
        p, q = MODEL_LINES[name]
        ts = np.linspace(0, 1, 50)
        pts = np.float32([(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])) for t in ts])
        proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
        d = dist[proj[inside, 1].astype(int), proj[inside, 0].astype(int)]
        print(f"held-out {name}: mean dist to white px = {d.mean():.1f} px (max {d.max():.1f})")

    print("-> outputs/m1/model_reprojection.png, H_court_to_img.npy, H_img_to_court.npy")


if __name__ == "__main__":
    main()
