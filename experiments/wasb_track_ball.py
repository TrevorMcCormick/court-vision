"""WASB-SBDT ball tracking — the specialist challenger to SAM-3.

WASB (BMVC 2023, NTT, MIT license) is a per-frame heatmap ball detector
with pretrained TENNIS weights (F1 94-96 on the broadcast-tennis
benchmark). No prompt, no bootstrap, no per-clip fal spend: every clip
is trackable, including the 13 t1 clips SAM never touched because no
toss/mover bootstrap was trusted.

Inference reproduces the authors' eval path exactly (their Detector
class hard-asserts CUDA, so the model + affine code are imported from
their src and the thin driver loop is re-implemented for MPS/CPU):
  input    3 consecutive RGB frames, each warped by their aspect-
           preserving affine to 512x288 (for our 1280x720 clips this is
           a pure 0.4x resize), ImageNet-normalized, concatenated to 9
           channels (dataset_loader.py get_transform + build_img_transforms)
  model    HRNet per configs/model/wasb.yaml, checkpoint
           wasb_tennis_best.pth.tar['model_state_dict']
  output   3 heatmaps (one per input frame), sigmoid, threshold 0.5,
           connected components, heatmap-weighted centroid + score,
           inverse affine back to pixel coords (TracknetV2Postprocessor,
           blob_det_method=concomp, use_hm_weight=True)
  windows  non-overlapping triplets, step=3 (their detector step);
           a short tail reuses the last full triplet
  tracker  their OnlineTracker: drop candidates >300 px from the last
           visible position, keep the best score (trackers/online.py)

Output schema matches ours exactly: frame,cx_raw,cy_raw,w,h,x_stab,y_stab
with a nominal w=h=0.01 box (WASB emits points, not boxes) and the
per-frame stabilization shifts subtracted, same as t1_track_ball.py.
No-detection frames are dropped, not faked.

Usage:
    uv run experiments/wasb_track_ball.py --tree t1 t1_point_01 ...
    uv run experiments/wasb_track_ball.py --tree t1 --sanity t1_point_01
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
WASB_SRC = ROOT / "third_party" / "WASB-SBDT" / "src"
WEIGHTS = ROOT / "third_party" / "weights" / "wasb_tennis_best.pth.tar"
sys.path.insert(0, str(WASB_SRC))

from models.hrnet import HRNet                          # noqa: E402

# utils/__init__.py pulls pandas (dataset CSV code we don't need); load
# the affine helpers straight from the file instead.
import importlib.util                                   # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "wasb_image", WASB_SRC / "utils" / "image.py")
_wasb_image = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wasb_image)
get_affine_transform = _wasb_image.get_affine_transform
affine_transform = _wasb_image.affine_transform

INP_W, INP_H = 512, 288
FRAMES_IN = 3
SCORE_T = 0.5          # their postprocessor score_threshold
MAX_DISP = 300         # their online tracker gate, px at original res
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(device):
    # HRNet reads the config with attribute access (cfg.MODEL.EXTRA), the
    # omegaconf style their hydra harness provides
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(
        yaml.safe_load(open(WASB_SRC / "configs" / "model" / "wasb.yaml")))
    model = HRNet(cfg)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def frame_transforms(w, h):
    """Their dataset_loader.get_transform: aspect-preserving affine into
    the model input, and its inverse for mapping peaks back."""
    c = np.array([w / 2.0, h / 2.0], np.float32)
    s = max(h, w) * 1.0
    fwd = get_affine_transform(c, s, 0, [INP_W, INP_H])
    inv = get_affine_transform(c, s, 0, [INP_W, INP_H], inv=1)
    return fwd, inv


def preprocess(frame_bgr, fwd):
    img = cv2.warpAffine(frame_bgr, fwd, (INP_W, INP_H), flags=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)          # CHW


def detect_blobs(hm, inv):
    """TracknetV2Postprocessor._detect_blob_concomp, use_hm_weight=True."""
    dets = []
    if hm.max() <= SCORE_T:
        return dets
    _, hm_th = cv2.threshold(hm, SCORE_T, 1, cv2.THRESH_BINARY)
    n, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
    for m in range(1, n):
        ys, xs = np.where(labels == m)
        ws = hm[ys, xs]
        score = float(ws.sum())
        x = float((xs * ws).sum() / ws.sum())
        y = float((ys * ws).sum() / ws.sum())
        xy = affine_transform(np.array([x, y]), inv)
        dets.append({"xy": xy, "score": score})
    return dets


def track_clip(model, device, clip_path, batch=8):
    """Run WASB over every frame; returns {frame_idx: (x_px, y_px, score)}."""
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n < FRAMES_IN:
        return {}, n
    h, w = frames[0].shape[:2]
    fwd, inv = frame_transforms(w, h)

    # non-overlapping triplets; tail reuses the last full triplet
    starts = list(range(0, n - FRAMES_IN + 1, FRAMES_IN))
    if starts[-1] + FRAMES_IN < n:
        starts.append(n - FRAMES_IN)

    dets_per_frame = {}
    with torch.no_grad():
        for b0 in range(0, len(starts), batch):
            chunk = starts[b0:b0 + batch]
            inp = np.stack([
                np.concatenate([preprocess(frames[s + k], fwd)
                                for k in range(FRAMES_IN)], axis=0)
                for s in chunk])
            out = model(torch.from_numpy(inp).to(device))[0]
            hms = torch.sigmoid(out).cpu().numpy()
            for i, s in enumerate(chunk):
                for k in range(FRAMES_IN):
                    fi = s + k
                    if fi in dets_per_frame:      # tail overlap
                        continue
                    dets_per_frame[fi] = detect_blobs(hms[i, k], inv)

    # their OnlineTracker: gate by last visible position, keep best score
    track = {}
    last_xy = None
    for fi in range(n):
        dets = dets_per_frame.get(fi, [])
        if last_xy is not None:
            dets = [d for d in dets
                    if np.linalg.norm(d["xy"] - last_xy) < MAX_DISP]
        if dets:
            best = max(dets, key=lambda d: d["score"])
            track[fi] = (float(best["xy"][0]), float(best["xy"][1]),
                         best["score"])
            last_xy = best["xy"]
        else:
            last_xy = None
    return track, n


def load_shifts(out_base, stem):
    shifts = {}
    with open(out_base / "plates" / f"shifts_{stem}.csv") as f:
        for row in csv.DictReader(f):
            shifts[int(row["frame"])] = (float(row["dx"]), float(row["dy"]))
    return shifts


def render_sanity(clip_path, track, out_dir, stem):
    fis = sorted(track.keys())
    picks = [fis[len(fis) // 5], fis[len(fis) // 2], fis[4 * len(fis) // 5]]
    cap = cv2.VideoCapture(str(clip_path))
    for j, fi in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        x, y, sc = track[fi]
        cv2.circle(fr, (int(round(x)), int(round(y))), 14, (0, 255, 255), 2)
        cv2.putText(fr, f"{stem} f{fi} score={sc:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        crop = fr[max(0, int(y) - 160):min(720, int(y) + 160),
                  max(0, int(x) - 240):min(1280, int(x) + 240)]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0,
                          interpolation=cv2.INTER_CUBIC)
        p = out_dir / f"sanity_{stem}_{j}.png"
        cv2.imwrite(str(p), crop)
        print(f"  sanity frame -> {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--tree", required=True, choices=["t1", "t2"])
    parser.add_argument("--sanity", action="store_true",
                        help="render 3 circled frames for eyeball sign-off")
    args = parser.parse_args()

    out_base = ROOT / "outputs" / args.tree
    ball_dir = out_base / "ball_wasb"
    ball_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = ROOT / "clips" / f"points_{args.tree}"

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    print(f"WASB tennis weights on {device}")

    for stem in args.clips:
        clip_path = clip_dir / f"{stem}.mp4"
        track, n = track_clip(model, device, clip_path)
        if not track:
            print(f"{stem}: 0/{n} frames detected — nothing written")
            continue
        shifts = load_shifts(out_base, stem)
        rows = []
        for fi in sorted(track):
            x, y, _ = track[fi]
            dx, dy = shifts.get(fi, (0, 0))
            rows.append({"frame": fi,
                         "cx_raw": round(x / 1280, 6),
                         "cy_raw": round(y / 720, 6),
                         "w": 0.01, "h": 0.01,
                         "x_stab": round(x - dx, 2),
                         "y_stab": round(y - dy, 2)})
        with open(ball_dir / f"ball_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        fis = sorted(track)
        holes = [b - a for a, b in zip(fis, fis[1:])]
        max_hole = max(holes) if holes else 0
        print(f"{stem}: {len(rows)}/{n} frames ({100 * len(rows) / n:.0f}%), "
              f"max hole {max_hole}")
        if args.sanity:
            render_sanity(clip_path, track, ball_dir, stem)


if __name__ == "__main__":
    main()
