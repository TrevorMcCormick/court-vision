"""WASB-SBDT ball tracking — the specialist that retired SAM from ball duty.

WASB (BMVC 2023, NTT, MIT license) is a per-frame heatmap ball detector
with pretrained TENNIS weights (F1 94-96 on the broadcast-tennis
benchmark). No prompt, no bootstrap, no per-clip spend: every clip is
trackable. Lifted from experiments/wasb_track_ball.py (frozen outputs).

Inference reproduces the authors' eval path exactly (their Detector
class hard-asserts CUDA, so the model + affine code are imported from
their src and the thin driver loop is re-implemented for MPS/CPU):
  input    3 consecutive RGB frames, each warped by their aspect-
           preserving affine to 512x288, ImageNet-normalized,
           concatenated to 9 channels
  model    HRNet per configs/model/wasb.yaml, checkpoint
           wasb_tennis_best.pth.tar['model_state_dict']
  output   3 heatmaps, sigmoid, threshold 0.5, connected components,
           heatmap-weighted centroid + score, inverse affine back to
           pixel coords (TracknetV2Postprocessor, concomp, hm-weighted)
  windows  non-overlapping triplets, step=3; a short tail reuses the
           last full triplet
  tracker  their OnlineTracker: drop candidates >300 px from the last
           visible position, keep the best score

Output schema: frame,cx_raw,cy_raw,w,h,x_stab,y_stab with a nominal
w=h=0.01 box (WASB emits points, not boxes) and the per-frame
stabilization shifts subtracted (from the players pass-A shifts CSVs).
No-detection frames are dropped, not faked.

Requires third_party/WASB-SBDT (their src) and the tennis checkpoint in
third_party/weights/ — see docs/USAGE.md.

Usage:
    uv run python -m courtvision track-ball t3 [clips...]
"""

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

INP_W, INP_H = 512, 288
FRAMES_IN = 3
SCORE_T = 0.5          # their postprocessor score_threshold
MAX_DISP = 300         # their online tracker gate, px at original res
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _wasb_imports():
    """Import HRNet + affine helpers from the vendored WASB src.
    utils/__init__.py pulls pandas (dataset CSV code we don't need);
    the affine helpers load straight from the file instead."""
    sys.path.insert(0, str(WASB_SRC))
    from models.hrnet import HRNet
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wasb_image", WASB_SRC / "utils" / "image.py")
    wasb_image = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wasb_image)
    return HRNet, wasb_image.get_affine_transform, wasb_image.affine_transform


def load_model(device):
    HRNet, _, _ = _wasb_imports()
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
    _, get_affine_transform, _ = _wasb_imports()
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
    _, _, affine_transform = _wasb_imports()
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


def track_match(cfg, stems=None, batch=8):
    """Track every clip (default: all in the match's clips dir) and write
    ball_<clip>.csv into the config's ball dir."""
    if stems is None:
        stems = sorted(p.stem for p in cfg.clips_dir.glob("*.mp4"))
    cfg.ball_dir.mkdir(parents=True, exist_ok=True)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    print(f"WASB tennis weights on {device}")
    for stem in stems:
        track, n = track_clip(model, device, cfg.clip_path(stem), batch)
        if not track:
            print(f"{stem}: 0/{n} frames detected — nothing written")
            continue
        shifts = load_shifts(cfg.out_dir, stem)
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
        with open(cfg.ball_dir / f"ball_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        fis = sorted(track)
        holes = [b - a for a, b in zip(fis, fis[1:])]
        max_hole = max(holes) if holes else 0
        print(f"{stem}: {len(rows)}/{n} frames ({100 * len(rows) / n:.0f}%), "
              f"max hole {max_hole}")
