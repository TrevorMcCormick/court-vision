"""M3 experiment 6 — ball tracking over point clips: the toss IS the bootstrap.

M0's open problem: SAM needs a box drawn around THE game ball in one
frame per clip, and nobody is clicking 60 boxes. The serve anatomy pays
it off: in the frames before serve contact the toss ball hangs ABOVE the
server's head — a small, isolated, fast diff blob in a known place. Find
it in the bg-sub mask, box it, done. No serve call? Fallback: the fastest
small mover over the court anywhere in the clip.

Bootstraps are in STABILIZED coords; SAM sees the RAW clip, so boxes are
un-shifted before prompting (and the returned track is re-shifted after).
Box-prompt responses are [cx, cy, w, h] normalized (M0's discovery);
lost-track frames are the usual giant stuck boxes, dropped at w > 0.05.

ALWAYS dry-run first: renders every prompt box to
outputs/m3/ball/prompt_<clip>.png for eyeball sign-off, spends nothing.

Usage:
    uv run experiments/m3_track_ball.py point_16 point_25 ... --dry-run
    uv run experiments/m3_track_ball.py point_16 point_25 ... --send
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

OUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "m3"
BALL_DIR = OUT_BASE / "ball"
ROOT = Path(__file__).resolve().parent.parent

DIFF_T = 30
BALL_AREA = (6, 220)
W_C, L_C = 10.97, 23.77


def load_shifts(stem):
    shifts = {}
    with open(OUT_BASE / "plates" / f"shifts_{stem}.csv") as f:
        for row in csv.DictReader(f):
            shifts[int(row["frame"])] = (float(row["dx"]), float(row["dy"]))
    return shifts


def load_players(stem):
    per = {}
    with open(OUT_BASE / "players" / f"players_{stem}.csv") as f:
        for row in csv.DictReader(f):
            per.setdefault(int(row["frame"]), {})[row["player"]] = row
    return per


def read_frames(path, lo, hi):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    out = {}
    for fi in range(lo, hi):
        ok, fr = cap.read()
        if not ok:
            break
        out[fi] = fr
    return out


def stabilized_diff(frame, shift, plate):
    dx, dy = shift
    if dx or dy:
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    d = cv2.absdiff(frame, plate).max(axis=2)
    m = (d > DIFF_T).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def small_components(mask, area_rng):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for lab in range(1, n):
        x, y, w, h, area = stats[lab]
        if area_rng[0] <= area <= area_rng[1] and w < 40 and h < 40:
            out.append((x, y, w, h, area))
    return out


def toss_bootstrap(stem, serve, players, shifts, plate, clip_path):
    """Small blob above the server's head shortly before contact."""
    sf = int(serve["serve_frame"])
    side = serve["server"]
    frames = read_frames(clip_path, max(0, sf - 18), sf + 1)
    for fi in range(sf - 2, max(0, sf - 18) - 1, -1):
        if fi not in frames or fi not in players or side not in players[fi]:
            continue
        p = players[fi][side]
        cx = float(p["cx"]) * 1280
        head_y = (float(p["cy"]) - float(p["h"]) / 2) * 720
        m = stabilized_diff(frames[fi], shifts.get(fi, (0, 0)), plate)
        x1 = int(max(0, cx - 90)); x2 = int(min(1280, cx + 90))
        y1 = int(max(0, head_y - 130)); y2 = int(max(1, head_y + 6))
        sub = np.zeros_like(m)
        sub[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        cands = small_components(sub, BALL_AREA)
        if 1 <= len(cands) <= 3:
            # nearest to straight above the head center wins
            x, y, w, h, _ = min(cands, key=lambda c: abs(c[0] + c[2] / 2 - cx))
            return fi, (x, y, w, h), "toss"
    return None


def mover_bootstrap(stem, players, shifts, plate, clip_path, n_frames, region):
    """Fallback: small mover with a BALLISTIC 3-point chain.

    A single 2-frame hop matched broadcast overlays — the score bug's flip
    animation and a vibrating net sign are 'fast small movers' too. Real
    ball flight gives three consecutive detections with consistent
    direction and step; overlay flicker doesn't. Overlay rectangles are
    masked out as cheap insurance regardless.
    """
    step = 5
    frames = read_frames(clip_path, 0, n_frames)
    masks = {}
    overlay = np.ones(plate.shape[:2], np.uint8)
    overlay[585:695, 70:430] = 0     # score bug
    overlay[630:715, 420:880] = 0    # MONTREAL 375 watermark
    overlay[245:300, 300:1000] = 0   # net band signs (Emirates etc.)

    def mask_at(fi):
        if fi not in masks and fi in frames:
            m = stabilized_diff(frames[fi], shifts.get(fi, (0, 0)), plate) & region & overlay
            for side in ("near", "far"):
                p = players.get(fi, {}).get(side)
                if p:
                    px1 = int((float(p["cx"]) - float(p["w"]) / 2) * 1280) - 12
                    py1 = int((float(p["cy"]) - float(p["h"]) / 2) * 720) - 12
                    px2 = int((float(p["cx"]) + float(p["w"]) / 2) * 1280) + 12
                    py2 = int((float(p["cy"]) + float(p["h"]) / 2) * 720) + 12
                    m[max(0, py1):py2, max(0, px1):px2] = 0
            masks[fi] = m
        return masks.get(fi)

    def centers(fi):
        m = mask_at(fi)
        if m is None:
            return []
        return [(x + w / 2, y + h / 2, (x, y, w, h))
                for (x, y, w, h, _) in small_components(m, BALL_AREA)]

    best = None
    for fi in range(4, n_frames - 8, step):
        for c0 in centers(fi):
            for c1 in centers(fi + 2):
                v1 = (c1[0] - c0[0], c1[1] - c0[1])
                d1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
                if not (7 <= d1 <= 90):
                    continue
                for c2 in centers(fi + 4):
                    v2 = (c2[0] - c1[0], c2[1] - c1[1])
                    d2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
                    if not (7 <= d2 <= 90) or not (0.5 <= d2 / d1 <= 2.0):
                        continue
                    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)
                    if cos < 0.8:      # ~<37 degrees of turn
                        continue
                    if best is None or d1 + d2 > best[0]:
                        best = (d1 + d2, fi, c0[2])
    if best:
        return best[1], best[2], "mover"
    return None


def court_region(shape):
    Hc2i = np.load(ROOT / "outputs/m1/H_court_to_img.npy")
    poly = np.float32([[-2, -4], [W_C + 2, -4],
                       [W_C + 3, L_C + 3], [-3, L_C + 3]]).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(poly, Hc2i).reshape(-1, 2).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [img], 1)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.send and not args.dry_run:
        args.dry_run = True

    serves = {r["clip"]: r for r in csv.DictReader(open(OUT_BASE / "serves.csv"))}
    BALL_DIR.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    prompts = {}
    for stem in args.clips:
        clip_path = ROOT / "clips/points" / f"{stem}.mp4"
        cap = cv2.VideoCapture(str(clip_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames += n
        shifts = load_shifts(stem)
        players = load_players(stem)
        plate = cv2.imread(str(OUT_BASE / "plates" / f"plate_{stem}.png"))
        region = court_region(plate.shape)

        boot = None
        s = serves.get(stem)
        if s and s["server"] != "?":
            boot = toss_bootstrap(stem, s, players, shifts, plate, clip_path)
        if boot is None:
            boot = mover_bootstrap(stem, players, shifts, plate, clip_path, n, region)
        if boot is None:
            print(f"{stem}: NO BOOTSTRAP FOUND — skipping")
            continue
        fi, (x, y, w, h), how = boot
        # pad to a ball-friendly box, un-shift to RAW clip coords
        dx, dy = shifts.get(fi, (0, 0))
        pad = 6
        box_raw = (x - pad + dx, y - pad + dy, x + w + pad + dx, y + h + pad + dy)
        prompts[stem] = {"frame": fi, "box_raw": box_raw, "how": how}
        print(f"{stem}: bootstrap via {how} @ f{fi}, raw box "
              f"({box_raw[0]:.0f},{box_raw[1]:.0f})-({box_raw[2]:.0f},{box_raw[3]:.0f})")

        # render for eyeball sign-off
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        bx = tuple(int(v) for v in box_raw)
        cv2.rectangle(fr, bx[:2], bx[2:], (0, 255, 255), 2)
        crop = fr[max(0, bx[1] - 110):min(720, bx[3] + 110),
                  max(0, bx[0] - 170):min(1280, bx[2] + 170)]
        crop = cv2.resize(crop, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, f"{stem} f{fi} ({how})", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(BALL_DIR / f"prompt_{stem}.png"), crop)

    est = total_frames / 16 * 0.005
    print(f"\n{len(prompts)}/{len(args.clips)} bootstrapped; "
          f"{total_frames} frames ~= ${est:.2f} if sent")
    if args.dry_run:
        print("dry run: nothing sent. Review prompt_*.png, then rerun with --send")
        return

    import fal_client
    for stem, pr in prompts.items():
        clip_path = ROOT / "clips/points" / f"{stem}.mp4"
        url = fal_client.upload_file(str(clip_path))
        x1, y1, x2, y2 = (int(round(v)) for v in pr["box_raw"])  # API wants ints
        result = fal_client.subscribe(
            "fal-ai/sam-3/video-rle",
            arguments={"video_url": url, "box_prompts": [{
                "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                "frame_index": pr["frame"]}]},
        )
        (BALL_DIR / f"rle_{stem}.json").write_text(json.dumps(result, default=str))
        shifts = load_shifts(stem)
        rows, dropped = [], 0
        for i, box in enumerate(result.get("boxes") or []):
            if not box or len(box) != 4:
                dropped += 1
                continue
            cx, cy, w, h = box
            if w > 0.05:
                dropped += 1
                continue
            dx, dy = shifts.get(i, (0, 0))
            rows.append({"frame": i,
                         "cx_raw": cx, "cy_raw": cy, "w": w, "h": h,
                         "x_stab": cx * 1280 - dx, "y_stab": cy * 720 - dy})
        with open(BALL_DIR / f"ball_{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"{stem}: {len(rows)} ball frames, {dropped} dropped")


if __name__ == "__main__":
    main()
