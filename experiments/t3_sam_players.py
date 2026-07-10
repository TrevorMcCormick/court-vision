"""T3 — SAM-3 player tracking via fal, the buy-vs-build experiment.

The box_letter_audit + player_boxes session established: the letter
sink's remaining mass on t3 is the far player simply NOT being in the
bgsub CSV at contact (tape ghosts, plate bakes, partial blobs), and no
temporal hygiene conjures a player the tracker never saw. M3 measured
SAM-3 player tracking excellent at ~$0.15/clip; this script re-runs
that recipe over the t3 benchmark tree ONLY (authorized spend, one
tree) so the letters delta SAM-vs-bgsub can be measured on MCP ground
truth. bgsub CSVs stay untouched in players/; SAM CSVs land in
players_sam/ and the twin switches via its PLAYERS_DIR config.

Recipe (M3 scripts, updated for the t3 fleet):
  prompts   two box prompts (near + far player), one video-rle call —
            derived AUTOMATICALLY from the bgsub boxes: the prompt
            frame is the one whose far box is tallest among frames
            where both sides pass player_boxes' court-half hygiene
            (the far box is the scarce resource; a full-body far box
            frame is a good near-box frame too). bgsub boxes live in
            ECC-STABILIZED clip space; prompts are shifted back to raw
            clip space via plates/shifts_*.csv.
  chunking  fal chunks videos past ~490 frames and drops prompts for
            later chunks ("No prompts available for this video chunk",
            M3 receipt) — clips over MAX_SEG frames are split into
            equal segments, each prompted independently, stitched by
            frame offset.
  masks     video-rle returns ONE merged RLE mask per frame; the M3
            split applies: static-kill (>85% duty pixels are scenery),
            connected components, largest component per half. Output
            rows are converted raw -> stabilized space (subtract the
            ECC shift) so the twin's touch search sees the same
            coordinate frame the ball CSV uses.

Usage:
    uv run experiments/t3_sam_players.py t3_point_24 ...   # pilot
    uv run experiments/t3_sam_players.py                   # whole tree
    (--parse-only: re-split cached raw JSON without new API calls)
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import player_boxes

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "t3"
RAW_DIR = OUT_BASE / "sam_raw"
SAM_DIR = OUT_BASE / "players_sam"
CLIP_DIR = ROOT / "clips" / "points_t3"

W, H = 1280, 720
MAX_SEG = 450            # frames; fal chunk limit is ~490 (M3 receipt)
STATIC_FRAC = 0.85       # M3 split constant
MIN_AREA_NEAR = 250      # M3 split constant
MIN_AREA_FAR = 120       # t3 far player is smaller than the dev reel's
NEAR_H_MIN_PX = 40       # prompt frame must have a credible near box
FAR_H_MIN_PX = 25        # slivers make bad prompts
MAX_ASPECT = 1.2         # prompt box w/h cap — wider is tape/banner
MAX_ASPECT_LOOSE = 3.0   # far fallback tier (player+tape merged boxes)
PAD_PX = 4               # prompt-box padding (bgsub boxes hug the diff)


def read_shifts(stem):
    shifts = {}
    with open(OUT_BASE / "plates" / f"shifts_{stem}.csv") as f:
        for row in csv.DictReader(f):
            shifts[int(row["frame"])] = (float(row["dx"]), float(row["dy"]))
    return shifts


def _prompt_box(row, fi, seg_lo, shifts):
    dx, dy = shifts.get(fi, (0.0, 0.0))
    cx, cy = float(row["cx"]) * W, float(row["cy"]) * H
    w, h = float(row["w"]) * W, float(row["h"]) * H
    return {
        "x_min": int(round(cx - w / 2 + dx)) - PAD_PX,
        "y_min": int(round(cy - h / 2 + dy)) - PAD_PX,
        "x_max": int(round(cx + w / 2 + dx)) + PAD_PX,
        "y_max": int(round(cy + h / 2 + dy)) + PAD_PX,
        "frame_index": fi - seg_lo,
    }


def _side_ok(row, side, loose=False):
    h_px, w_px = float(row["h"]) * H, float(row["w"]) * W
    if side == "near" and h_px < NEAR_H_MIN_PX:
        return False
    if side == "far" and h_px < FAR_H_MIN_PX:
        return False
    # people are taller than wide; a wide flat candidate is the net
    # tape / a banner (point_24's 358x17 "far player"). The loose tier
    # admits player+tape merges — SAM segments the salient person
    # inside the box more reliably than bgsub separates the diff.
    return w_px <= (MAX_ASPECT_LOOSE if loose else MAX_ASPECT) * h_px


def pick_side_prompt(stem, side, seg_lo, seg_hi, Hm, offsets, shifts):
    """(frame, box) for ONE side — the repair-call picker. Far: tallest
    strict candidate, then tallest loose-aspect fallback. Near: median
    height (the max is the shadow merge, the min is a partial)."""
    players = player_boxes.load(
        OUT_BASE / "players" / f"players_{stem}.csv", Hm, offsets)
    for loose in ((False, True) if side == "far" else (False,)):
        cands = [(float(s[side]["h"]), fi) for fi, s in players.items()
                 if seg_lo <= fi < seg_hi and side in s
                 and not s[side].get("interp")
                 and _side_ok(s[side], side, loose)]
        if not cands:
            continue
        cands.sort()
        _, fi = cands[-1] if side == "far" else cands[len(cands) // 2]
        return fi, _prompt_box(players[fi][side], fi, seg_lo, shifts)
    return None


def pick_prompts(stem, seg_lo, seg_hi, Hm, offsets, shifts):
    """(prompt_frame, [box, ...]) from the bgsub boxes.

    BOTH prompts must sit on the SAME frame: the fleet's first pass
    prompted each side on its own best frame and SAM silently kept only
    ONE object per request (near 0/114 on seven clips; M3 and the
    same-frame clips all worked). Ladder:
      1. frame where both sides pass the strict gates — tallest far box
         wins the tie (the far player is the scarce resource);
      2. no such frame: the scarce FAR side alone (tallest strict, then
         tallest loose-aspect fallback), else near alone (median h).
    """
    players = player_boxes.load(
        OUT_BASE / "players" / f"players_{stem}.csv", Hm, offsets)
    seg = {fi: sides for fi, sides in players.items()
           if seg_lo <= fi < seg_hi
           and not any(r.get("interp") for r in sides.values())}
    both = [(float(s["far"]["h"]), fi) for fi, s in seg.items()
            if "near" in s and "far" in s
            and _side_ok(s["near"], "near") and _side_ok(s["far"], "far")]
    if both:
        _, fi = max(both)
        return fi, [_prompt_box(seg[fi][side], fi, seg_lo, shifts)
                    for side in ("near", "far")]
    for loose in (False, True):
        far = [(float(s["far"]["h"]), fi) for fi, s in seg.items()
               if "far" in s and _side_ok(s["far"], "far", loose)]
        if far:
            _, fi = max(far)
            return fi, [_prompt_box(seg[fi]["far"], fi, seg_lo, shifts)]
    near = sorted((float(s["near"]["h"]), fi) for fi, s in seg.items()
                  if "near" in s and _side_ok(s["near"], "near"))
    if near:
        _, fi = near[len(near) // 2]
        return fi, [_prompt_box(seg[fi]["near"], fi, seg_lo, shifts)]
    return None


def cut_segment(clip, lo, hi, fps, out_path):
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (W, H))
    for _ in range(hi - lo):
        ok, fr = cap.read()
        if not ok:
            break
        wr.write(fr)
    wr.release()
    cap.release()


def decode(rle_str):
    toks = list(map(int, rle_str.split()))
    flat = np.zeros(W * H, np.uint8)
    for i in range(0, len(toks), 2):
        start, run = toks[i], toks[i + 1]
        flat[start:start + run] = 1
    return flat.reshape(H, W)


def split_segment(raw, seg_lo, shifts):
    """M3 component split, rows in STABILIZED space."""
    rles = raw["rle"]
    masks = [decode(r) if r else np.zeros((H, W), np.uint8) for r in rles]
    on_frac = np.mean([m.astype(np.float32) for m in masks], axis=0)
    static = (on_frac > STATIC_FRAC).astype(np.uint8)
    rows = []
    for i, m in enumerate(masks):
        fi = seg_lo + i
        dx, dy = shifts.get(fi, (0.0, 0.0))
        m = m & ~static
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            m, connectivity=8)
        halves = {"near": [], "far": []}
        for lab in range(1, n):
            x, y, w, h, area = stats[lab]
            side = "near" if y + h > H / 2 else "far"
            if area < (MIN_AREA_NEAR if side == "near" else MIN_AREA_FAR):
                continue
            halves[side].append((area, x, y, w, h))
        for side in ("near", "far"):
            if not halves[side]:
                continue
            area, x, y, w, h = max(halves[side])
            rows.append({"frame": fi, "player": side,
                         "cx": (x + w / 2 - dx) / W,
                         "cy": (y + h / 2 - dy) / H,
                         "w": w / W, "h": h / H,
                         "foot_x": x + w / 2 - dx,
                         "foot_y": y + h - dy, "area": area})
    return rows, len(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="*")
    parser.add_argument("--parse-only", action="store_true",
                        help="re-split cached raw JSON, no API calls")
    args = parser.parse_args()

    Hm = np.load(OUT_BASE / "H_img_to_court.npy")
    offs = {r["clip"]: (float(r["dx"]), float(r["dy"]))
            for r in csv.DictReader(open(OUT_BASE / "clip_offsets.csv"))}

    stems = args.clips or [p.stem for p in
                           sorted(CLIP_DIR.glob("t3_point_*.mp4"))]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAM_DIR.mkdir(parents=True, exist_ok=True)

    if not args.parse_only:
        import fal_client

    sent = 0
    for stem in stems:
        clip = CLIP_DIR / f"{stem}.mp4"
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        n_seg = max(1, -(-n // MAX_SEG))
        bounds = [(round(k * n / n_seg), round((k + 1) * n / n_seg))
                  for k in range(n_seg)]
        shifts = read_shifts(stem)

        all_rows = []
        ok = True
        for k, (lo, hi) in enumerate(bounds):
            raw_path = RAW_DIR / f"{stem}_seg{k}.json"
            if not raw_path.exists():
                if args.parse_only:
                    print(f"{stem} seg{k}: no cached raw, skipped")
                    ok = False
                    break
                picked = pick_prompts(stem, lo, hi, Hm,
                                      offs.get(stem, (0.0, 0.0)), shifts)
                if picked is None:
                    print(f"{stem} seg{k}: NO usable prompt frame, skipped")
                    ok = False
                    break
                pfi, boxes = picked
                if n_seg == 1:
                    video_url = fal_client.upload_file(str(clip))
                else:
                    seg_path = RAW_DIR / f"{stem}_seg{k}.mp4"
                    cut_segment(clip, lo, hi, fps, seg_path)
                    video_url = fal_client.upload_file(str(seg_path))
                    seg_path.unlink()
                print(f"{stem} seg{k} [{lo},{hi}): prompt f{pfi}, "
                      f"boxes {[(b['x_min'], b['y_min'], b['x_max'], b['y_max']) for b in boxes]}")
                try:
                    result = fal_client.subscribe(
                        "fal-ai/sam-3/video-rle",
                        arguments={"video_url": video_url,
                                   "box_prompts": boxes})
                except Exception as e:          # noqa: BLE001 — batch survives
                    print(f"{stem} seg{k}: SAM call failed: {e}")
                    ok = False
                    break
                raw_path.write_text(json.dumps(result, default=str))
                sent += 1
            raw = json.loads(raw_path.read_text())
            if not raw.get("rle"):
                print(f"{stem} seg{k}: empty rle, skipped")
                ok = False
                break
            rows, n_masks = split_segment(raw, lo, shifts)

            # REPAIR pass — the two-boxes-one-call recipe is flaky in
            # the current API: some calls silently track only ONE of
            # the prompted objects (the mask is empty inside the other
            # prompt box AT the prompt frame — receipts in the LOG).
            # A dropped side gets its own one-box call, cached to
            # {stem}_seg{k}_{side}.json, and replaces that side's rows.
            for side in ("near", "far"):
                got = sum(1 for r in rows if r["player"] == side)
                side_raw = RAW_DIR / f"{stem}_seg{k}_{side}.json"
                if got >= 0.5 * (hi - lo) and not side_raw.exists():
                    continue
                if not side_raw.exists():
                    if args.parse_only:
                        continue
                    picked = pick_side_prompt(
                        stem, side, lo, hi, Hm,
                        offs.get(stem, (0.0, 0.0)), shifts)
                    if picked is None:
                        print(f"{stem} seg{k}: {side} dropped, no prompt "
                              f"candidate for a repair call")
                        continue
                    pfi, box = picked
                    if n_seg == 1:
                        video_url = fal_client.upload_file(str(clip))
                    else:
                        seg_path = RAW_DIR / f"{stem}_seg{k}.mp4"
                        cut_segment(clip, lo, hi, fps, seg_path)
                        video_url = fal_client.upload_file(str(seg_path))
                        seg_path.unlink()
                    print(f"{stem} seg{k}: {side}-only repair call, "
                          f"prompt f{pfi}")
                    try:
                        result = fal_client.subscribe(
                            "fal-ai/sam-3/video-rle",
                            arguments={"video_url": video_url,
                                       "box_prompts": [box]})
                    except Exception as e:      # noqa: BLE001
                        print(f"{stem} seg{k}: repair call failed: {e}")
                        continue
                    side_raw.write_text(json.dumps(result, default=str))
                    sent += 1
                side_rows, _ = split_segment(
                    json.loads(side_raw.read_text()), lo, shifts)
                side_rows = [r for r in side_rows if r["player"] == side]
                if len(side_rows) > got:
                    rows = [r for r in rows if r["player"] != side]
                    rows += side_rows
            rows.sort(key=lambda r: (r["frame"], r["player"]))
            all_rows.extend(rows)

        if not ok or not all_rows:
            continue
        out = SAM_DIR / f"players_{stem}.csv"
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wr.writeheader()
            wr.writerows(all_rows)
        near = sum(1 for r in all_rows if r["player"] == "near")
        far = sum(1 for r in all_rows if r["player"] == "far")
        print(f"  {stem}: near {near}/{n} far {far}/{n} -> {out.name}")
    print(f"API calls this run: {sent}")


if __name__ == "__main__":
    main()
