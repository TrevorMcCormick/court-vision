"""M0 — first tracked rally.

Uploads a broadcast rally clip to fal.ai, runs SAM 3 video segmentation with
text prompts, and saves:

  outputs/m0/overlay.mp4          — SAM 3's masked overlay video (the artifact)
  outputs/m0/response_video.json  — raw response from fal-ai/sam-3/video
  outputs/m0/response_rle.json    — raw response from fal-ai/sam-3/video-rle
  outputs/m0/bboxes/              — unzipped per-frame bounding box data, if any
  outputs/m0/trajectory.csv/.png  — ball trajectory, if per-frame data is parseable

The RLE/bbox payload formats aren't fully documented, so this script dumps
everything raw first and parses best-effort second.

Usage:
    uv run experiments/m0_track_rally.py clips/rally.mp4 [--prompt "tennis ball, tennis player"]
"""

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import fal_client
import requests

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m0"


def log_queue(update):
    if isinstance(update, fal_client.InProgress):
        for entry in update.logs or []:
            print(f"    fal: {entry.get('message', '')}")


def run_endpoint(endpoint, video_url, prompt, threshold):
    print(f"==> {endpoint} (prompt={prompt!r}, threshold={threshold})")
    result = fal_client.subscribe(
        endpoint,
        arguments={
            "video_url": video_url,
            "prompt": prompt,
            "apply_mask": True,
            "detection_threshold": threshold,
        },
        with_logs=True,
        on_queue_update=log_queue,
    )
    return result


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"    saved {dest} ({len(r.content):,} bytes)")


def save_json(obj, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, indent=2, default=str))
    print(f"    saved {dest}")


def extract_files(result, stem):
    """Download every file-like value in the response and unzip archives."""
    for key, val in (result or {}).items():
        if isinstance(val, dict) and "url" in val:
            name = val.get("file_name") or f"{key}"
            dest = OUT_DIR / f"{stem}_{name}"
            download(val["url"], dest)
            if dest.suffix == ".zip":
                target = OUT_DIR / f"{stem}_{dest.stem}"
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(target)
                members = zf.namelist()
                print(f"    unzipped {len(members)} files -> {target}")
                print(f"    first entries: {members[:5]}")


def try_trajectory(rle_result):
    """Best-effort: pull per-frame, per-object detections out of the RLE response.

    Looks for a list of frames each carrying objects with either a bbox or an
    RLE mask. Logs the shape of whatever it finds so the format can be
    understood even when parsing fails.
    """
    candidates = []
    for key, val in (rle_result or {}).items():
        if isinstance(val, list) and val:
            candidates.append((key, val))
    if not candidates:
        print("    no list-valued fields in RLE response — inspect the raw JSON")
        return

    for key, frames in candidates:
        print(f"    candidate field {key!r}: {len(frames)} entries; "
              f"first entry keys: {list(frames[0].keys()) if isinstance(frames[0], dict) else type(frames[0])}")

    key, frames = candidates[0]
    rows = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        objects = frame.get("objects") or frame.get("detections") or frame.get("masks") or []
        for obj in objects if isinstance(objects, list) else []:
            box = obj.get("bbox") or obj.get("box")
            if box and len(box) == 4:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                rows.append({
                    "frame": frame.get("frame_index", i),
                    "object_id": obj.get("id", obj.get("object_id", "?")),
                    "label": obj.get("label", obj.get("prompt", "?")),
                    "cx": cx,
                    "cy": cy,
                })
    if not rows:
        print("    could not extract centroids automatically — raw JSON saved for manual inspection")
        return

    csv_path = OUT_DIR / "trajectory.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"    saved {csv_path} ({len(rows)} detections)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    by_obj = {}
    for r in rows:
        by_obj.setdefault((r["object_id"], r["label"]), []).append(r)
    for (oid, label), pts in by_obj.items():
        pts.sort(key=lambda r: r["frame"])
        ax.plot([p["frame"] for p in pts], [p["cy"] for p in pts],
                marker=".", markersize=3, linewidth=1, label=f"{label} #{oid}")
    ax.invert_yaxis()
    ax.set_xlabel("frame")
    ax.set_ylabel("y position (px, image coords)")
    ax.set_title("M0 — vertical position per tracked object (bounces/hits = kinks)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = OUT_DIR / "trajectory.png"
    fig.savefig(plot_path, dpi=150)
    print(f"    saved {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", help="path to local video clip")
    parser.add_argument("--prompt", default="tennis ball, tennis player")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--skip-rle", action="store_true")
    args = parser.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        sys.exit(f"clip not found: {clip}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"==> uploading {clip} ({clip.stat().st_size:,} bytes) to fal")
    video_url = fal_client.upload_file(str(clip))
    print(f"    {video_url}")

    result = run_endpoint("fal-ai/sam-3/video", video_url, args.prompt, args.threshold)
    save_json(result, OUT_DIR / "response_video.json")
    extract_files(result, "video")

    if not args.skip_rle:
        rle = run_endpoint("fal-ai/sam-3/video-rle", video_url, args.prompt, args.threshold)
        save_json(rle, OUT_DIR / "response_rle.json")
        extract_files(rle, "rle")
        try_trajectory(rle)

    print("==> done. Artifacts in", OUT_DIR)


if __name__ == "__main__":
    main()
