"""M0 iteration 2 — track a single concept per call.

The combined prompt ("tennis ball, tennis player") drifted: the RLE endpoint
returned one merged object and the overlay flipped to segmenting the whole
court in fal's second processing chunk. This run tracks ONE concept per call
and extracts the per-frame box centroid trajectory.

Usage:
    uv run experiments/m0_ball_only.py <video_url_or_path> --prompt "tennis ball" --tag ball
"""

import argparse
import csv
import json
from pathlib import Path

import fal_client

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="fal URL (https://...) or local path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tag", required=True, help="output filename tag, e.g. 'ball'")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.video.startswith("http"):
        video_url = args.video
    else:
        video_url = fal_client.upload_file(args.video)
        print(f"uploaded -> {video_url}")

    print(f"==> fal-ai/sam-3/video-rle prompt={args.prompt!r} threshold={args.threshold}")
    result = fal_client.subscribe(
        "fal-ai/sam-3/video-rle",
        arguments={
            "video_url": video_url,
            "prompt": args.prompt,
            "detection_threshold": args.threshold,
        },
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"rle_{args.tag}.json"
    raw_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"saved {raw_path}")

    boxes = result.get("boxes") or []
    scores = result.get("scores") or [None] * len(boxes)
    print(f"{len(boxes)} frames of boxes")

    rows = []
    for i, box in enumerate(boxes):
        if not box or len(box) != 4:
            continue
        rows.append({
            "frame": i,
            "cx": (box[0] + box[2]) / 2,
            "cy": (box[1] + box[3]) / 2,
            "w": box[2] - box[0],
            "h": box[3] - box[1],
            "score": scores[i] if i < len(scores) else None,
        })

    csv_path = OUT_DIR / f"trajectory_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "cx", "cy", "w", "h", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path} ({len(rows)} rows)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    frames = [r["frame"] for r in rows]
    axes[0].plot(frames, [r["cx"] for r in rows], ".-", markersize=3, linewidth=0.8)
    axes[0].set_ylabel("x (normalized)")
    axes[1].plot(frames, [r["cy"] for r in rows], ".-", markersize=3, linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("y (normalized)")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("frame (30 fps)")
    fig.suptitle(f"M0 — {args.prompt!r} centroid per frame")
    fig.tight_layout()
    plot_path = OUT_DIR / f"trajectory_{args.tag}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
