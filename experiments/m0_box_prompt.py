"""M0 iteration 3 — prompt SAM 3 with a box around THE game ball.

Text prompts match every tennis ball in frame (ball kids hold spares), and the
returned box is the union of all instances. A visual box prompt selects the
single moving game ball instead (SAM 2-style single-object tracking).

Usage:
    uv run experiments/m0_box_prompt.py <video_url_or_path>
"""

import argparse
import json
from pathlib import Path

import fal_client

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m0"

# Game ball in frame 240, pixel coords of 1280x720 (found by inspection)
BALL_PX = (612, 452, 632, 472)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=240)
    parser.add_argument("--normalized", action="store_true",
                        help="send normalized coords instead of pixels")
    args = parser.parse_args()

    if args.video.startswith("http"):
        video_url = args.video
    else:
        video_url = fal_client.upload_file(args.video)
        print(f"uploaded -> {video_url}")

    x1, y1, x2, y2 = BALL_PX
    if args.normalized:
        x1, y1, x2, y2 = x1 / 1280, y1 / 720, x2 / 1280, y2 / 720

    box_prompt = {
        "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
        "frame_index": args.frame,
    }
    print(f"==> fal-ai/sam-3/video-rle box_prompt={box_prompt}")
    result = fal_client.subscribe(
        "fal-ai/sam-3/video-rle",
        arguments={"video_url": video_url, "box_prompts": [box_prompt]},
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "ballbox_norm" if args.normalized else "ballbox"
    raw = OUT_DIR / f"rle_{tag}.json"
    raw.write_text(json.dumps(result, indent=2, default=str))
    boxes = result.get("boxes") or []
    print(f"saved {raw}; {len(boxes)} frames of boxes")
    nonempty = [b for b in boxes if b and len(b) == 4]
    print(f"non-empty boxes: {len(nonempty)}")
    if nonempty:
        import csv
        rows = [{"frame": i,
                 "cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2,
                 "w": b[2] - b[0], "h": b[3] - b[1], "score": None}
                for i, b in enumerate(boxes) if b and len(b) == 4]
        csv_path = OUT_DIR / f"trajectory_{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "cx", "cy", "w", "h", "score"])
            w.writeheader()
            w.writerows(rows)
        print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
