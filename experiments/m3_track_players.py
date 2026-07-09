"""M3 experiment 1 — track BOTH players with SAM 3 box prompts.

M0's unfinished business: players were carried forward when ball tracking
took the milestone. The proto-chart (m3_proto_chart.py) made them critical
path — shot type f/b needs the striker's position and contact side.

Approach: the exact trick that caught the game ball. Visual box prompts on
frame 240 (one per player, pixel coords), single call to
fal-ai/sam-3/video-rle. Open question this run answers: does the endpoint
return PER-OBJECT boxes when given multiple box prompts, or does it merge
them like text prompts did? Dump raw first, parse second (M0 house rule).

Usage:
    uv run experiments/m3_track_players.py clips/rally.mp4
"""

import argparse
import json
from pathlib import Path

import fal_client

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"

# Player boxes in frame 240, pixel coords of 1280x720 (found by inspection,
# verified rendered: outputs/m3/prompt_boxes_240.png). The red-shirted line
# judge behind Gasquet is OUTSIDE the box — exclusion by construction.
PROMPTS = {
    "zverev": (498, 408, 568, 570),   # near player, dark kit, back to camera
    "gasquet": (695, 70, 738, 139),   # far player, white shirt, facing camera
}
PROMPT_FRAME = 240


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    if args.video.startswith("http"):
        video_url = args.video
    else:
        video_url = fal_client.upload_file(args.video)
        print(f"uploaded -> {video_url}")

    box_prompts = []
    for name, (x1, y1, x2, y2) in PROMPTS.items():
        box_prompts.append({
            "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
            "frame_index": PROMPT_FRAME,
        })
        print(f"prompt {name}: {box_prompts[-1]}")

    print("==> fal-ai/sam-3/video-rle, 2 box prompts, one call")
    result = fal_client.subscribe(
        "fal-ai/sam-3/video-rle",
        arguments={"video_url": video_url, "box_prompts": box_prompts},
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "rle_players.json"
    raw.write_text(json.dumps(result, indent=2, default=str))
    print(f"saved {raw}")

    # Shape diagnostics before any parsing — the multi-prompt response
    # format is undocumented, same as the box format was in M0.
    for key, val in result.items():
        if isinstance(val, list):
            first = next((v for v in val if v), None)
            print(f"  {key!r}: list of {len(val)}; first non-empty entry: "
                  f"{type(first).__name__} = {str(first)[:120]}")
        else:
            print(f"  {key!r}: {type(val).__name__} = {str(val)[:120]}")


if __name__ == "__main__":
    main()
