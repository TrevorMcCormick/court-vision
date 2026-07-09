# Court Vision

Auto-charting tennis matches from broadcast video.

**The goal:** broadcast video in → [Match Charting Project](https://tennisabstract.com/charting/meta.html)
point-by-point notation out. Ball and player tracking comes from
[SAM 3](https://ai.meta.com/blog/segment-anything-model-3/) (via the
[fal.ai hosted endpoint](https://fal.ai/models/fal-ai/sam-3/video)); the
charting logic is what this repo builds on top of it.

Progress is documented in public at [trmccormick.com](https://trmccormick.com).

## Milestones

- **M0** — Track ball and players through one rally with SAM 3 *(done — ball tracked via box prompt; players carried forward)*
- **M1** — Court keypoints → homography → real court coordinates *(done — clean plate + Hough + four corners, sub-pixel on held-out lines)*
- **M2** — Detect hits and bounces from ball trajectory *(done — 13/13 events frame-verified on the M0 rally)*
- **M3** — Rally segmentation → shot sequences → MCP notation *(in progress — players tracked, f/b letters 7/7 frame-verified on the M0 rally; needs full-point clips)*
- **M4** — Validate against a human-charted MCP match

## Setup

```bash
uv sync
export FAL_KEY=...   # fal.ai API key
```

## Layout

- `experiments/` — one-off milestone scripts, numbered by milestone
- `courtvision/` — the package; code graduates here from experiments
- `clips/` — input video (gitignored, never committed)
- `outputs/` — generated artifacts: overlay videos, trajectories, plots (gitignored)

Video files are never committed or redistributed — clips are downloaded
locally for research use only.
