# Court Vision

Auto-charting tennis matches from broadcast video.

**The goal:** broadcast video in → [Match Charting Project](https://tennisabstract.com/charting/meta.html)
point-by-point notation out, drafted by machine and finished by a
human charter. Ball tracking is [WASB](https://github.com/nttcom/WASB-SBDT)
(local, $0); player boxes are background-subtraction (with
[SAM 3](https://ai.meta.com/blog/segment-anything-model-3/) via
[fal.ai](https://fal.ai/models/fal-ai/sam-3/video) measured as the
paid alternative); the charting logic is what this repo builds.

Progress is documented in public at [trmccormick.com](https://trmccormick.com).

## Where it stands

Seven benchmark matches, 491 points scored against human-charted MCP
ground truth. About 2 in 3 draft points land within 5 token edits of
the human chart; a calibrated flag marks the half of points worth
starting from (92% held-out precision). The open question — does
correcting a draft beat charting from scratch? — is the cv-18
stopwatch experiment, which gates the roadmap.

- **[docs/scorecard.md](docs/scorecard.md)** — what to trust, what to
  check, what's broken (plain English, gaps register)
- **[docs/benchmark.md](docs/benchmark.md)** — the running numbers,
  oldest first, receipts for everything
- **[docs/model-card.md](docs/model-card.md)** — system, intended
  use, limitations, licensing
- **[docs/data-product.md](docs/data-product.md)** — consumers,
  contracts, quality commitments, lifecycle gates
- **[docs/USAGE.md](docs/USAGE.md)** — running the pipeline CLI
- **[LOG.md](LOG.md)** — the full build log, dead ends included

## Setup

```bash
uv sync
export FAL_KEY=...   # fal.ai API key — only needed for the SAM 3 player A/B
```

Ball tracking also needs WASB cloned into `third_party/WASB-SBDT`
with the tennis checkpoint at
`third_party/weights/wasb_tennis_best.pth.tar` — see
[docs/USAGE.md](docs/USAGE.md).

## Layout

- `courtvision/` — the package; code graduates here from experiments
- `experiments/` — one-off scripts and receipts, frozen history
- `data/matches/` — per-match staging configs; `data/mcp/` — ground-truth alignments
- `clips/` — input video (gitignored, never committed)
- `outputs/` — generated artifacts: charts, exports, overlays (gitignored)

Video files are never committed or redistributed — clips are downloaded
locally for research use only. Draft exports join MCP columns and
inherit CC BY-NC-SA 4.0 with attribution to Tennis Abstract and the
volunteer charters.
