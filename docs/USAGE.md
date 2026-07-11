# Court Vision — usage

Broadcast tennis video in; per-point chart CSVs, MCP-style strings,
and a confidence-flagged charting draft out. Everything runs locally
at $0 marginal cost per match.

## Install

```bash
git clone <repo> && cd court-vision
uv sync
```

That covers charting, eval, confidence, and export against existing
trees. Only the **ball tracker** needs more: clone
[WASB-SBDT](https://github.com/nttcom/WASB-SBDT) into
`third_party/WASB-SBDT` and put their tennis checkpoint at
`third_party/weights/wasb_tennis_best.pth.tar` (MIT license; see
MODEL_ZOO.md in their repo). Runs on MPS/CUDA/CPU.

## The commands

```bash
uv run python -m courtvision chart t3          # video+config -> chart CSVs
uv run python -m courtvision eval t3           # charts vs MCP -> scorecard
uv run python -m courtvision draft t3          # chart + confidence + export
uv run python -m courtvision export t3         # just the draft CSV
uv run python -m courtvision calibrate         # refit + report confidence
uv run python -m courtvision decompose         # where the edits live

# upstream stages (their outputs are frozen for t1-t4; run for NEW matches)
uv run python -m courtvision fitcourt t5       # reel + court_detect -> homography
uv run python -m courtvision probe t5          # court-view detection -> segments
uv run python -m courtvision.boundaries --tree t5   # score-bug point splits
uv run python -m courtvision extract t5        # segments -> clips + offsets
uv run python -m courtvision players t5        # bgsub boxes (pass A + B)
uv run python -m courtvision track-ball t5     # WASB ball tracks
uv run python -m courtvision serve t5          # serve detection v3
uv run python -m courtvision align t5          # clips -> MCP rows by score
```

`t3` is a match id = a file in `data/matches/`. `all` fans out over
every configured match.

## Config anatomy (`data/matches/<id>.yaml`)

```yaml
id: t3
clips_dir: clips/points_t3        # one mp4 per point
out_dir: outputs/t3               # homography, tracks, charts live here
players_dir: players              # or players_sam (same schema, paid boxes)
clip_offsets: clip_offsets.csv    # wandering camera; null when it holds
lefty: {near: false, far: false}  # the letter read mirrors for a lefty

staging:                          # the sanctioned per-broadcaster gates
  lock_serve: true                # confident serve call locks the chain
  serve_zone_requires_side: true  # no deuce/ad stance -> no zone claim
  near_ending_fill: true          # near-half V-cusp ending recovery
  coda_report: false              # report v5's dead-ball-coda cuts (t4)

serve_detect:
  variant: ball                   # ball-adjudicated (clay) | stance (grass)
  center_tol_m: 4.3               # clay servers stand wide

video: clips/t3_reel_30fps.mp4    # the full reel (staging stages only)
court_detect:                     # fitcourt + probe: the court's color
  hull_lo: [0, 60, 120]           # band (court AND its apron — boundary
  hull_hi: [15, 255, 255]         # lines erode off a court-only hull),
  fit_lo: 36240                   # plus a hand-picked STATIC court-view
  fit_hi: 36540                   # run for the plate fit
  # optional: hull2_lo/hull2_hi (second color band, e.g. USO's green
  # apron), tophat_t / v_min (line-mask thresholds per lighting)

eval:                             # changeover parity vs MCP ground truth
  mcp_map: data/mcp/t3_mcp_map.csv
  alignment: data/mcp/t3_clip_alignment.csv
  start_end: far                  # player 1's end in game 1
  set_priors: {"*,1": 13, "*,2": 22}   # games played before each set state
  tiebreak_states: ["0,0"]        # set states where 6-6 is a tiebreak
```

Adding a match = adding a YAML and running the staging stages in the
order above (t1-t4 staging predates them and stays frozen). Two steps
remain by-eye work: picking the static fit window (and LOOKING at
model_reprojection.png), and transcribing each clip's score bug into
data/mcp/<id>_clip_alignment.csv. New score-bug layouts also need a
CFG entry in courtvision/boundaries.py (measured per-broadcast crop
windows stay in-module; see the docstring there).

## What comes out

`chart` writes `outputs/<t>/charts_wasb/`:
- `chart2_<clip>.csv` — one row per shot: frame, contact frame +
  distance, striker end, f/b letter, zone/direction digit, landing.
- `match_chart_v2.csv` — one row per point: server (detector and
  chart-overridden), hit/bounce/hole/conflict counts, ending, and the
  MCP-style string (e.g. `s6b3f1d@`).

`eval` prints the scorecard: server end, rally length ±1, serve zone,
letters (aligned = length-matched clips only, the honest denominator),
ending type, and **acceptance** — points within ONE token edit of the
human MCP chart, the project's north star.

`draft`/`export` writes `outputs/<t>/export/<t>_mcp_draft.csv` — the
MCP points schema (match_id, Pt, Set/Gm/Pts, Svr, 1st/2nd) with the
machine string in `1st` plus the triage columns: `confidence`,
`conf_p`, `clip`, `serve_s` (jump-to timestamp), `n_shots`. The
pipeline cannot see faults, so every draft is written as a
first-serve point; fixing that is part of the charter's pass.

## What the confidence flags mean

Calibrated leave-one-match-out on the 4 benchmark matches against
token edit distance to MCP truth (full tables in `docs/benchmark.md`):

- **high** — start from the draft. ~93% of high-flagged points are
  within 5 token edits (LOMO pooled; 84-100% per held-out match), at
  ~1/3 coverage. It does NOT mean the point is right: only ~27% of
  high-flagged points are within 2 edits.
- **low** — expect heavy correction; more than half of low-flagged
  points are 6+ edits out. Treat as "re-chart with the draft as a
  hint."

There is deliberately no "sign-off" tier: a within-2-edits flag at
≥85% precision did not survive leave-one-match-out (50% at 1.5%
coverage) — 135 points at an 11% base rate cannot support it. When
the benchmark grows, `calibrate` refits and reports honestly.

## Correcting drafts: the review tool

`review` opens a local web UI for correcting a match's draft export
against its clips — the charter-assist surface, and the instrument
for the cv-18 stopwatch experiment (protocol: docs/cv18-protocol.md).

    uv run python -m courtvision review t6 --mode review --session r1
    uv run python -m courtvision review t6 --mode cold --session c1 \
        --seed cv18-a --n 10          # drafts hidden, gradeable rows

Sessions live in `outputs/<match>/review/<session>/` (manifest,
events.jsonl telemetry, corrected.csv) and resume if rerun. `cold`
mode hides everything machine-derived (draft string, confidence,
serve timestamp); `review` mode pre-fills the draft. Lint warnings
under the inputs are advisory MCP-legality checks and never block.

    uv run python -m courtvision review-analyze \
        --cold-a t6:c1 --review t6:r1 --cold-b t7:c2

writes the timing/accuracy/anchoring/histogram report to the review
session's directory.
