# courtvision — the living package

The consolidation of the `experiments/` era into the tool someone else
can run: broadcast video in, per-point chart CSVs + MCP-style strings +
a calibrated per-point confidence flag out, graded against Match
Charting Project ground truth.

Quick start (full command docs in `docs/USAGE.md`):

    uv run python -m courtvision chart t3     # video+config -> chart CSVs
    uv run python -m courtvision eval t3      # charts vs MCP -> scorecard
    uv run python -m courtvision draft t3     # chart + confidence + export

Per-match configuration lives in `data/matches/<id>.yaml` — paths,
handedness, and the sanctioned per-broadcaster staging gates. Adding a
match means adding a YAML, not forking a script.

## Module map

| module | what it is | lifted from |
|---|---|---|
| `fitcourt` | homography fit (hull + tophat + mask-scored labels) | `t3/t4_fit_homography.py` |
| `probe` | court-view detection (interior color + line probes) | `t3/t4_court_probe.py` |
| `extract` | point clips + camera offsets + bug contact sheets | `t3/t4_extract_points.py` |
| `ball` | WASB-SBDT ball tracking (local, free, promptless) | `wasb_track_ball.py` |
| `players` | $0 bgsub player boxes (SAM-3 alt via `players_dir`) | `t*_bgsub_players.py` |
| `boxes` | player-box hygiene (plausibility, teleport, interp) | `player_boxes.py` |
| `serve` | serve detection v3 (ball-adjudicated / sliding stance) | `t3/t4_serve_detect.py` |
| `boundaries` | score-bug point boundaries (the bug is the point ID) | `point_boundary.py` |
| `events` | event detector v5 — the crossing skeleton | `events_v5.py` |
| `directions` | receiver-mirrored direction estimator (signal ladder) | `shot_direction.py` |
| `letters` | the gated f/b letter read | t*w twins |
| `endings` | ending v1 + near-half fill | t*w twins |
| `chart` | the chart assembler (striker chain, zones, MCP string) | t*w twins |
| `mcp` | tokenizers, token Levenshtein, edit backtrace | `mcp_accept.py`, `mcp_decompose.py` |
| `align` | clip -> MCP row join by score-bug score | `t*_align_mcp.py` |
| `evaluate` | the scorecard + acceptance (the north star) | `t*w_eval.py` |
| `decompose` | edit-distance decomposition + counterfactual headroom | `mcp_decompose.py` |
| `confidence` | calibrated per-point trust flags (high/low) | new (this package) |
| `export` | charting-ready MCP-schema draft CSV | new (this package) |

## Divergence policy

`experiments/` is the project's history and **stays frozen** — every
numbered freeze, dead end, and receipt in the LOG points at those
files. The package is the **living copy**: fixes and new work land
here, never there. Consolidation gate (2026-07-10): rerunning
chart+eval through this package reproduced all four benchmark
scorecards **byte-identically** (acceptance 7/135, same per-clip
lines) and every `chart2_*.csv` / `match_chart_v2.csv` byte-for-byte.

Known, deliberate divergences from the frozen scripts:

- **Event detector**: the package charts v5 only (the current
  benchmark configuration). The twins' `EVENTS = "v4"` back-switch and
  t4's post-hoc `truncate_coda` pass (superseded by v5's structural
  chain cut) remain in `experiments/` only.
- **Serve detection**: `courtvision.serve` ships the v3 variants
  (t3's ball-adjudicated, t4's sliding stance). The t1/t2 `serves.csv`
  artifacts were produced by the older fixed-window detectors and stay
  frozen as chart inputs; rerunning serve detection on those trees
  through the package would use the v3 stance recipe — the forward
  path, not a byte-reproduction of the frozen files.
- **SAM-3 players**: the fal-billed SAM fleet
  (`experiments/t3_sam_players.py`) is not lifted — it is a measured
  buy-vs-build record, not the shipped default. A match opts in by
  pointing `players_dir` at an existing `players_sam/` tree.

- **Staging stages** (`fitcourt` / `probe` / `extract`, added for the
  t5-t7 expansion): the config-driven forward path for staging NEW
  matches. The t1-t4 staging artifacts were produced by the frozen
  per-match experiment scripts and are never regenerated.

Frozen inputs the package reads but never rewrites: ground truth
(`data/mcp/`), point-boundary outputs, `serves.csv`, homographies,
clip offsets, and the non-`w` SAM-era trees — for t1-t4. The t5+
matches generate these artifacts through the package stages.
