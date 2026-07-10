# Landscape review — automated tennis charting (2026-07-09)

Deep-research pass: 5 search angles, 23 sources fetched, 111 claims
extracted, top 25 adversarially verified (25 confirmed, 0 refuted).
Full trail in the session workflow; distilled here for the roadmap.

## Verdict

**"Single broadcast camera → MCP string, graded against human charts"
is a genuine gap.** No commercial, academic, or open-source system
found that both (a) ingests plain broadcast video and (b) emits a
symbolic shot-by-shot chart in MCP or any standardized notation —
let alone grades itself against human charters. The novelty of this
project lives in the notation/grammar/grading layer, NOT in perception.

**The perception layer is a solved, open, free problem** — and our
SAM-3 ball tracking is reinventing it, worse:
- WASB-SBDT (BMVC 2023, NTT, MIT license): pretrained TENNIS weights,
  downloadable today (`wasb_tennis_best.pth.tar`), F1 94.0–95.6 on the
  broadcast-tennis benchmark (TrackNet dataset, τ=4px). Beat 6
  re-implemented SOTA trackers on all 5 sports.
- Same benchmark: TrackNetV2 89.4, MonoTrack 92.1. TrackNetV3 (97.5%)
  is badminton-only weights — do not confuse the lineages.
- Open pipelines already replicate our whole classical stack
  (TennisProject: TrackNet ball + 14-keypoint court net + homography +
  players + CatBoost bounce; ArtLabss/tennis-tracking similar; TRACE
  abandoned/404). Every one stops before notation. None charts.
- No published SAM-vs-specialist comparison on small fast balls exists
  — our own A/B would be a first (devlog gold either way).

## Commercial (nobody does our input+output pair)
- SwingVision: closest on output (per-shot export, AI scoring) but
  consumer iOS self-recorded footage only; app stats, not notation.
- TennisViz (ATP Tennis IQ): closest on semantics (shot type/quality/
  tactic per shot) but consumes Hawk-Eye tracking data, not video.
- Wingfield (~EUR 7k install + sub), Baseline Vision (net-post unit):
  hardware, no broadcast input, no notation export.
- Hawk-Eye/FoxTenn: tour installs, ~EUR 40–70k/court/week.

## Demand signal (search-level, not adversarially verified)
- Sackmann, Jan 2026 ("17,000 Matches"): ~17,800 charted matches,
  only 32 volunteer charters active in 2025, ~1/4 of tour matches
  covered, explicit "plea for a few more elite charters."
  Machine-drafts-human-corrects has an obvious customer.

## Prior art to study (search-level)
- F3Set / F3Set-Tennis (arXiv 2504.08222): 114 broadcast matches from
  YouTube, 11,584 rally clips, 42,846 shots annotated, 1,000+ event
  types. Closest dataset to shot-by-shot broadcast understanding —
  read the paper; check label provenance and license before leaning
  on it (possibly MCP-derived).
- TOTNet (arXiv 2508.09650): 2025 occlusion-aware ball detector,
  claims SOTA over WASB lineage.
- HaydenFaulkner/Tennis: event-level broadcast annotations.

## What this changes
1. Ball: benchmark WASB tennis weights vs our SAM tracks on our own
   MCP-graded clips. If WASB wins (likely), SAM retires from ball duty
   and marginal cost/match → ~$0. SAM stays a candidate for players.
2. Effort concentrates on the charting layer: rally grammar, striker
   chains, letters, zones, endings, eval — the part nobody has built.
3. F3Set may provide large-scale intermediate training/eval data.
