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

## Open questions the research did not settle
1. Has anyone in the MCP community attempted assisted/automated
   charting, and is the charter shortage documented beyond the Jan
   2026 post? (Zero claims survived on this — absence of evidence.)
2. Published SAM2/SAM3-vs-specialist evidence on small fast balls:
   none found. Our A/B is potentially a first.
3. What do Hawk-Eye/PlaySight output per-shot, and is any of it
   individually accessible via TDI/Tennis IQ licensing?
4. F3Set label provenance/license — is it MCP-derived, and can it be
   intermediate training/eval data for us?

## Verified sources (23 fetched; quality per research pass)
- swing.vision/subscribe/compare; apps.apple.com SwingVision listing;
  techinthesun.com/swingvision (review)
- tennisviz.com; atptour.com Tennis IQ announcement; ubitennis.net
  data-methods + ATP-data pieces (Hawk-Eye ~EUR 60-70k/court/week)
- wingfield.io/en/products + shop.wingfield.io; baselinevision.com;
  tennisleo.com Baseline Vision review
- github.com/nttcom/WASB-SBDT (+MODEL_ZOO.md weights, spot-check 200);
  arxiv.org/pdf/2311.05237 (Table 2 p.8 = the F1 numbers);
  proceedings.bmvc2023.org/310
- github.com/yastrebksv/{TennisProject,TrackNet,TennisCourtDetector};
  github.com/ArtLabss/tennis-tracking; github.com/hgupt3/TRACE (404,
  Wayback 2025-04-30) + Slimold/TRACE mirror
- TrackNetV3: people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf,
  github.com/qaz812345/TrackNetV3 (badminton), DOI 10.1145/3595916.3626370
- arxiv 2508.09650 (TOTNet), 2411.11922, 2511.16719, 2511.17045 (SAM-line)
- tennisabstract.com/blog/2026/01/03/17000-matches; Sackmann MCP repo;
  on-the-t.github.io Match-Charting-GUI (2016)
- arxiv 2504.08222 (F3Set); github.com/HaydenFaulkner/Tennis;
  arxiv 2207.10213

Method note: claims verified by 3-vote adversarial panels (2/3 refutes
to kill); 25/25 survived. Commercial claims lean on vendor pages, each
corroborated by at least one independent source. Benchmark figures are
the WASB authors' re-implementations on a single match's dataset.
