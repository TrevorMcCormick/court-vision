# Model card — Court Vision draft charter

*Following the Model Cards framework (Mitchell et al. 2019), adapted
for a composite pipeline rather than a single model. Honest by
construction: every number is out-of-sample unless marked in-sample,
and every limitation names its mechanism. Last updated 2026-07-20.*

## System description

Court Vision is not one model. It is a pipeline of narrow components,
most of them classical CV or explicit rules, with two learned pieces:

| stage | what it is | learned? |
|---|---|---|
| Ball tracking | WASB (pretrained sports-ball tracker, run locally, $0/match) | pretrained, not fine-tuned |
| Player boxes | Background-subtraction blobs + hygiene rules ($0); SAM 3 via fal.ai measured as a paid alternative (+9 strict letters on t3, ~$12/match, not shipped) | no |
| Court mapping | One homography per match from a hand-staged static window; per-frame pixel probes (50 interior + 96 line probes) reject non-court views (replays, cutaways, other angles) | no |
| Point extraction & identity | Court-view segmentation + score-bug reading; the score plateau is the point ID | no |
| Events / notation | Rule-based: net-crossing spine, serve detection (ball-launch or stance variant per feed), letters from ball-vs-box geometry, directions from a measured signal ladder, endings from landing rules | no |
| Confidence tier | Logistic regression (numpy, 11 features) + 3 mechanistic gates, predicting P(draft within 5 token edits of truth) | fit on the 491-point benchmark |

Output: one draft MCP-*style* notation string per point (not yet
MCP-legal — `s` prefix, `?` refusal tokens) plus a HIGH/LOW
confidence flag, exported as CSV
(`courtvision draft <match>`); a local review UI for correcting drafts
(`courtvision review`).

## Intended use

- **Primary:** produce draft point charts, triaged by confidence, for
  a human charter to correct — the bet (unmeasured until cv-18) is
  that correcting a green-flagged draft beats charting from scratch.
- **Users:** the project author (now); volunteer Match Charting
  Project charters (invited 2026-07-11, cv-17).
- **Out of scope / do not use for:**
  - **Unreviewed submission to the Match Charting Project** — the
    single most important boundary. Drafts are inputs to human
    correction; pushing raw drafts upstream would pollute the very
    dataset this project depends on and shares back into.
  - Publication of drafts as match statistics — at 5.7% sign-off
    acceptance the drafts are not statistics.
  - Betting, officiating, or player evaluation/scouting.
  - Faults and second serves — invisible to the pipeline; every draft
    is written as a first-serve point.
  - Forced vs unforced error attribution — charter judgment the
    pipeline does not claim.
  - Doubles, non-broadcast camera angles, amateur video, real-time
    use — never tested.
  - Commercial use of exports — they inherit CC BY-NC-SA (below).

## Factors

The axes along which performance actually varies, declared up front:
**component** (server end 84% → serve zone 26% is the whole story — a
blended accuracy would be meaningless), **broadcast feed / lighting**
(day wide-camera best, night and clay-editor feeds worst),
**confidence tier** (HIGH 92% vs LOW 44% against the ≤5-edit bar),
and **serve-call source** (ball-adjudicated vs stance-called serves
fail differently). Disaggregated tables: benchmark.md; per-feed
support tiers: scorecard.md.

## Evaluation data

**benchmark-v2** — seven matches, 491 aligned points (the three
newest chosen to maximize aligned-point count first, feed/surface
coverage second), each scored against a volunteer-charted MCP ground
truth via score-bug point identity:

hard night (t1, t5), hard day (t2), clay (t3), grass WTA (t4),
hard WTA (t6), indoor hard (t7) — both tours, five broadcast feeds,
three condensed-match uploads, two highlight reels, two extended
highlight reels.

Truth is itself audited and versioned: four staged changeover-parity
priors were found inverting server-end truth and corrected on
2026-07-20 — that correction is the v1→v2 boundary, and pre-correction
server-end numbers are non-comparable (marked in benchmark.md). t4's
letter scoring is known to be partially poisoned by a synth-serve
index shift (diagnosed, correction parked — t4's published letter
accuracy understates the pipeline). Ground-truth noise ceiling:
unmeasured — MCP charts are single-charter volunteer labels with no
inter-charter agreement estimate, so an unknown share of the gap to
100% is label ambiguity rather than pipeline error (most plausible on
letters and endings, where charter judgment enters).

## Performance (disaggregated)

Headline, pooled over 491 points: **acceptance (≤1 token edit) 5.7%**;
**67% of points within 5 token edits**; mean token distance ≈ 5.1.
Confidence tier, leave-one-match-out: **HIGH flags 48% of points at
92% precision** against the ≤5-edit bar (per-fold floor: 77% on t5,
80% on t3 — reported, not hidden). Per-component strict accuracy and
per-match tables: [benchmark.md](benchmark.md). Per-condition verdicts
in plain English: [scorecard.md](scorecard.md).

Component summary (strict, refusals count wrong): server end 84%,
directions 75% attempted at 97% attempt rate, letters ~55–67%,
endings ~30%, serve zone ~26–46%.

## Calibration discipline

- All rule constants are tuned on a named staging match and frozen
  before scoring others; every re-tune is a numbered freeze in LOG.md
  with before/after tables.
- The confidence scorer's honest numbers are leave-one-match-out; the
  shipped scorer (`data/confidence_model.json`) is an all-data refit
  by the same threshold rule, and its in-sample numbers are labeled
  as such.
- Experiment exports are frozen behind sha256 verification so
  post-hoc pipeline changes cannot silently regrade a human's session.

## Limitations (mechanisms, not disclaimers)

1. **Crossing recall is the binding structural constraint** — the
   notation is built on a net-crossing spine; track holes amputate
   rallies (night feeds worst).
2. **Footage is a hard ceiling on clay** — the RG editor cuts into
   rallies; shots that never aired cannot be charted (mean charted
   4.6 vs true 8.1 on t3).
3. **Serve zone and endings are below useful accuracy** (see
   scorecard) — far-half in/out was refuted with receipts from this
   camera geometry (airborne-ball projection error).
4. **The confidence tier cannot see whole-point failures it has no
   signal for** — two feeds (t5, t3) under-deliver the HIGH contract
   because their diseases are invisible to the features.
5. **Small n everywhere** — 491 points, 7 matches; the strict
   sign-off tier (≤2 edits at ≥85% precision) has died in
   leave-one-match-out three times and is not shipped.

## Provenance, licensing, and ethics

- Ground truth is volunteer labor: Match Charting Project charts via
  Tennis Abstract, **CC BY-NC-SA 4.0**. Exports join MCP columns and
  therefore inherit that license, with attribution — noncommercial,
  share-alike.
- Broadcast video is downloaded locally for research use only, never
  committed or redistributed; the repo ships coordinates and
  notation, not pixels.
- WASB is a pretrained third-party tracker (see `third_party/`);
  SAM 3 usage is via fal.ai's hosted endpoint under their terms.
- The system drafts observations of public professional play; it
  makes no claims about players and stores no data beyond the point
  notation the MCP format itself defines.

## Human oversight expectations

The human is part of the system — final chart quality is set by the
correction pass, not the pipeline. Per tier: HIGH → start from the
draft but verify every token against video, and always re-key serve
zones and endings regardless of flag (the tier promise is per-point
edit count, not per-component correctness); LOW → chart from scratch.
Spot-check a sample of your accepted HIGH points against video each
session. Report failures via GitHub issue (match, clip, draft string,
corrected string, flag). Maintenance is one person, best-effort, no
SLA.

## Versioning & maintenance

The unit of change is the **freeze**: a numbered LOG.md entry with
before/after scorecards. The current shipped state is reproducible
from the package CLI (`fitcourt → probe → extract → boundaries →
align → chart → eval → calibrate → draft`, see USAGE.md); the
regression gate is byte-identical chart CSVs and scorecards.
Confidence model: `data/confidence_model.json` (fields include
`t_high`, feature list, and provenance note). Roadmap and its gate:
[scorecard.md](scorecard.md) bottom section; the cv-18 stopwatch
experiment decides build order.
