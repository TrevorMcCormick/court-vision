# Court Vision benchmark — auto-charts vs the Match Charting Project

Seven matches, four surfaces/conditions, both tours, five broadcast
feeds. 491 points aligned to human-charted MCP ground truth via
score-bug point IDs. Ball tracking: WASB (local, free, promptless).
Marginal cost per match: ~$0. (The tables below are the running
record, oldest first; the 7-match numbers live in "Three matches in
one pass" at the bottom.)

Current numbers (as of 2026-07-10, event detector v5 — the crossing
skeleton — plus shot-direction v2, the receiver-mirrored both-halves
direction estimator, plus player-box hygiene (player_boxes.py) and the
widened letter gate; all four columns are the t*w WASB chart twins
scored by the t*w evals):

| metric            | t1 night/lefty | t2 day ctrl | t3 clay RG | t4 grass WTA |
|-------------------|:---:|:---:|:---:|:---:|
| server end        | 10/22 | 2/5 | 48/59 (81%) | 37/49 (76%) |
| rally length ±1   | 13/22 | 5/5 | 36/59 (61%) | 28/49 (57%) |
| serve zone        | 11/12 | 1/3 | 8/17 | 15/32 |
| letters (aligned) | 9/11 | 12/12 | 67/85 (79%) | 17/31 |
| ending type       | 9/16 | 2/3 | 19/42 | 9/33 |
| **acceptance ≤1 token edit** | 3/22 | 1/5 | 3/59 | 0/49 |

**Acceptance is the north star from here on:** tokenize machine and MCP
strings as [serve+zone][letter+direction]*[ending]; a point is accepted
when the two need at most ONE token edit (Levenshtein over tokens,
strict equality — a '?' matches nothing). Overall: 7/135 (5.2%), up
from 0/135 before v5 and 3/135 before the direction model; mean token
distance 7.18 -> 6.19 (direction v2) -> 6.05 (box hygiene). The metric
is brutal by design; it is the distance to "a human charter would sign
this," measured per point.
Implementation: experiments/mcp_accept.py, reported by every t*w eval.

Matches: t1 = Nadal–Shapovalov, Canada Masters R16 2017 (hard, night,
both left-handed). t2 = Federer–Haase, Canada Masters SF 2017 (hard,
day). t3 = Djokovic–Ruud, Roland Garros F 2023 (clay). t4 =
Krejcikova–Paolini, Wimbledon F 2024 (grass, WTA).

## Reading the table honestly
- t1/t2 are small-n highlights reels (the non-w SAM-era trees remain
  frozen as the A/B baseline; these columns are the WASB twins, whose
  tracked-clip base grew past the old 11/5).
- t1 rally ±1 REGRESSED under v5 (16/22 -> 13/22): the crossing
  skeleton needs crossings, and the night reel's low-recall tracks
  amputate rallies the old cusp counter padded out. On the record,
  not hidden — v5's constants were tuned on t3 only and t1 was scored
  untouched. t4 endings likewise 12/33 -> 9/33 (endings inherit the
  final-shot identity, which v5 reshuffled).
- t3 rally length is capped by the footage, not the pipeline: the RG
  editor cuts INTO rallies — early shots of long points were never
  broadcast (mean charted 4.6 vs MCP 8.1 on what exists).
- "Letters (aligned)" counts only clips whose rally length matches
  MCP exactly — the honest denominator (index comparisons are
  unreliable when lengths differ).
- Endings compare TYPE only (winner / net / wide / deep); forced vs
  unforced attribution is charter judgment the pipeline doesn't claim.
- Every number is out-of-sample: constants are tuned per staging layer
  and frozen before scoring; each re-tune is a numbered freeze in
  LOG.md with before/after tables.

## Named, still-open failure modes
- Crossing recall is now the binding constraint on rally length: v5
  counts what the spine sees, and track holes/low coverage eat
  crossings (t1 night reel worst; ~9 t3 and ~7 t4 undercounts).
- The 28-second replayed let (t4_point_23) still charts as one long
  point — score identity AND the crossing spine are both blind to a
  let by construction (the let is genuinely played ball).
- Long-rally late-track blindness vs true codas on grass (3 clips).
- Far-half in/out for endings: refuted with receipt (airborne-ball
  projection error); near-half bounces recoverable only in a 1.2 s
  fill window.
- t2 server end (2/5): thin serve detection on the control's few
  clips; superseded approach-wise by serve v3 but tree stays frozen.
- Deuce/ad stance refusal on clay (34/56 clips refuse a side).

## Acceptance decomposition (2026-07-10)

(This section is the PRE-direction-model record — the analysis that
picked the build. The after-numbers live in "Shot direction v2" below.)

Where do the 7.18 mean token edits per point actually live? The token
Levenshtein was backtraced to an alignment and every edit binned
(experiments/mcp_decompose.py — pure analysis, nothing in the pipeline
changed). Mean edits per point, all 135 scored points:

| edit category | t1 night | t2 ctrl | t3 clay | t4 grass | overall |
|---|:---:|:---:|:---:|:---:|:---:|
| deletion (shot we never charted) | 1.59 | 0.40 | 2.68 | 1.39 | **1.95** |
| sub: letter AND direction wrong  | 2.14 | 1.00 | 1.71 | 1.33 | **1.61** |
| sub: direction digit only        | 0.86 | 2.60 | 1.15 | 1.27 | **1.20** |
| sub: ending token                | 0.59 | 0.60 | 0.66 | 0.82 | 0.70 |
| insertion (phantom extra shot)   | 0.14 | 0.00 | 0.15 | 1.24 | 0.54 |
| sub: cross-type (structural)     | 0.41 | 0.20 | 0.39 | 0.53 | 0.44 |
| sub: letter only                 | 0.27 | 0.40 | 0.39 | 0.55 | 0.43 |
| sub: serve zone                  | 0.09 | 0.60 | 0.47 | 0.16 | 0.30 |
| **total (mean token distance)**  | 6.09 | 5.80 | 7.61 | 7.29 | **7.18** |

The top sinks: **direction digits** (2.81 edits/pt across the two bins
they appear in — 39% of the whole budget), **structure** (deletions +
insertions, 2.49/pt — t3's deletions are largely the editor cutting
into rallies, t4's insertions are phantom shots), and **endings**
(0.70/pt, but it's one token per point — the ending is simply wrong
more often than right). Refusal vs error, within wrong components:
directions 221 committed-wrong / 159 refused-'?', letters 113w/163r,
endings 59w/36r, serve zones 7w/34r.

Strict per-component accuracy on the 43 length-matched points ('?'
counts as wrong, unlike the committed-only rows in the table above):
serve zone 11/43 (26%), rally letter 114/209 (55%), rally direction
58/209 (28%), letter+direction both right 39/209 (19%), ending 13/43
(30%). Directions are the weakest component: attempted on 70% of rally
shots (459/660 — the far-half-only landing detector refuses the rest)
and only 48% right when attempted (206/426, vs 33% chance and vs MCP
committing a direction on 848/850 strokes). Both the recall and the
accuracy are broken.

**Edit-effort curve** — acceptance at ≤k token edits (how far the
draft is from useful, before the strict ≤1 bar):

| ≤k edits | full tokens | structural only (count + letters) |
|:---:|:---:|:---:|
| 1 | 3/135 (2.2%) | 32/135 (23.7%) |
| 2 | 9/135 (6.7%) | 57/135 (42.2%) |
| 3 | 20/135 (14.8%) | 79/135 (58.5%) |
| 5 | 56/135 (41.5%) | 102/135 (75.6%) |

Structure alone would accept 10x more points than the full metric —
the annotations stacked on the skeleton (zones, directions, endings)
are now the bottleneck, not the skeleton v5 just rebuilt.

**Acceptance headroom** — fix ONE component to MCP truth at aligned
positions, leave everything else as charted, re-score (the deliverable
table; ordering = priority):

| counterfactual | acceptance ≤1 | mean dist |
|---|:---:|:---:|
| baseline (v5 as charted) | 3/135 (2.2%) | 7.18 |
| directions perfect (all shots) | 9/135 (6.7%) | 5.88 |
| endings perfect | 7/135 (5.2%) | 6.47 |
| serve zone perfect | 5/135 (3.7%) | 6.87 |
| letters perfect | 5/135 (3.7%) | 6.63 |
| directions perfect (attempted only) | 4/135 (3.0%) | 6.33 |
| letters + directions perfect | 21/135 (15.6%) | 3.93 |
| letters + dirs + endings perfect | 53/135 (39.3%) | 3.23 |
| all components perfect (structure-only residual) | 59/135 (43.7%) | 2.93 |

Two honest reads. First: no single fix rescues acceptance — the mean
point is wrong on several axes at once, so singles top out at 6.7%.
Second: the compounding is steep and ordered. Directions are the
largest single lever; directions + letters reach 15.6%; adding endings
reaches 39.3%. Structure caps everything at 43.7% — but t3's missing
shots are footage (unrecoverable) while t4's phantom insertions are
pipeline (recoverable). The next build is a real shot-direction model:
the current digit comes from far-half-only landing detection, which is
why 30% of shots get no attempt and the attempts are near chance.
Fixing "attempted only" is worth almost nothing (3.0%) — the sparsity
and the accuracy have to be fixed together, which means inferring
direction on BOTH halves (e.g. from receiver contact geometry, not
just ball landing).

## Shot direction v2 (2026-07-10)

The build the decomposition ordered — in two parts, semantics first.

**Semantics.** MCP's direction digit is RECEIVER-END relative and NOT
handedness-flipped ("1 = a right-hander's forehand side (a lefty's
backhand)" names a fixed side of the receiving half). Our zone() had
mapped landing court-x into absolute image thirds. Every plausible
mapping, tested on all 150 committed aligned landings on
length-matched points (experiments/dir_calibrate.py):

| mapping | t1 (lefty) | t2 | t3 | t4 | overall |
|---|:---:|:---:|:---:|:---:|:---:|
| absolute asc (old zone()) | 7/24 | 6/17 | 32/79 | 12/30 | 57/150 (38%) |
| absolute desc | 9/24 | 6/17 | 33/79 | 14/30 | 62/150 (41%) |
| **receiver-end mirror, no handedness** | 11/24 | 8/17 | 46/79 | 14/30 | **79/150 (53%)** |
| receiver-end + handedness flip | 5/24 | 8/17 | 46/79 | 14/30 | 73/150 (49%) |

The both-lefty t1 match adjudicates the handedness question: 11/24 vs
5/24 against the flip. Serve zones passed the same audit (shipped
serve_zone() 11/16 vs 3/16 for its 4↔6 swap) — no serve change.

**Estimator.** shot_direction.py infers a digit for EVERY rally shot
from a measured signal ladder (quality/precedence tuned on t3 only,
n=113 aligned pairs; t1/t2/t4 held out): near-half landing 86% >
receiver's next-shot contact x 77% (81% coverage — where the ball was
received is where it went) > net-crossing x + flight slope
extrapolated to the receiver's baseline 67% (93% coverage) > far-half
landing 47% (the OLD only signal). The top available signal commits;
'?' only when no signal exists — the disagreement veto was measured
and rejected (78% vs 77% precision for 62 vs 85 net-right tokens;
acceptance charges refusal and error the same edit).

**Direction component, before → after** (attempt rate / accuracy when
attempted, aligned pairs): t3 TUNED 71%/48% → 98%/76%; held out: t1
80%/36% → 97%/66%, t2 73%/42% → 96%/88%, t4 64%/56% → 96%/75%.
Overall 70%/48% → 97%/75%.

**Acceptance:** 3/135 → 7/135 (t1 2→3, t2 0→1, t3 1→3, t4 0→0); mean
token distance 7.18 → 6.19; effort curve ≤1/2/3/5: 2.2/6.7/14.8/41.5%
→ 5.2/11.1/23.7/57.0%. Every other scorecard metric is byte-identical
— the change touches only rally direction digits. The lever is spent:
"directions perfect" headroom is now +1 point (8/135); the re-run
decomposition names the next sinks as structure (2.57 edits/pt),
letters (1.71), endings (0.70), with letters+dirs+endings-perfect at
35.6% and all-components at 44.4%.

## Letters: the box audit, box hygiene, and the SAM buy-vs-build (2026-07-10)

After direction v2 the largest substitution sink was letters (1.71
edits/pt across their two bins; 55% strict positional accuracy). The
letter reads ball-x vs box-center-x at the refined contact frame,
gated on the ball reaching the box — so it inherits every failure of
the $0 bgsub player boxes. Quantified before building
(experiments/box_letter_audit.py, on the pre-fix charts): 45% of
aligned rally letters were read off a BAD box (implausible by
court-half/size sanity, or absent), and those ran 39% right vs 71% on
sane boxes. Box-quality ceiling: ~+26 strict letters if every box hit
the sane-box rate.

**Cheap fixes shipped** (player_boxes.py — court-half plausibility,
x-only teleport rejection, short-gap interpolation — plus a letter
gate widened to the clip's typical body height; constants tuned on t3
only): strict letters 114/209 -> 117/209, letter edits 1.71 -> 1.59/pt,
mean token distance 6.19 -> 6.05, acceptance unchanged at 7/135. Every
non-letter metric byte-identical on all four scorecards. Dead ends on
the record: a height-vs-depth gate (box size is bimodal — partials vs
full-body — on both ends), a y-term in the teleport gate (partial-blob
foot_y flicker fakes teleports), and a multi-frame letter vote
(post-contact flight frames poison it). Cheap fixes plateau at ~1/8th
of the box-condition ceiling: the far player is simply NOT IN the
bgsub CSV at contact on the failing shots.

**SAM-3 A/B, one tree (t3), authorized spend:** players re-tracked via
fal sam-3/video-rle (experiments/t3_sam_players.py — prompts derived
automatically from the bgsub boxes, split-and-stitch past the ~490
frame chunk limit, per-side repair calls for the API's silent
one-object regression; ~97 calls, ~$12 est). bgsub CSVs untouched;
the twin switches via PLAYERS_DIR. Result, same eval, same hygiene:

| t3 letters              | bgsub+hygiene | SAM-3 |
|---|:---:|:---:|
| strict positional (aligned) | 67/114 (59%) | 76/114 (67%) |
| committed-aligned (eval)    | 67/85 (79%)  | 75/89 (84%)  |
| letters (all)               | 138/174      | 162/205      |

+21 gains (16 far-side; 15 are refusals turned right) vs -13 losses —
and the losses concentrate in the 6 clips whose far player bgsub NEVER
boxed (no automatic prompt is derivable, so those segments stay
far-less under SAM too; t3_point_33 alone returns 4). Acceptance and
every non-letter metric unchanged — letters alone do not move the ≤1
bar. The shipped default stays bgsub ($0); the SAM CSVs, raw masks,
and the delta are the buy-vs-build record for the consolidation
decision.

## Consolidation + per-point confidence (2026-07-10)

**The package.** The experiment sprawl is now `courtvision/` — one
chart assembler and one eval instead of four hand-synced twins each,
per-broadcaster staging in `data/matches/<id>.yaml`, CLI in
`docs/USAGE.md`. The regression gate PASSED: rerunning chart+eval
through the package reproduces all 138 chart CSVs byte-for-byte and
all four scorecards byte-identically — acceptance stays 7/135.
`experiments/` is frozen history; divergence policy in
`courtvision/README.md`.

**The confidence layer.** A 57%-within-5-edits draft is only usable if
the charter knows WHICH points to trust before looking. Per-point
signals already computed by the pipeline (serve commit/margin, striker
conflicts, ball coverage + holes, letter/direction refusals + contact
distances, direction signal-tier quality, ending commit,
crossings-vs-shots consistency, shot count, and the mid-rally-start
signature — a "serve" called 0 s into the clip whose ball-launch cy
sits INSIDE the court is a rally crossing in costume) feed a numpy
logistic + one mechanistic gate. Calibration discipline:
leave-one-match-out across the 4 matches; target = token edit distance
to MCP truth.

**Two tiers ship, not three.** The wished-for top tier — "sign off at
a glance," within 2 token edits at ≥85% precision — was built first
and does NOT survive LOMO: 50% held-out precision at 1.5% coverage.
135 points at an 11% base rate cannot support it; that's on the
record, not hidden. What survives is the **usable-draft bar (≤5 token
edits)**, the same line the effort curve already named:

| LOMO (held-out) | high precision (≤5 edits) | coverage | low-tier ≤5 rate |
|---|:---:|:---:|:---:|
| t1 night  | 11/11 (100%) | 50.0% | 27% |
| t2 ctrl   | 3/3 (100%)   | 60.0% | 50% |
| t3 clay   | 16/19 (84%)  | 32.2% | 43% |
| t4 grass  | 11/11 (100%) | 22.4% | 50% |
| **pooled**| **41/44 (93%)** | **32.6%** | 44% |

Flag × edit-distance confusion (LOMO, pooled):

| flag | 0-1 | 2 | 3-5 | 6+ | total |
|---|:---:|:---:|:---:|:---:|:---:|
| high | 6 | 6 | 29 | 3  | 44 |
| low  | 1 | 2 | 37 | 51 | 91 |

Read it honestly: HIGH means "start from the draft" (93% of
high-flagged points need ≤5 token edits; only 3/44 are disasters),
not "the draft is right" (27% of high-flagged are within 2). LOW
means "expect heavy correction or re-chart" — 56% of low-flagged
points are 6+ edits out. t3 is the weak fold because its disease
(the editor cutting into rallies) is only partially observable; the
launch-plausibility gate catches the clips whose "serve" was a rally
crossing, and what remains is footage we cannot see.

The shipped scorer (`data/confidence_model.json`, all-data fit by the
same threshold rule) flags 45/135 scored points high at 96% in-sample
precision; the honest generalization estimate is the LOMO table above.

**The exporter.** `courtvision draft <match>` emits
`outputs/<t>/export/<t>_mcp_draft.csv` — MCP points schema (match_id,
Pt, Set/Gm/Pts, Svr, 1st/2nd) with the machine string in 1st plus
confidence, conf_p, clip, serve_s (jump-to timestamp), n_shots. Across
the four matches: 138 draft points, 46 flagged high.

## Three matches in one pass: AO, USO-W, Turin (2026-07-11)

The calibration layer's stated disease was n: 135 points, an 11% base
rate, and a top tier that died in LOMO. The fix is more aligned points,
so three matches were staged end-to-end through the package stages
(`fitcourt` / `probe` / `extract` / `boundaries` / `align` — no script
twins), chosen to maximize aligned-point count first and axis coverage
second: **t5** Sinner–Zverev, AO F 2025 (hard, night, AO feed; 190 MCP
points), **t6** Sabalenka–Pegula, US Open F 2024 (hard, WTA, USO feed;
167), **t7** Djokovic–Sinner, ATP Finals RR 2023 (INDOOR hard, Tennis
TV feed; 218). All from "condensed match" uploads — 20–40 min re-cuts
that keep most points, which beat extended highlights for alignment
yield. 372 clips extracted, 356 points scored.

What each feed broke, on the record:
- **t5 (AO)**: the fit window is everything — the first pick had
  26.5 px of pan smearing the ECC-translation plate; a motion scan
  found a static 5 s serve setup and the fit landed at ≤0.2 px
  residuals. The court-only hull erodes the boundary lines (t3's
  lesson re-learned); the hull must include the apron.
- **t6 (USO)**: blue court + GREEN apron — one HSV band can't hold
  both, so `court_detect` grew a second hull band (`hull2_lo/hi`),
  OR'd in. Generalization by config, not code forks.
- **t7 (Turin)**: the light-blue apron rides V≈252 — the value ceiling
  had to open to 255. Far-baseline reprojection bows ~7.7 px (lens
  distortion, same class as t3's documented bow; accepted).
- **All three bugs hide zero-valued columns**: at 0-0 the points
  column vanishes, the era window sees live background, and the
  plateau machinery splits game-start points into duplicate-score
  fragment groups (t6: 4 groups, t7: 7). Adjudication receipts: MCP's
  own 1st/2nd columns arbitrate most of them (a fragment pair whose
  MCP row says `4w` + second-serve rally IS the fault + the point,
  filmed separately); the rest fall to changeover parity + eyeballed
  frame strips. Losers are blanked in the alignment CSV, so the eval
  skips them rather than double-charging one MCP row.
- **Changeover parity after a tiebreak set is NOT "13 games"**: the
  set-2 TB at Turin went 7-4 (11 points → one internal end-change at
  6 pts), and the set-end change cancels it — the TB set contributes
  EVEN swap parity. The set-3 fold of the server-end vote flipped
  from 17/37 wrong to 37/17 right when the prior moved 25 → 24.
  Verified on video before shipping.

Scorecards (same eval, same constants — nothing was re-tuned on the
new matches; the only new knobs are per-feed `court_detect` staging):

| metric            | t5 AO night | t6 USO WTA | t7 Turin indoor |
|-------------------|:---:|:---:|:---:|
| server end        | 53/71 (75%) | 121/128 (95%) | 133/157 (85%) |
| rally length ±1   | 47/71 (66%) | 99/128 (77%) | 131/157 (83%) |
| serve zone        | 11/26 | 44/96 | 62/125 |
| letters (aligned) | 93/117 (79%) | 120/148 (81%) | 112/157 (71%) |
| ending type       | 20/50 | 24/83 | 23/95 |
| **acceptance ≤1 token edit** | 2/71 | 10/128 | 9/157 |

Pooled acceptance across all seven matches: **28/491 (5.7%)**, from
7/135 (5.2%) — the bar held under a 3.6x bigger, feed-diverse test.
t6 is the best single-match acceptance on record (7.8%): WTA + the
USO feed's stable wide camera chart cleanly. t5's weak server end
(75%) traces to the AO night feed's serve-end detection, not
alignment — its parity vote was 53/17, decisive but noisy.

**Recalibration, 4 → 7 matches (491 points, base rate ≤5 edits 67%).**
The LOMO table, each match scored by a model that never saw it:

| LOMO (held-out) | high precision (≤5 edits) | coverage | low-tier ≤5 rate |
|---|:---:|:---:|:---:|
| t1 night   | 10/10 (100%) | 45.5% | 33% |
| t2 ctrl    | 3/3 (100%)   | 60.0% | 50% |
| t3 clay    | 11/12 (92%)  | 20.3% | 47% |
| t4 grass   | 18/26 (69%)  | 53.1% | 52% |
| t5 AO      | 3/4 (75%)    |  5.6% | 49% |
| t6 USO     | 34/35 (97%)  | 27.3% | 69% |
| t7 Turin   | 13/14 (93%)  |  8.9% | 71% |
| **pooled** | **88% (92/104)** | **21.2%** | 61% |

Flag × edit-distance confusion (LOMO, pooled):

| flag | 0-1 | 2 | 3-5 | 6+ | total |
|---|:---:|:---:|:---:|:---:|:---:|
| high | 13 | 15 | 64 | 12  | 104 |
| low  | 15 | 52 | 170 | 150 | 387 |

Read against the old 93%/32.6%: precision -5 pts, coverage -11 pts —
and that is the honest direction. The 4-match table was 44 flags; this
one is 104, and the new folds expose t4 as the weak match (69% — its
phantom-insertion disease flies under signals tuned to detect missing
data, not invented data). The strict ≤2-edit tier was re-attempted at
n=491 and still dies in LOMO (0% coverage at a 19% base rate) — still
on the record, still not shipped. The shipped all-data scorer
(`data/confidence_model.json`, t_high=0.731) flags 102/491 high at 94%
in-sample; exports regenerated for all seven matches — 508 draft
points, 104 flagged high.

Process note, on the record: this run stalled overnight when the
background-job watcher died silently between t6 ball tracking and t7
players — the jobs finished; nothing was listening. The remaining long
stages were polled from the driving loop instead.

## The t4 autopsy: whole-point gates (2026-07-11)

The 7-match recalibration named t4 the weak fold: 69% held-out HIGH
precision (18/26), every other fold ≥ 84%. Autopsy of the 8 false-highs
(held-out flag, > 5 token edits), mechanisms named from pixels per the
house rule:

| clip | d_tok | mechanism (verified on frames) |
|---|:---:|---|
| t4_point_02 | 13 | **half-cadence chart**: real exchange every ~0.9 s (near hit f245, far hit ~f272, near winding up f293), charted shots every ~1.5 s — every other stroke missed, striker chain alternates cleanly over the top (conflicts 0, crossings_gap 0: the track is blind in the same places) |
| t4_point_08 | 7 | half-cadence + 1 pre-serve weak crossing |
| t4_point_11 | 10 | **dissolve-cut mid-rally join**: clip opens on a crossfade INTO a live rally; a rally stroke at 1.5 s passes the stance check as the "serve"; the 8 strokes before it don't exist to any signal |
| t4_point_35 | 14 | dissolve-cut join: crowd cutaway, then ~5 s of live rally BEFORE the charted "serve" at f148 (3 weak net crossings pre-serve — the tell) |
| t4_point_39 | 8 | mid-rally join (1 pre-serve crossing) + letter garbage |
| t4_point_43 | 6 | letter/direction accumulation on a 10-shot rally, one edit over the bar |
| t4_point_46 | 6 | same class as 43 |
| t4_point_49 | 6 | **spineless rally**: 4-shot chart, ZERO net crossings in the window even at the weak gates — a rally story the track never told |

What made them look trustworthy: the t4-fold model leans on serve
commitment (committed + zone + launch plausible + n_shots), and
`serve_launch_plausible` is VACUOUS on t4 — every t4 serve is
stance-called (src=players), so the gate defaults to pass and
contributes +0.59 logit to every point unexamined. The failures are
whole-point failures (the chart is a fragment, or an invention) that
per-shot quality signals structurally cannot see.

The fix — three additions to `courtvision/confidence.py`, two of them
mechanistic gates in the launch-gate family (rules, not weights):

- `xr_pre_serve` (signal + gate at ≥ 2): weak-gated net crossings that
  END before the charted serve. Rally-speed ball flight before our
  "serve" means the clip joined the point mid-rally and the chart
  cannot be the whole point — the stance-called-serve blind spot the
  launch gate can't inspect. Also fed to the logistic as a feature.
- `rally_spineless` (gate): a chart claiming 3+ shots whose window
  holds zero weak crossings has no spine at all.
- `mean_shot_gap_s` (sidecar signal, not in the model): charted
  inter-shot cadence. A live exchange runs ~0.7–1.1 s; 1.5 s+ means
  every other stroke is missing (t4_point_02). Cross-feed AUC 0.62 —
  real but too weak to earn a model seat (its marginal LOMO flags ran
  50/50); it travels to the export sidecar for the charter's eyes.

**Recalibration (LOMO, 491 points, same discipline):**

| LOMO (held-out) | before | after |
|---|:---:|:---:|
| t1 night   | 10/10 (100%) @ 45.5% | 10/10 (100%) @ 45.5% |
| t2 ctrl    | 3/3 (100%) @ 60.0%   | 3/3 (100%) @ 60.0% |
| t3 clay    | 11/12 (92%) @ 20.3%  | 11/12 (92%) @ 20.3% |
| t4 grass   | 18/26 (69%) @ 53.1%  | **17/20 (85%) @ 40.8%** |
| t5 AO      | 3/4 (75%) @ 5.6%     | 3/3 (100%) @ 4.2% |
| t6 USO     | 34/35 (97%) @ 27.3%  | 35/36 (97%) @ 28.1% |
| t7 Turin   | 13/14 (93%) @ 8.9%   | 11/12 (92%) @ 7.6% |
| **pooled** | **92/104 (88%) @ 21.2%** | **90/96 (94%) @ 19.6%** |

Flag × edit-distance confusion (LOMO, pooled, after):

| flag | 0-1 | 2 | 3-5 | 6+ | total |
|---|:---:|:---:|:---:|:---:|:---:|
| high | 13 | 16 | 61 | 6   | 96 |
| low  | 15 | 51 | 173 | 156 | 395 |

Disasters in the high tier halved, 12 → 6. Coverage paid 1.6 pts
(21.2% → 19.6%) — the honest trade: t4's flags drop from 53% of the
match (barely above its 61% base rate) to 41% at 85%. Five of the
eight false-highs are de-flagged, including all the worst (d 13, 14,
8, 7). The residue is on the record: t4_point_11 (d=10) stays flagged
— its dissolve-cut join leaves no pre-serve crossings because the
track is blind there too — and 43/46 (d=6) are one edit over the bar.
The strict ≤2-edit tier under the new gates reads 86% (6/7) at 1.4%
coverage in LOMO — first time above water, still nowhere near
shippable n; still not shipped. Shipped scorer refit on all 491
(t_high=0.753) flags 97/491 high at 96% in-sample; exports
regenerated for all seven matches — 508 draft points, 99 flagged high.

Full history: LOG.md. Landscape context: docs/landscape-2026-07.md.
