# Court Vision benchmark — auto-charts vs the Match Charting Project

Four matches, three surfaces, both tours, two broadcasters beyond the
dev reel's. 168 points aligned to human-charted MCP ground truth
(alignment 108/108 on t3/t4 via score-bug point IDs). Ball tracking:
WASB (local, free, promptless). Marginal cost per match: ~$0.

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

Full history: LOG.md. Landscape context: docs/landscape-2026-07.md.
