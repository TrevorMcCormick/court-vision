# Court Vision benchmark — auto-charts vs the Match Charting Project

Four matches, three surfaces, both tours, two broadcasters beyond the
dev reel's. 168 points aligned to human-charted MCP ground truth
(alignment 108/108 on t3/t4 via score-bug point IDs). Ball tracking:
WASB (local, free, promptless). Marginal cost per match: ~$0.

Current numbers (as of 2026-07-10, event detector v5 — the crossing
skeleton; all four columns are the t*w WASB chart twins scored by the
t*w evals):

| metric            | t1 night/lefty | t2 day ctrl | t3 clay RG | t4 grass WTA |
|-------------------|:---:|:---:|:---:|:---:|
| server end        | 10/22 | 2/5 | 48/59 (81%) | 37/49 (76%) |
| rally length ±1   | 13/22 | 5/5 | 36/59 (61%) | 28/49 (57%) |
| serve zone        | 11/12 | 1/3 | 8/17 | 15/32 |
| letters (aligned) | 9/11 | 12/12 | 65/85 (76%) | 15/28 |
| ending type       | 9/16 | 2/3 | 19/42 | 9/33 |
| **acceptance ≤1 token edit** | 2/22 | 0/5 | 1/59 | 0/49 |

**Acceptance is the north star from here on:** tokenize machine and MCP
strings as [serve+zone][letter+direction]*[ending]; a point is accepted
when the two need at most ONE token edit (Levenshtein over tokens,
strict equality — a '?' matches nothing). Overall: 3/135 (2.2%), up
from 0/135 before v5. The metric is brutal by design; it is the
distance to "a human charter would sign this," measured per point.
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

Full history: LOG.md. Landscape context: docs/landscape-2026-07.md.
