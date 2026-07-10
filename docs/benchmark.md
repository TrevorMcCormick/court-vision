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

Full history: LOG.md. Landscape context: docs/landscape-2026-07.md.
