# Court Vision benchmark — auto-charts vs the Match Charting Project

Four matches, three surfaces, both tours, two broadcasters beyond the
dev reel's. 168 points aligned to human-charted MCP ground truth
(alignment 108/108 on t3/t4 via score-bug point IDs). Ball tracking:
WASB (local, free, promptless). Marginal cost per match: ~$0.

Current numbers (as of 2026-07-10, commit 5797080):

| metric            | t1 night/lefty | t2 day ctrl | t3 clay RG | t4 grass WTA |
|-------------------|:---:|:---:|:---:|:---:|
| server end        | 8/11 | 2/5 | 48/59 (81%) | 38/49 (78%) |
| rally length ±1   | 10/11 | 4/5 | 25/59 | 26/49 |
| serve zone        | 2/2 | 1/1 | 9/18 | 16/27 |
| letters (aligned) | 2/2 | 11/14 | 19/24 | 16/32 |
| ending type       | 3/7 | 2/3 | 10/31 | 12/33 |

Matches: t1 = Nadal–Shapovalov, Canada Masters R16 2017 (hard, night,
both left-handed). t2 = Federer–Haase, Canada Masters SF 2017 (hard,
day). t3 = Djokovic–Ruud, Roland Garros F 2023 (clay). t4 =
Krejcikova–Paolini, Wimbledon F 2024 (grass, WTA).

## Reading the table honestly
- t1/t2 are small-n (highlights yielded few tracked points under the
  retired SAM tracker; both trees remain frozen as the A/B baseline).
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
- Long-rally late-track blindness vs true codas on grass (3 clips).
- Far-half in/out for endings: refuted with receipt (airborne-ball
  projection error); near-half bounces recoverable only in a 1.2 s
  fill window.
- t2 server end (2/5): thin serve detection on the control's few
  clips; superseded approach-wise by serve v3 but tree stays frozen.
- Deuce/ad stance refusal on clay (34/56 clips refuse a side).

Full history: LOG.md. Landscape context: docs/landscape-2026-07.md.
