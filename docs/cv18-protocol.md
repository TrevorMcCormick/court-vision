# cv-18 stopwatch protocol — the exact session commands

Seeds are frozen here so the row draw is decided before the first
point is charted. Every block is one sitting; use `p` (pause clock)
for any break. If a clip is unchartable, `x` skips it with a reason —
skips leave the denominator.

## 0. Warm-up (untimed, not analyzed, ~5 points)

    uv run python -m courtvision review t3 --mode cold \
        --session cv18-warmup --seed warmup --n 5

Learn the keys: j/k rows, space play, arrows frame-step, [ ] speed,
l loop, 1/2/n fields, Enter accept, x skip, c cheat sheet, p pause.
Chart all 5. Nothing here is analyzed.

## 1. Cold-A — 10 t6 points, drafts hidden

    uv run python -m courtvision review t6 --mode cold \
        --session cv18-cold-a2 --seed cv18-a2 --n 10

Protocol amendment, on the record: the first Cold-A attempt
(session cv18-cold-a, seed cv18-a, 8.5 min wall) was charted without
direction digits — sub-MCP-grade work, so its timing would flatter
the cold arm and its accuracy is incomparable. It is kept as a
labeled practice block, voided for the arms. The redo uses seed
cv18-a2 (verified disjoint draw). Both sessions' rows are excluded
from review-pass timing via --contaminated in step 4.

## 2. Review pass — all 134 t6 rows, drafts shown

    uv run python -m courtvision review t6 --mode review \
        --session cv18-review

Match order, top to bottom. The 10 Cold-A rows reappear here — chart
them anyway (the byproduct chart must be complete); the analysis
excludes them from review timing automatically via the Cold-A
manifest. This block is the long one; split sittings are fine BETWEEN
rows (Ctrl-C and rerun the same command — sessions resume).

## 3. Cold-B — 10 t7 points, drafts hidden

    uv run python -m courtvision review t7 --mode cold \
        --session cv18-cold-b --seed cv18-b --n 10

## 4. Analysis

    uv run python -m courtvision review-analyze \
        --cold-a t6:cv18-cold-a2 --review t6:cv18-review \
        --cold-b t7:cv18-cold-b --contaminated t6:cv18-cold-a

Output: outputs/t6/review/cv18-review/analysis.md — the cv-18 tables.

## Notation

Strict MCP throughout, faults included: if the point was played on a
second serve, the fault goes in 1st (e.g. "4d") and the point string
in 2nd. Directions (1/2/3) are scored — write one per shot when you
saw where the ball went; omit when you couldn't tell (scores as
unknown, same cost as wrong). Depth digits (7/8/9) are charting
detail the eval drops — include only if effortless. The in-tool cheat sheet (c) has the vocabulary; warnings are
advisory and never block accept.

## Charting rules for imperfect clips

The condensed-match editors cut into points (documented since t3/t4:
the clay and grass editors both do it; two confidence gates exist
because of it). The machine drafts from the same clips, so these
rules keep the arms comparable — chart what the video supports,
never what it implies:

1. Scrub the whole clip before concluding anything — in cold mode
   the serve is often a few seconds in (space, arrows, ] speed).
2. Serve visible anywhere in the clip → chart normally from it.
3. Clip joins mid-rally (no serve shown) → serve digit `0`
   (unknown), chart ONLY the shots you can see, ending as seen,
   and note "joins mid-rally" (press n). Never guess hidden shots
   or faults.
4. Clip cuts out before the point ends (ball still live at the
   final frame) → chart the shots you saw and end the string with
   `?` (unknown ending), note "cut before ending". Never infer the
   ending from the score bug — who won is not how it ended. The
   lint will warn on `?`; that is expected and advisory.
5. Replay / second showing of the same point → chart the first
   full live showing, ignore the repeat.
6. `x` (skip + reason) only when there is no chartable tennis at
   all — pure cutaway, wrong point, broken video.
