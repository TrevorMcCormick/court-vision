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
        --session cv18-cold-a --seed cv18-a --n 10

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
        --cold-a t6:cv18-cold-a --review t6:cv18-review \
        --cold-b t7:cv18-cold-b

Output: outputs/t6/review/cv18-review/analysis.md — the cv-18 tables.

## Notation

Strict MCP throughout, faults included: if the point was played on a
second serve, the fault goes in 1st (e.g. "4d") and the point string
in 2nd. The in-tool cheat sheet (c) has the vocabulary; warnings are
advisory and never block accept.
