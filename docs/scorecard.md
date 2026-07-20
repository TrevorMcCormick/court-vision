# Court Vision scorecard — what to trust, what to check, what's broken

*For the person correcting a draft (today: Trevor; tomorrow: any MCP
charter). Numbers: **benchmark-v2** (7 matches, 491 scored points,
truth corrections of 2026-07-20 applied) — the source of truth is
[benchmark.md](benchmark.md); if this page disagrees, benchmark.md
wins and this page has a bug. Assembled by hand today; a generator
that emits these tables from the eval run is the proposed fix so they
can never drift (not yet built or planned — see data-product.md).*

## The one-line status

The system is a **rough drafter, not a charter**: about 2 in 3 points
come out within 5 small corrections of the human chart, but only ~1 in
18 is sign-off clean. Its product is the draft **plus a green flag**;
the current scorer flags about half the points at 92% held-out
reliability. Note: the draft CSVs shipped so far (the cv-17 exports)
are frozen at the older scorer's flags (99 HIGH of 508) for
experiment integrity — the 48%-coverage flags apply to exports
generated after the cv-18 experiment closes.

**Drafts exist to be corrected. Never submit an uncorrected draft to
the Match Charting Project.** The known failure mode of draft-assisted
annotation is accepting plausible-but-wrong suggestions (automation
bias) — the cv-18 experiment measures exactly this anchoring risk
before the workflow is offered to volunteers.

## The trust contract (what the flags promise, and what to DO)

| flag | measured guarantee (benchmark-v2, held-out) | your action | NOT promised |
|---|---|---|---|
| **HIGH** | 92% of flagged points within ≤5 token edits, at 48% coverage | Start from the draft; verify every token against video, especially zones and endings | NOT "the point is right" (~35% are within 2 edits) — and the promise is per-point edit count, **not per-component**: a HIGH point's serve zone is still untrustworthy |
| **LOW** | none — 56% are 6+ edits out | Chart from scratch; use the draft as a hint at most | nothing |

Per-feed honesty: on t5 (AO night) and t3 (clay RG) HIGH delivers 77%
and 80% instead of 92% — those feeds' failures are invisible to the
confidence signals. There is deliberately **no sign-off tier**: it has
failed held-out validation at every attempted n.

Correction budget & promotion rule: HIGH's tolerated miss rate is 8%
held-out. A pipeline change that blows the budget on the frozen
benchmark does not ship — coverage is sacrificed before precision
(precedent: the launch-gate repair shipped at 92%/48% only after the
alternatives measured worse; the t4 gates gave back 1.6 pts of
coverage to halve disasters).

## Component scorecard

Thresholds (declared, not vibes): 🟢 trust-with-spot-checks ≥80% ·
🟡 verify-every-time 45–80% · 🔴 re-key-always <45% · ⛔ not produced.
"Strict accuracy" = exact match vs the human chart at aligned
positions, refusals count as wrong.

| component | status | strict accuracy (benchmark-v2) | your action |
|---|---|---|---|
| Who served, which end | 🟢 84% pooled (95% best feed) | Spot-check; distrust on night feeds |
| Rally length (±1 shot) | 🟡 57–83% by match (100% on the 5-point control) | Verify — expect missing shots at night (track holes) and on clay (editor cuts) |
| Forehand/backhand letters | 🟡 ~55–67% | Verify each; wrong mostly where the player box is smeared or absent |
| Shot direction (1/2/3) | 🟡 75% when attempted (97% attempted) | Verify; the strongest annotation since the 2026-07-10 rebuild |
| Serve zone (4/5/6) | 🔴 26% strict (pooled); never above 50% committed-only on any new match | **Re-key always** — near coin-flip |
| Point ending type | 🔴 ~30% | **Re-key always** — wrong more often than right |
| Faults / second serves | ⛔ not attempted | Every draft is written as a first-serve point — restore faults yourself |
| Point boundaries & replays | 🟢 508 points/7 matches; replays & alternate angles rejected structurally | Trust; one known open case (a replayed let charts as one long point) |

## Feed support matrix

| feed / condition | tier | evidence |
|---|---|---|
| Stable wide day hard (t6 USO) | **supported** — best on record | 7.8% acceptance, 95% server end, HIGH 97% @ 55% |
| Indoor hard (t7 Turin) | **supported** — structure 🟢, annotations 🟡 | rally 83%, server end 85% |
| Grass WTA (t4 Wimbledon) | **experimental** | stance-serve early-fire poisons letters/insertions (gap #2) |
| Night hard (t1, t5) | **experimental** | late ball acquisition; serve components degrade first |
| Clay RG (t3) | **experimental** | editor cuts into rallies — a footage ceiling, not a pipeline bug |
| Anything unbenchmarked (other feeds, doubles, amateur video) | **unsupported** | never tested — no numbers, no promises |

## Gaps register

Status vocabulary: **open** (no fix known) · **diagnosed** (mechanism
named from pixels, fix designed) · **parked** (fix measured, waiting on
the cv-18 gate) · **expected** (structural — won't fix under current
approach) · **fixed** (date).

| # | gap | charter impact | status | receipt |
|---|---|---|---|---|
| 1 | Ball-track crossing recall caps rally length (night worst) | ~2 missing shots/pt — the largest edit bin | **open** — first build after cv-18 | benchmark.md failure modes |
| 2 | t4 synth-serve early-fire corrupts drafts (phantom shot 1) and shifts letter scoring | t4 letters really ~84–87%, not 55%; the phantom insertions are real draft defects | **diagnosed** 2026-07-20 — fix leads named, unmeasured | LOG.md 2026-07-20 |
| 3 | Serve zone near coin-flip (clay deuce/ad: 34/56 refuse a side) | re-key every zone | **diagnosed** — bgsub far-half cut assigned, unbuilt | benchmark.md |
| 4 | Endings ~30%; far-half in/out refuted from this camera | re-key every ending | **open** (far-half: **expected**) — bounce-first rebuild is the design | benchmark.md |
| 5 | Return-mistaken-for-serve flips server end on night feeds | 11 t5 clips | **parked** — guarded fix measured (+3 t5, no damage t6/t7) | LOG.md 2026-07-20 |
| 6 | Replayed let charts as one long point | rare, unbounded per-point cost | **open** — no fix designed; needs a signal class beyond score identity and the crossing spine | benchmark.md (t4_point_23) |
| 7 | HIGH under-delivers on t5/t3 (77%/80% vs 92%) | trust the flag less on those feeds | **open** — disease invisible to signals | benchmark.md launch-gate repair |
| 8 | Faults/second serves invisible | every fault point mis-structured | **expected** — no signal class exists today | USAGE.md |
| 9 | Changeover-parity truth bug inverted server-end answer key (4 matches) | 10 pts false "error" | **fixed** 2026-07-20 → benchmark-v2 | benchmark.md truth correction |
| 10 | Launch gate starved HIGH tier on late-acquisition feeds | HIGH coverage 20%→48% | **fixed** 2026-07-19 | benchmark.md launch-gate repair |
| 11 | Zoom drift breaks court probes (131 s lost on t1) | minutes of lost chartable play | **open** — per-block H refit designed, unbuilt | LOG.md t1 staging |

Reporting an issue: open a GitHub issue with match, point/clip ID,
the draft string, your corrected string, and the flag it carried.
Solo project — best-effort response, no SLA.

## What decides what gets built next

One measurement: **cv-18, the stopwatch experiment** — chart points
cold vs correct green-flagged drafts, timed, with an anchoring check.
If correcting beats charting, gaps get attacked in edit-cost order
(structure → letters → endings). If it doesn't, the review tool and
the confidence tier are the product and the roadmap reorders.
Protocol: [cv18-protocol.md](cv18-protocol.md). The charting app it
runs on shipped 2026-07-20; the experiment restart is queued.
