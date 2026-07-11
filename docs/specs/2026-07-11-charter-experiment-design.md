# Design: the review tool, the stopwatch, and the submission arc

Date: 2026-07-11. Status: approved (design review with Trevor).
Supersedes the cv-17 wait-for-a-charter posture: Trevor charts himself.

## Context

cv-17 published the seven draft exports and asked the MCP's 32 charters
for a stopwatch test. Decision: don't wait — Trevor becomes the
charter. That collapses the outreach dependency and unlocks the
stronger move: contribute a finished machine-assisted chart of an
un-charted match to the MCP, with measured numbers attached.

Decisions locked during design review:

1. **Endpoint:** chart & submit an un-charted match to the MCP
   (show-don't-ask). Not stopwatch-only, not notation-tourism.
2. **Tooling:** build `courtvision review` — the charter-assist MVP
   the project has claimed since cv-15. Both experiment arms run in
   it. Rejected: MCP xlsm + video player (no telemetry, builds no
   product); hybrid (arms stop being comparable).
3. **Experiment size:** 20 cold points + the full 134-row t6
   correction pass (an afternoon of charting).

## The arc (three moves, three posts)

- **cv-18 — tool + stopwatch.** Build the review tool; run the
  experiment; write up the verify-vs-transcribe ratio. Byproduct: a
  complete machine+human chart of t6 graded row-by-row against MCP.
- **cv-19 — the un-charted match.** Scout (3 candidates, Trevor
  picks), stage config-only per the cv-16 playbook, draft, full
  correction pass — strict notation, real faults.
- **cv-20 — the submission.** The chart goes to Sackmann as a
  contribution ("match #17,80x, machine-drafted, human-corrected in
  N minutes — happy to do more"), with cv-18's numbers. Supersedes
  the cv-17 outreach draft in the blog repo's `drafts/`.

Rejected sequencings: straight-to-uncharted-match (corrections would
be ungradeable — no MCP truth), experiment-in-xlsm-first (see above).

**Implementation-plan scope note:** the plan that follows this spec
covers cv-18's build only — the tool, the experiment harness, and the
analysis. Running the experiment is Trevor's afternoon, not a build
task. cv-19/cv-20 get their own cycles informed by cv-18's telemetry.

## The review tool

**Invocation.** `uv run python -m courtvision review <match>
--mode review|cold --session <name> [--port 8765]` → serves
localhost, opens the browser. Package stage behind the CLI, same
pattern as `fitcourt`/`probe`/`extract`.

**Implementation constraints.**
- One new module `courtvision/review.py` + one static HTML/JS page.
- **Stdlib `http.server` only — zero new dependencies.** The handler
  implements HTTP Range responses (~30 lines) so `<video>` seeking
  works; clips are already h264/mp4 (verified: browser-native).
- Reads: `outputs/<match>/export/<match>_mcp_draft.csv`,
  `clips/points_<match>/*.mp4`. Never reads MCP truth — grading is a
  separate post-hoc step and truth is never shown in the UI.

**Three panes.**
- *Row list* (left): all rows, confidence-colored (HIGH green, low
  dim, unplaced amber), j/k navigation, progress counter, filter by
  confidence tier.
- *Clip player* (center): auto-seek to `serve_s − 1.5s`, speed
  0.25–2×, frame-step keys, loop toggle, pause-the-clock key.
- *Edit bar* (bottom): `1st` and `2nd` string inputs, live
  MCP-legality lint, notes field, Enter-to-accept, skip-with-reason.

**Modes differ by exactly one thing.** `review` pre-fills the draft
string and shows confidence; `cold` presents blank inputs and hides
confidence. Same player, same keys, same layout — the arms differ
only by the presence of the draft.

**Legality lint.** Token-level grammar for serve direction, fault
codes, rally letters, direction digits, depth, court positions, and
ending marks per the MCP charting instructions. Unknown tokens warn;
**warnings never block accept** (escape hatch for MCP edge vocab).
The same grammar table is reused later as the strict-export validator
for the cv-20 submission.

**Cheat sheet.** Toggleable MCP notation panel (one keystroke), so
vocabulary lookups don't leave the tool (and their time stays inside
the measurement, honestly).

**Telemetry.** Append-only `events.jsonl`:
`{ts_ms, session, mode, row, event, payload}` for row_open,
play/pause/seek, edit bursts, accept, skip, clock_pause/resume.
Per-row active time = accept − open − paused. Session manifest
records mode, block id, row order, RNG seed, git SHA of the export.

**Outputs.** `outputs/<match>/review/<session>/corrected.csv`
(export schema + `corrected_1st`, `corrected_2nd`, `notes`, `flags`)
plus `events.jsonl` and `manifest.json`. Grading runs afterward via
the existing tokenizer in `courtvision/mcp.py`.

## The experiment protocol

**Warm-up (untimed, not analyzed):** 5 practice points on t3 to learn
keys and notation.

**Blocks** (each block one sitting; pause key excludes breaks):

1. **Cold-A** — 10 seeded-random t6 rows, cold mode, before anything
   else. Sampled from rows with an MCP join (unplaced rows are
   ungradeable and excluded from cold sampling).
2. **Review pass** — all 134 t6 rows, match order (real-session
   simulation), review mode. The 10 Cold-A rows still get corrected
   (byproduct chart stays complete) but are excluded from timing
   analysis — the analysis script derives the contaminated set from
   the Cold-A session manifest; no manual flagging.
3. **Cold-B** — 10 seeded-random t7 rows (same sampling rule), cold
   mode, after the review pass. Fresh material with its own MCP
   truth: separates practice effect from draft effect.

**Metrics.**
- *Primary:* minutes-per-point — cold vs HIGH-draft vs low-draft.
- Trevor's accuracy vs MCP in both arms (novice-cold error rate vs
  corrected-draft error rate).
- **Anchoring check:** on HIGH rows where the machine is wrong (~6%),
  caught or rubber-stamped? The scariest failure mode of assisted
  charting; unmeasured anywhere.
- Token-class histogram of corrections (directions, endings, rally
  length, faults) → pipeline priorities. **All pipeline work is
  deferred until this histogram exists.**
- Triage honesty: measured edits on HIGH vs low rows.

**Notation tier:** strict MCP throughout, faults included — the `2nd`
column gets real content and the submission vocabulary gets practiced.

**Analysis:** a small script over `events.jsonl` + `corrected.csv` +
MCP truth producing the cv-18 tables. Accuracy metrics skip unplaced
rows; skipped-with-reason rows leave the denominator.

## The submission arc (cv-19/20, scoped later)

- **Scouting criteria:** absent from the MCP charted-match list
  (mechanically checkable against the repo), condensed upload exists,
  stable wide feed (USO-style scored best of the five), notable
  2025–26 match. Three candidates scouted; Trevor picks.
- **Staging:** config-only per cv-16; budget one debugging session
  for the new feed's one weird thing (precedent: every feed had
  exactly one).
- **Export:** strict-legal MCP submission shape, validated by the
  review tool's grammar table.
- **The letter:** contribution + cv-18 ratio + "happy to do more."

## Risks

- New-feed staging surprise (precedent says one config knob;
  precedent also said a tiebreak is 13 games).
- MCP format rejection — worst case Jeff replies explaining what's
  wrong, which is contact established.
- Anchoring: if the experiment shows Trevor rubber-stamps wrong HIGH
  drafts, the HIGH tier's whole framing changes — that result is a
  finding, not a failure; it goes in cv-18 either way.
- Solo-charter caveat, on the record: Trevor is a novice charter, so
  cold-arm times overstate an experienced charter's; the ratio is
  still the right first number, and the submission arc doesn't
  depend on it.

## Non-goals

- No auth, localhost only; no multi-user; no state beyond files.
- No pipeline changes in cv-18 (priorities come from the histogram).
- No xlsm emulation in the tool; the submission exporter comes in
  the cv-19/20 cycle.
