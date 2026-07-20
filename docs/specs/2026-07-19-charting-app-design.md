# Design: the charting app — Court Vision's ground-truth factory

Date: 2026-07-19. Status: approved (design review with Trevor).
Supersedes the review bench as the charting surface; the bench stays
frozen in-repo as the cv-18 practice-era artifact.

## Purpose

A web app where anyone — Trevor first — charts a tennis match in MCP
notation without Excel, fast and *accurately*. The app's real product
is training data: (clip, chart) pairs accurate enough to benchmark
and train the pipeline against. Priority order everywhere: accuracy,
then charting speed, then everything else. Historical MCP data
remains the other training source; this app manufactures new ground
truth with aligned video timestamps the MCP itself doesn't have.

Origin pain (observed): watch a shot → pause → swipe to the Excel
instructions → swipe back → type → play. 6/134 review rows in a week
was the verdict on the bench. The cheat sheet must BE the input
surface, not a thing consulted.

## Decisions locked during design review

1. **Audience:** Trevor-first, public-ready core. Built as the Court
   Vision charting surface (clips + drafts first-class), but the core
   page is serverless-static by construction — publishing later at
   trmccormick.com/chart is a copy, not a rewrite.
2. **Entry UX:** keys + live palette. The palette bar always shows
   the legal moves *with meanings*; every option is both a keyboard
   key and a clickable button. Chips build the string; backspace
   pops; raw-text escape hatch for exotic vocabulary.
3. **Approach:** new page, shared bones. A fresh `chart` app; the
   review bench is not evolved (frozen control-era artifact). The
   new app absorbs review-mode as a "load machine draft" power-up.
4. **Sequencing:** MVP first; segmentation is a byproduct of
   charting (accepted points stamp their own video boundaries), not
   a prerequisite. Pipeline pre-cut clips and machine drafts are
   power-ups when a match is staged, never requirements.

## Shape and modes

One single-page app, two data flavors behind one adapter interface:

- **Local flavor** (default): served by the courtvision CLI.
  - `courtvision chart <match>` — staged match: pre-cut clips,
    machine drafts loadable per point (review-mode), sessions +
    events.jsonl telemetry via the existing session API.
  - `courtvision chart --video <path> --new <id>` — unstaged match:
    one continuous video, chart-along.
- **Static flavor** (public-ready core): same page, zero server.
  Drag a video file in; localStorage autosave; exports are file
  downloads. **Core-loop rule: no feature in the charting loop may
  depend on the server.** Server-only features (drafts, pre-cut
  clips, server telemetry) must degrade away cleanly.

**Boundaries as byproduct (chart-along):** a point's `start_s` is
stamped when its serve entry begins (or on an explicit mark key);
`end_s` at accept. The segments that fall out match the extract
stage's shape, so a human-charted match becomes machine-usable with
zero alignment work.

## The palette engine

A grammar state machine drives entry: **serve → (fault?) → rally
shots → ending**, plus second-serve routing. The palette renders the
current state's legal moves with labels:

- Serve state: `4 wide · 5 body · 6 T · 0 unknown`, fault key.
- Fault entered: fault-type letters (n/w/d/x/g/!/e); the entry lands
  in `1st`; palette returns to serve state for the 2nd.
- Rally state: shot letters with names (f/b drive, r/s slice, v/z
  volley, o/p overhead, u/y drop, l/m lob, h/i half-volley, j/k
  swinging volley, t trick, q unknown), then direction `1/2/3`
  (skippable = unknown), optional depth `7/8/9`.
- Ending state: `* winner`, error letters then `@ unforced · # forced`,
  `? unknown ending` (clip cut out — same token the pipeline uses).
- Chips show the string as built; backspace pops one token; a raw
  text field stays in sync for hand edits and exotic marks (lets,
  time violations — full MCP vocabulary always reachable).

**Grammar single-source:** the vocabulary and state transitions live
in one checked-in `grammar.json`. The palette renders from it,
`notation.py`'s lint loads its vocabulary tables from it, and the
static copy gets it baked in. One source, three consumers — this
kills the mcp.py/notation.py drift class the cv-18 reviews flagged.
(`mcp.py` itself stays frozen: it is the benchmark metric, not a
grammar consumer.)

## The score engine

Match setup: players, best-of, tiebreak rules, first server. Then
the app owns the scoreboard:

- **Winner derivation** from each accepted string (ending mark +
  shot-count parity — deterministic in MCP notation); games, sets,
  tiebreaks, and serve rotation advance automatically. Score columns
  in the export are machine-perfect. Strings ending `?` (unknown
  ending — clip cut out) can't derive a winner: the app prompts a
  one-key "who won?" at accept so the scoreboard never stalls.
- **The mismatch nudge:** at every game boundary, a one-glance
  prompt — "chart says 2–1, 30–0 — does the screen agree?" Confirm
  or flag; disagreement is caught within a game, not at export.
- **The missing-point flow:** condensed videos drop points. When the
  on-screen score jumps past the chart, one keystroke inserts an
  "unseen point" row — winner inferred from the score delta, string
  blank, flagged `unseen` — so the scoreboard stays true and the
  chart stays honest.

## Persistence & telemetry

Adapter interface with two implementations:

- **Server adapter** (local flavor): the existing session API.
  Event vocabulary grows: `palette_key`, `chip_pop`, `nudge_shown`,
  `nudge_ok`, `nudge_mismatch`, `point_mark`, `unseen_insert` join
  the current set — the stopwatch experiment gets richer
  instrumentation, not less.
- **Browser adapter** (static flavor): localStorage autosave per
  match id; explicit export downloads. No telemetry upload.

## Export: the training bundle

Three files per match:

1. **points CSV** — MCP points-file columns (xlsm-paste compatible),
   score columns from the score engine.
2. **segments CSV** — per-point `clip/start_s/end_s` in the extract
   stage's shape.
3. **manifest JSON** — match meta, format rules, video filename +
   size/hash, app version, grammar.json version.

Together: an import-ready new benchmark match. (An actual
`courtvision import` command is post-MVP; the bundle format is
designed now so nothing is thrown away.)

## Relationship to cv-18

Everything charted before the app exists — warm-up, cv18-cold-a
(voided practice), cv18-cold-a2, the 6 cv18-review rows — becomes
labeled practice. The experiment restarts on the app with fresh
seeds (all arms, one tool); `review-analyze --contaminated` unions
every practice session's rows out of review timing. The frozen bench
and its sessions remain in-repo as the control-era artifact. The
protocol runbook gets amended when the restart is scheduled.

## Non-goals (MVP discipline)

No YouTube ingestion (local files only), no accounts or multi-user,
no mobile-first layout, no court-coordinate capture, no in-browser
auto-segmentation, no double-charter verification workflow, no .xlsm
binary writing (CSV + clipboard block), no `courtvision import`
command yet.

## Risks / notes

- Grammar completeness: the guided path covers the scored core +
  common vocabulary; the raw field is the pressure valve. Lint stays
  advisory everywhere.
- Static parity is a discipline, not a feature — the plan should
  include a "static smoke" step (open the page from file/plain
  http.server, chart a point, export) to keep the core honest.
- Prior art: on-the-t's Match-Charting-GUI (2016, Shiny) — one look
  during planning for UX lessons, not code.
- Winner-derivation edge cases (retirements, penalties mid-game) are
  out of scope for MVP; the unseen-point flow plus raw notes cover
  the rare cases.

## Implementation-plan scope

The plan following this spec covers: grammar.json + palette engine,
the chart page (both flavors), score engine, session-API extensions,
training-bundle export, CLI wiring (`courtvision chart`), tests, and
the static smoke. The experiment restart, `courtvision import`, and
publishing to trmccormick.com are separate later cycles.
