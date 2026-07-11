# cv-18 Review Tool + Stopwatch Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `courtvision review` — a stdlib-only local web app for correcting draft charts against clips with full telemetry — plus the post-hoc analysis that produces the cv-18 verify-vs-transcribe tables.

**Architecture:** Three new package modules behind the existing argparse CLI: `notation.py` (MCP-legality lint, pure functions), `review.py` (session model + `http.server` app serving one static HTML page), `review_analysis.py` (timing/accuracy/anchoring/histogram over session outputs). The UI is one self-contained HTML/JS file. Grading reuses `courtvision/mcp.py` verbatim.

**Tech Stack:** Python ≥3.12 stdlib only at runtime (`http.server`, `json`, `csv`, `random`, `subprocess`, `webbrowser`). `pytest` as the only new dev dependency. Vanilla HTML/JS/CSS, no frameworks, no CDN.

**Spec:** `docs/specs/2026-07-11-charter-experiment-design.md` (approved 2026-07-11).

## Global Constraints

- Runtime dependencies: **stdlib only** — `pyproject.toml` `[project] dependencies` must not change. `pytest` goes in a `[dependency-groups] dev` table.
- `courtvision/review.py` **never reads MCP truth** (`eval.mcp_map` / `eval.mcp_points`). Only `review_analysis.py` may.
- Lint **warnings never block accept** — there is no "error" level.
- Cold mode strips ALL machine-derived fields server-side: `first_draft`, `confidence`, `conf_p`, `serve_s`, `n_shots`. (Score-state columns stay — a cold charter sees the score bug on screen anyway.)
- Cold sampling draws only rows with a non-blank `Pt` (gradeable rows), returns them in match order, seeded and deterministic.
- Sessions are resumable: re-running the CLI with an existing session name reloads `corrected.csv` and keeps appending to `events.jsonl`. Mode/match mismatch on resume is an error.
- Timing rule: inter-event gaps > 180 s are excluded from active time and reported separately as idle holes.
- Match existing code style: narrative module docstrings, ~79-col lines, no type annotations (the package has none), `csv.DictReader`/`DictWriter`, f-strings.
- Repo has no CI; "run the tests" means `uv run pytest -q` locally. Commit after every task.

## File Structure

```text
courtvision/notation.py         MCP-legality lint (new)
courtvision/review.py           session model + HTTP app (new)
courtvision/review_ui.html      the single-page UI (new)
courtvision/review_analysis.py  post-hoc analysis -> cv-18 tables (new)
courtvision/cli.py              +2 subcommands (modify)
pyproject.toml                  +dev dependency group (modify)
tests/conftest.py               tmp MatchConfig factory (new)
tests/test_notation.py          (new)
tests/test_review_session.py    (new)
tests/test_review_server.py     (new)
tests/test_review_analysis.py   (new)
docs/cv18-protocol.md           the runbook Trevor follows (new)
docs/USAGE.md                   +review section (modify)
```

Data layout produced at runtime (consumed by Task 6):

```text
outputs/<match>/review/<session>/
  manifest.json     {session, match, mode, rows, seed, n, created_ts_ms,
                     export_file, git_sha}
  events.jsonl      one JSON object per line (see Task 2 event schema)
  corrected.csv     clip, corrected_1st, corrected_2nd, notes, flags,
                    accepted_ts_ms
```

---

### Task 1: pytest scaffold + `notation.py` (MCP-legality lint)

**Files:**
- Modify: `pyproject.toml`
- Create: `courtvision/notation.py`
- Create: `tests/test_notation.py`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces: `lint(first, second) -> list[dict]`, each issue
  `{"field": "1st"|"2nd", "pos": int|None, "msg": str}`. Also module
  constants `SERVE_DIGITS`, `SHOT_LETTERS`, `DIRECTIONS`, `DEPTHS`,
  `ERROR_LETTERS`, `ENDING_MARKS`, `OTHER_MARKS` (Tasks 3/4 call
  `lint` via the `/api/lint` endpoint; the cv-19/20 strict exporter
  will import the same constants).

- [ ] **Step 1: Add the dev dependency group**

Run: `cd /Users/trevor.mccormick/Documents/court-vision && uv add --dev pytest`
Expected: `pyproject.toml` gains a `[dependency-groups] dev = ["pytest>=..."]` table; `uv.lock` updates; `[project] dependencies` unchanged (verify with `git diff pyproject.toml`).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_notation.py`:

```python
"""Lint is advisory: it returns issues, it never raises, and an empty
list means 'nothing to warn about'. Vocabulary follows the MCP
charting instructions; anything outside it warns with a position so
the UI can point at the character."""

from courtvision.notation import lint


def msgs(issues):
    return [i["msg"] for i in issues]


def test_legal_rally_string_is_clean():
    # serve T, bh crosscourt deep, fh down the line, fh winner
    assert lint("6b29f1f3*", "") == []


def test_legal_fault_then_second_serve_point():
    # 1st = wide fault; 2nd = body serve, bh into net, unforced
    assert lint("4w", "5b2n@") == []


def test_unknown_char_warns_with_position():
    issues = lint("6b2Z*", "")
    assert len(issues) == 1
    assert issues[0]["field"] == "1st"
    assert issues[0]["pos"] == 3
    assert "unknown" in issues[0]["msg"]


def test_orphan_digit_warns():
    # direction digit with no shot letter before it
    issues = lint("62b1*", "")
    assert any("digit" in m for m in msgs(issues))


def test_missing_serve_digit_warns():
    issues = lint("b2f1*", "")
    assert any("serve digit" in m for m in msgs(issues))


def test_no_ending_mark_warns():
    issues = lint("6b2f1", "")
    assert any("ending" in m for m in msgs(issues))


def test_second_filled_but_first_not_a_fault_warns():
    # if the point was played on the 2nd, the 1st should be a fault
    issues = lint("6b2f1*", "5b2n@")
    assert any("fault" in m for m in msgs(issues))


def test_empty_first_warns():
    issues = lint("", "")
    assert any("empty" in m for m in msgs(issues))


def test_error_ending_needs_error_letter():
    # '@' with no n/w/d/x/!/e before it
    issues = lint("6b2f1@", "")
    assert any("error letter" in m for m in msgs(issues))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_notation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.notation'`

- [ ] **Step 4: Write `courtvision/notation.py`**

```python
"""MCP-notation legality lint — advisory, never blocking.

The review tool lints what the charter types; the same vocabulary
tables later validate the cv-19/20 strict export. Sources: the Match
Charting Project charting instructions (MatchChart 0.3.2). The lint
is deliberately humble: everything it doesn't recognize is a WARNING
with a position, never a rejection — MCP has edge vocabulary (lets,
time violations, challenges) that a charter may legitimately use.

    from courtvision.notation import lint
    issues = lint(first, second)   # [{"field","pos","msg"}, ...]

An empty list means nothing to warn about. `pos` is a 0-based index
into the offending field's string, or None for whole-string issues.
"""

SERVE_DIGITS = set("0456")           # 4 wide, 5 body, 6 T; 0 unknown
SHOT_LETTERS = set("fbrsvzopuylmhijktq")
DIRECTIONS = set("0123")             # 1 rh-fh side, 2 middle, 3 rh-bh
DEPTHS = set("789")                  # 7 short, 8 mid, 9 deep
ERROR_LETTERS = set("nwdx!e")        # net, wide, deep, both, shank, unk
ENDING_MARKS = set("*@#")            # winner, unforced, forced
FAULT_LETTERS = set("nwdxg!e")       # serve faults incl. foot fault
OTHER_MARKS = set("+-=;^c")          # approach, net pos, stop, let...


def _lint_field(s, field, is_fault_string):
    issues = []
    if not s:
        issues.append({"field": field, "pos": None,
                       "msg": f"{field} is empty"})
        return issues
    if s[0] not in SERVE_DIGITS:
        issues.append({"field": field, "pos": 0,
                       "msg": "no serve digit (0/4/5/6) at start"})
    known = (SERVE_DIGITS | SHOT_LETTERS | DIRECTIONS | DEPTHS |
             ERROR_LETTERS | ENDING_MARKS | OTHER_MARKS)
    last_shot_idx = None
    for i, c in enumerate(s):
        if c not in known:
            issues.append({"field": field, "pos": i,
                           "msg": f"unknown mark '{c}'"})
        elif c in SHOT_LETTERS:
            last_shot_idx = i
        elif c in (DIRECTIONS | DEPTHS) and i > 0:
            # a direction/depth digit must trail a shot (or the serve)
            if last_shot_idx is None and i != 1:
                issues.append({"field": field, "pos": i,
                               "msg": f"digit '{c}' follows no shot"})
    if is_fault_string:
        if not any(c in FAULT_LETTERS for c in s[1:]):
            issues.append({"field": field, "pos": None,
                           "msg": "2nd is filled, so 1st should be a "
                                  "fault (needs a fault letter)"})
        return issues
    if s[-1] not in ENDING_MARKS:
        issues.append({"field": field, "pos": len(s) - 1,
                       "msg": "no ending mark (* winner, @ unforced, "
                              "# forced)"})
    elif s[-1] in "@#":
        body = s[:-1]
        if not (body and body[-1] in ERROR_LETTERS):
            issues.append({"field": field, "pos": len(s) - 1,
                           "msg": "error ending without an error letter "
                                  "(n/w/d/x/!/e) before it"})
    return issues


def lint(first, second):
    """Lint an MCP 1st/2nd pair. Returns a list of warning dicts."""
    first, second = (first or "").strip(), (second or "").strip()
    issues = _lint_field(first, "1st", is_fault_string=bool(second))
    if second:
        issues += _lint_field(second, "2nd", is_fault_string=False)
    return issues
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_notation.py -q`
Expected: `9 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock courtvision/notation.py tests/test_notation.py
git commit -m "Review tool: MCP-legality lint (advisory, never blocking) + pytest scaffold"
```

---

### Task 2: `review.py` session model (rows, sampling, manifest, events, corrections)

**Files:**
- Create: `courtvision/review.py` (session half; the HTTP half lands in Task 3)
- Create: `tests/conftest.py`
- Create: `tests/test_review_session.py`

**Interfaces:**
- Consumes: `config.MatchConfig` (only `.id`, `.out_dir`, `.clips_dir`); export CSV at `cfg.out_dir / "export" / f"{cfg.id}_mcp_draft.csv"` with columns `match_id,Pt,Set1,Set2,Gm1,Gm2,Pts,Svr,1st,2nd,confidence,conf_p,clip,serve_s,n_shots`.
- Produces (Tasks 3/5/6 rely on these exact names):
  - `load_export_rows(cfg) -> list[dict]` — export rows, file order.
  - `sample_cold_rows(rows, n, seed) -> list[str]` — clip stems.
  - `class ReviewSession` with: `ReviewSession(cfg, mode, name, seed=None, n=None)` (creates or resumes), attributes `.dir`, `.mode`, `.name`, `.rows` (list of export-row dicts scoped to the session), methods `.state() -> dict`, `.append_event(evt: dict) -> None`, `.accept(clip, first, second, notes, skip_reason="") -> None`.
  - Event line schema: client fields `ts_ms:int, row:str, event:str, payload:dict` plus injected `session:str, mode:str, server_ts_ms:int`. Event names used by the UI: `row_open, play, pause, seek, rate, edit, accept, skip, clock_pause, clock_resume, cheat`.
  - `state()` returns `{"session","match","mode","rows":[...]}` where each row has `clip, pt, pts, svr, set1, set2, gm1, gm2, first_draft, confidence, conf_p, serve_s, n_shots, done, skipped, corrected_1st, corrected_2nd, notes`; in cold mode `first_draft` is `""` and `confidence, conf_p, serve_s, n_shots` are `None`.

- [ ] **Step 1: Write the shared test fixture**

Create `tests/conftest.py`:

```python
"""A minimal on-disk match for review/session tests: a MatchConfig
pointing at tmp dirs, a 4-row export CSV (one blank-Pt row), and two
tiny fake 'clips' (bytes on disk are enough for Range tests)."""

import csv

import pytest

from courtvision.config import MatchConfig, Staging, EvalCfg

EXPORT_FIELDS = ["match_id", "Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts",
                 "Svr", "1st", "2nd", "confidence", "conf_p", "clip",
                 "serve_s", "n_shots"]

ROWS = [
    {"match_id": "m", "Pt": "2", "Set1": "0", "Set2": "0", "Gm1": "0",
     "Gm2": "0", "Pts": "0-15", "Svr": "1", "1st": "s4b3f1w@", "2nd": "",
     "confidence": "high", "conf_p": "0.966", "clip": "tt_point_01",
     "serve_s": "3.2", "n_shots": "3"},
    {"match_id": "m", "Pt": "3", "Set1": "0", "Set2": "0", "Gm1": "0",
     "Gm2": "0", "Pts": "0-30", "Svr": "1", "1st": "s?f2?3?", "2nd": "",
     "confidence": "low", "conf_p": "0.51", "clip": "tt_point_02",
     "serve_s": "3.0", "n_shots": "3"},
    {"match_id": "m", "Pt": "", "Set1": "0", "Set2": "0", "Gm1": "1",
     "Gm2": "0", "Pts": "0-0", "Svr": "", "1st": "s6*", "2nd": "",
     "confidence": "low", "conf_p": "0.40", "clip": "tt_point_03",
     "serve_s": "0.8", "n_shots": "1"},
    {"match_id": "m", "Pt": "5", "Set1": "0", "Set2": "0", "Gm1": "1",
     "Gm2": "0", "Pts": "15-0", "Svr": "2", "1st": "s5b2f1*", "2nd": "",
     "confidence": "high", "conf_p": "0.91", "clip": "tt_point_04",
     "serve_s": "1.5", "n_shots": "3"},
]


@pytest.fixture
def cfg(tmp_path):
    clips = tmp_path / "clips"
    out = tmp_path / "out"
    (out / "export").mkdir(parents=True)
    clips.mkdir()
    for r in ROWS:
        (clips / f"{r['clip']}.mp4").write_bytes(
            bytes(range(256)) * 4)                 # 1024 known bytes
    with open(out / "export" / "tt_mcp_draft.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
        w.writeheader()
        w.writerows(ROWS)
    return MatchConfig(
        id="tt", title="tt", match="", clips_dir=clips, out_dir=out,
        ball_dir=out / "ball", players_dir=out / "players",
        charts_dir=out / "charts", homography=out / "H.npy",
        serves=out / "serves.csv", clip_offsets=None,
        lefty={"near": False, "far": False}, staging=Staging(),
        eval=EvalCfg(), serve_detect={}, players_detect={},
        video=None, court_detect={},
    )
```

- [ ] **Step 2: Write the failing session tests**

Create `tests/test_review_session.py`:

```python
import json

from courtvision.review import (load_export_rows, sample_cold_rows,
                                ReviewSession)


def test_load_export_rows_preserves_order(cfg):
    rows = load_export_rows(cfg)
    assert [r["clip"] for r in rows] == [
        "tt_point_01", "tt_point_02", "tt_point_03", "tt_point_04"]


def test_sample_excludes_blank_pt_and_is_deterministic(cfg):
    rows = load_export_rows(cfg)
    picked = sample_cold_rows(rows, 2, "cv18-a")
    assert picked == sample_cold_rows(rows, 2, "cv18-a")
    assert "tt_point_03" not in picked          # blank Pt: ungradeable
    # match order preserved
    idx = {r["clip"]: i for i, r in enumerate(rows)}
    assert picked == sorted(picked, key=idx.get)


def test_create_writes_manifest_and_resume_roundtrips(cfg):
    s = ReviewSession(cfg, "cold", "block-a", seed="cv18-a", n=2)
    man = json.loads((s.dir / "manifest.json").read_text())
    assert man["mode"] == "cold" and man["match"] == "tt"
    assert len(man["rows"]) == 2
    s.accept("tt_point_01", "4b3f1w@", "", "shaky serve read")
    s2 = ReviewSession(cfg, "cold", "block-a", seed="cv18-a", n=2)
    row = next(r for r in s2.state()["rows"]
               if r["clip"] == "tt_point_01")
    assert row["done"] and row["corrected_1st"] == "4b3f1w@"


def test_resume_mode_mismatch_raises(cfg):
    ReviewSession(cfg, "cold", "block-x", seed="s", n=2)
    try:
        ReviewSession(cfg, "review", "block-x")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cold_state_strips_machine_fields(cfg):
    s = ReviewSession(cfg, "cold", "block-b", seed="cv18-a", n=2)
    for r in s.state()["rows"]:
        assert r["first_draft"] == ""
        assert r["confidence"] is None and r["conf_p"] is None
        assert r["serve_s"] is None and r["n_shots"] is None


def test_review_state_keeps_draft_and_all_rows(cfg):
    s = ReviewSession(cfg, "review", "full")
    rows = s.state()["rows"]
    assert len(rows) == 4
    assert rows[0]["first_draft"] == "s4b3f1w@"
    assert rows[0]["confidence"] == "high"


def test_events_append_and_inject_context(cfg):
    s = ReviewSession(cfg, "review", "full")
    s.append_event({"ts_ms": 5, "row": "tt_point_01",
                    "event": "row_open", "payload": {}})
    s.append_event({"ts_ms": 9, "row": "tt_point_01",
                    "event": "accept", "payload": {}})
    lines = [json.loads(l) for l in
             (s.dir / "events.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["row_open", "accept"]
    assert lines[0]["session"] == "full" and lines[0]["mode"] == "review"
    assert "server_ts_ms" in lines[0]


def test_skip_records_flag(cfg):
    s = ReviewSession(cfg, "review", "full")
    s.accept("tt_point_02", "", "", "", skip_reason="broken clip")
    row = next(r for r in s.state()["rows"]
               if r["clip"] == "tt_point_02")
    assert row["skipped"] == "broken clip"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_review_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'courtvision.review'`

- [ ] **Step 4: Write the session half of `courtvision/review.py`**

```python
"""The charter's bench — review/correct draft charts against clips.

Session model (this half) + a stdlib http.server app (bottom half)
serving review_ui.html. Two modes that differ by exactly one thing:
'review' pre-fills the machine draft and shows confidence; 'cold'
presents blank inputs and hides everything machine-derived. Same
player, same keys — the stopwatch experiment's arms differ only by
the presence of the draft.

This module NEVER reads MCP truth. Grading is review_analysis.py's
job, after the session, so the truth can't leak into the UI.

Outputs, under outputs/<match>/review/<session>/:
  manifest.json   mode, row set, seed, export sha, git sha
  events.jsonl    every UI action, timestamped client+server side
  corrected.csv   the charter's strings, one row per accepted clip

Usage:
    uv run python -m courtvision review t6 --mode review --session r1
"""

import csv
import hashlib
import json
import random
import subprocess
import time

FIELDS = ["clip", "corrected_1st", "corrected_2nd", "notes", "flags",
          "accepted_ts_ms"]
COLD_STRIP = ("confidence", "conf_p", "serve_s", "n_shots")


def _export_path(cfg):
    return cfg.out_dir / "export" / f"{cfg.id}_mcp_draft.csv"


def load_export_rows(cfg):
    """Export rows in file order (the export is already Pt-sorted)."""
    with open(_export_path(cfg)) as f:
        return list(csv.DictReader(f))


def sample_cold_rows(rows, n, seed):
    """n gradeable clips (non-blank Pt), seeded, in match order."""
    idx = {r["clip"]: i for i, r in enumerate(rows)}
    pool = [r["clip"] for r in rows if r["Pt"].strip()]
    picked = random.Random(seed).sample(pool, n)
    return sorted(picked, key=idx.get)


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


class ReviewSession:
    """Create-or-resume a named session for one match and mode."""

    def __init__(self, cfg, mode, name, seed=None, n=None):
        assert mode in ("review", "cold"), mode
        self.cfg, self.mode, self.name = cfg, mode, name
        self.dir = cfg.out_dir / "review" / name
        all_rows = load_export_rows(cfg)
        by_clip = {r["clip"]: r for r in all_rows}
        manifest_p = self.dir / "manifest.json"
        if manifest_p.exists():
            man = json.loads(manifest_p.read_text())
            if man["mode"] != mode or man["match"] != cfg.id:
                raise ValueError(
                    f"session '{name}' is {man['match']}/{man['mode']}, "
                    f"asked for {cfg.id}/{mode}")
            row_ids = man["rows"]
        else:
            if mode == "cold":
                row_ids = sample_cold_rows(all_rows, n, seed)
            else:
                row_ids = [r["clip"] for r in all_rows]
            self.dir.mkdir(parents=True, exist_ok=True)
            export = _export_path(cfg)
            man = {"session": name, "match": cfg.id, "mode": mode,
                   "rows": row_ids, "seed": seed, "n": n,
                   "created_ts_ms": int(time.time() * 1000),
                   "export_file": str(export),
                   "export_sha256": hashlib.sha256(
                       export.read_bytes()).hexdigest(),
                   "git_sha": _git_sha()}
            manifest_p.write_text(json.dumps(man, indent=2))
        self.manifest = man
        self.rows = [by_clip[c] for c in row_ids]
        self._corrections = self._load_corrections()

    # -- corrections ---------------------------------------------------

    def _corrected_path(self):
        return self.dir / "corrected.csv"

    def _load_corrections(self):
        p = self._corrected_path()
        if not p.exists():
            return {}
        with open(p) as f:
            return {r["clip"]: r for r in csv.DictReader(f)}

    def accept(self, clip, first, second, notes, skip_reason=""):
        flags = f"skipped:{skip_reason}" if skip_reason else ""
        self._corrections[clip] = {
            "clip": clip, "corrected_1st": first,
            "corrected_2nd": second, "notes": notes, "flags": flags,
            "accepted_ts_ms": str(int(time.time() * 1000))}
        with open(self._corrected_path(), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for c in (r["clip"] for r in self.rows):
                if c in self._corrections:
                    w.writerow(self._corrections[c])

    # -- events --------------------------------------------------------

    def append_event(self, evt):
        evt = dict(evt)
        evt["session"], evt["mode"] = self.name, self.mode
        evt["server_ts_ms"] = int(time.time() * 1000)
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(evt) + "\n")

    # -- state for the UI ----------------------------------------------

    def state(self):
        out = []
        for r in self.rows:
            corr = self._corrections.get(r["clip"], {})
            flags = corr.get("flags", "")
            row = {"clip": r["clip"], "pt": r["Pt"], "pts": r["Pts"],
                   "svr": r["Svr"], "set1": r["Set1"], "set2": r["Set2"],
                   "gm1": r["Gm1"], "gm2": r["Gm2"],
                   "first_draft": r["1st"],
                   "confidence": r["confidence"],
                   "conf_p": r["conf_p"], "serve_s": r["serve_s"],
                   "n_shots": r["n_shots"],
                   "done": bool(corr),
                   "skipped": (flags.split(":", 1)[1]
                               if flags.startswith("skipped:") else ""),
                   "corrected_1st": corr.get("corrected_1st", ""),
                   "corrected_2nd": corr.get("corrected_2nd", ""),
                   "notes": corr.get("notes", "")}
            if self.mode == "cold":
                row["first_draft"] = ""
                for k in COLD_STRIP:
                    row[k] = None
            out.append(row)
        return {"session": self.name, "match": self.cfg.id,
                "mode": self.mode, "rows": out}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_review_session.py -q`
Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
git add courtvision/review.py tests/conftest.py tests/test_review_session.py
git commit -m "Review tool: session model — sampling, manifest, events, corrections, resume"
```

---

### Task 3: `review.py` HTTP server (routes + Range video + lint endpoint)

**Files:**
- Modify: `courtvision/review.py` (append the HTTP half)
- Create: `tests/test_review_server.py`

**Interfaces:**
- Consumes: `ReviewSession` (Task 2), `notation.lint` (Task 1), `courtvision/review_ui.html` (Task 4 — until then `GET /` may 404 in dev; the test writes a stub).
- Produces (the UI in Task 4 calls exactly these):
  - `GET /` → `review_ui.html`, `text/html`
  - `GET /api/state` → `session.state()` JSON
  - `GET /clip/<stem>.mp4` → video bytes; honors `Range: bytes=a-b` with 206 + `Content-Range`; 200 full file otherwise
  - `POST /api/lint` `{"first","second"}` → `{"issues":[...]}`
  - `POST /api/event` `{ts_ms,row,event,payload}` → `{"ok":true}`
  - `POST /api/accept` `{clip,corrected_1st,corrected_2nd,notes,skip_reason}` → `{"ok":true}` (also logs an `accept`/`skip` event server-side)
  - `make_server(session, port) -> ThreadingHTTPServer` and `run(cfg, mode, name, seed, n, port, open_browser) -> None`

- [ ] **Step 1: Write the failing server tests**

Create `tests/test_review_server.py`:

```python
import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import courtvision.review as review
from courtvision.review import ReviewSession, make_server


def _client(cfg, mode="review", name="srv"):
    s = ReviewSession(cfg, mode, name, seed="s", n=2)
    httpd = make_server(s, 0)                    # port 0 = ephemeral
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return s, httpd, HTTPConnection("127.0.0.1", httpd.server_port)


def _post(conn, path, body):
    conn.request("POST", path, json.dumps(body),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def test_root_serves_ui(cfg, monkeypatch, tmp_path):
    stub = tmp_path / "ui.html"
    stub.write_text("<!-- cv-review -->")
    monkeypatch.setattr(review, "UI_PATH", stub)
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/")
    r = conn.getresponse()
    assert r.status == 200 and b"cv-review" in r.read()
    httpd.shutdown()


def test_state_endpoint_matches_session(cfg):
    s, httpd, conn = _client(cfg, mode="cold", name="srv-cold")
    conn.request("GET", "/api/state")
    got = json.loads(conn.getresponse().read())
    assert got == s.state()
    assert got["rows"][0]["confidence"] is None      # cold: stripped
    httpd.shutdown()


def test_clip_range_request_returns_206_partial(cfg):
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/clip/tt_point_01.mp4",
                 headers={"Range": "bytes=10-19"})
    r = conn.getresponse()
    body = r.read()
    assert r.status == 206
    assert r.getheader("Content-Range") == "bytes 10-19/1024"
    assert body == bytes(range(256))[10:20]
    httpd.shutdown()


def test_clip_without_range_returns_full_200(cfg):
    _, httpd, conn = _client(cfg)
    conn.request("GET", "/clip/tt_point_01.mp4")
    r = conn.getresponse()
    assert r.status == 200 and len(r.read()) == 1024
    httpd.shutdown()


def test_lint_event_accept_roundtrip(cfg):
    s, httpd, conn = _client(cfg)
    st, got = _post(conn, "/api/lint", {"first": "6b2Z*", "second": ""})
    assert st == 200 and len(got["issues"]) == 1
    st, _ = _post(conn, "/api/event",
                  {"ts_ms": 1, "row": "tt_point_01",
                   "event": "row_open", "payload": {}})
    assert st == 200
    st, _ = _post(conn, "/api/accept",
                  {"clip": "tt_point_01", "corrected_1st": "4b3f1w@",
                   "corrected_2nd": "", "notes": "", "skip_reason": ""})
    assert st == 200
    events = (s.dir / "events.jsonl").read_text()
    assert '"row_open"' in events and '"accept"' in events
    assert "tt_point_01" in (s.dir / "corrected.csv").read_text()
    httpd.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_review_server.py -q`
Expected: FAIL — `ImportError: cannot import name 'make_server'`

- [ ] **Step 3: Append the HTTP half to `courtvision/review.py`**

Add these imports at the top of the file (keep the existing ones):

```python
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import notation
```

Append below the ReviewSession class:

```python
# ---------------------------------------------------------------------------
# The HTTP app. One handler, one session, localhost only.
# ---------------------------------------------------------------------------

UI_PATH = Path(__file__).with_name("review_ui.html")
_CLIP_RE = re.compile(r"^/clip/([A-Za-z0-9_]+)\.mp4$")


def make_server(session, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):            # keep the terminal quiet
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/":
                body = UI_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(session.state())
            elif m := _CLIP_RE.match(self.path):
                self._send_clip(m.group(1))
            else:
                self.send_error(404)

        def _send_clip(self, stem):
            p = session.cfg.clip_path(stem)
            if not p.exists():
                self.send_error(404)
                return
            data = p.read_bytes()
            total = len(data)
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d+)-(\d*)", rng or "")
            if m:
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else total - 1
                b = min(b, total - 1)
                chunk = data[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {a}-{b}/{total}")
            else:
                a, chunk = 0, data
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def do_POST(self):
            body = self._read_json()
            if self.path == "/api/lint":
                self._json({"issues": notation.lint(
                    body.get("first", ""), body.get("second", ""))})
            elif self.path == "/api/event":
                session.append_event(body)
                self._json({"ok": True})
            elif self.path == "/api/accept":
                session.accept(body["clip"],
                               body.get("corrected_1st", ""),
                               body.get("corrected_2nd", ""),
                               body.get("notes", ""),
                               body.get("skip_reason", ""))
                session.append_event(
                    {"ts_ms": int(time.time() * 1000),
                     "row": body["clip"],
                     "event": ("skip" if body.get("skip_reason")
                               else "accept"),
                     "payload": {}})
                self._json({"ok": True})
            else:
                self.send_error(404)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run(cfg, mode, name, seed=None, n=None, port=8765,
        open_browser=True):
    session = ReviewSession(cfg, mode, name, seed=seed, n=n)
    httpd = make_server(session, port)
    url = f"http://127.0.0.1:{httpd.server_port}/"
    done = sum(1 for r in session.state()["rows"] if r["done"])
    print(f"{cfg.id} '{name}' [{mode}] {done}/{len(session.rows)} done "
          f"-> {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nsession saved:", session.dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_review_server.py -q`
Expected: `5 passed` (and `uv run pytest -q` → `22 passed` total)

- [ ] **Step 5: Commit**

```bash
git add courtvision/review.py tests/test_review_server.py
git commit -m "Review tool: stdlib HTTP app — state/lint/event/accept routes, Range video"
```

---

### Task 4: `review_ui.html` — the single-page UI

**Files:**
- Create: `courtvision/review_ui.html`

**Interfaces:**
- Consumes: the five endpoints from Task 3, exactly as specified there.
- Produces: the UI contract Trevor uses (keys, panes, telemetry events). Keyboard map: `Enter` accept and `Esc` blur work everywhere; ALL other keys are inert while an input has focus (MCP strings contain `p`, `c`, `l`, `n`, `x` — blur first). When not typing: `j/k` next/prev row · `space` play/pause · `←/→` ±1 frame (1/30 s) · `[ ]` speed down/up through 0.25/0.5/0.75/1/1.25/1.5/2 · `l` loop toggle (loops back to the row's seek point) · `f` cycle row filter all→high→low (review mode) · `1` focus 1st · `2` focus 2nd · `n` focus notes · `x` skip (prompts for reason) · `c` cheat sheet · `p` pause clock.

- [ ] **Step 1: Write the page**

Create `courtvision/review_ui.html` with exactly this content:

```html
<!doctype html>
<meta charset="utf-8">
<title>cv-review</title>
<style>
  :root { --bg:#181818; --fg:#d2d2d2; --dim:#787878; --hi:#50ff78;
          --amber:#ffa500; --card:#222; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--fg);
         font:14px/1.45 ui-monospace,Menlo,monospace; height:100vh;
         display:grid; grid-template-columns:340px 1fr;
         grid-template-rows:1fr auto; }
  #rows { grid-row:1/3; overflow-y:auto; border-right:1px solid #333;
          padding:8px; }
  #rows h1 { font-size:14px; color:#fff; padding:4px 6px; }
  #rows .prog { color:var(--dim); padding:0 6px 8px; }
  .row { padding:3px 6px; cursor:pointer; white-space:nowrap;
         overflow:hidden; text-overflow:ellipsis; }
  .row.cur { background:#2e2e2e; }
  .row.done { opacity:.45; }
  .row .conf-high { color:var(--hi); }
  .row .conf-low { color:var(--dim); }
  .row .conf-none { color:var(--amber); }
  #stage { display:flex; align-items:center; justify-content:center;
           background:#000; min-height:0; }
  video { max-width:100%; max-height:100%; }
  #bar { background:var(--card); padding:10px 14px; border-top:1px solid #333;
         display:grid; gap:6px;
         grid-template-columns:auto 1fr auto 1fr auto; align-items:center; }
  #bar label { color:var(--dim); }
  #bar input[type=text] { background:#111; color:#fff; border:1px solid #444;
         padding:6px 8px; font:16px ui-monospace,Menlo,monospace; width:100%; }
  #meta { grid-column:1/6; color:var(--dim); }
  #meta b { color:#fff; } #meta .hi { color:var(--hi); }
  #lint { grid-column:1/6; color:var(--amber); min-height:1.3em; }
  #notes { grid-column:1/5; }
  #accept { background:var(--hi); color:#000; border:0; padding:8px 18px;
            font-weight:bold; cursor:pointer; }
  #overlay, #cheat { position:fixed; inset:0; background:rgba(0,0,0,.92);
            display:none; z-index:9; padding:40px; color:#fff; }
  #overlay { z-index:10; }
  #overlay.on, #cheat.on { display:block; }
  #overlay h1 { font-size:40px; margin-bottom:12px; }
  #cheat pre { font-size:13px; color:var(--fg); }
  kbd { background:#333; padding:0 5px; border-radius:3px; }
</style>
<div id="rows"></div>
<div id="stage"><video id="v" preload="auto"></video></div>
<div id="bar">
  <div id="meta"></div>
  <label>1st</label><input id="first" type="text" autocomplete="off">
  <label>2nd</label><input id="second" type="text" autocomplete="off">
  <button id="accept">accept ⏎</button>
  <input id="notes" type="text" placeholder="notes" autocomplete="off">
  <div id="lint"></div>
</div>
<div id="overlay"><h1>CLOCK PAUSED</h1><p>press <kbd>p</kbd> to resume</p></div>
<div id="cheat"><pre>
MCP CHEAT SHEET                                   (press c to close)
serve   4 wide  5 body  6 T          faults: n net w wide d deep
shots   f/b drive   r/s slice        x both  g footfault  ! shank
        v/z volley  o/p overhead     let: c before the serve digit
        u/y drop    l/m lob
        h/i half-volley  j/k swinging volley  t trick  q unknown
dir     1 to rh's forehand  2 middle  3 to rh's backhand
depth   7 short  8 mid  9 deep       (optional, after direction)
end     * winner   @ unforced error  # forced error
        error letter (n/w/d/x/!/e) goes BEFORE @ or #
example 4b28f1f3b2n@  = wide serve, deep bh cross, ... bh into net UE
faulted 1st serve: put the fault in 1st (e.g. "4d"), point in 2nd
</pre></div>
<script>
const $ = s => document.querySelector(s);
const v = $("#v");
const SPEEDS = [0.25,0.5,0.75,1,1.25,1.5,2];
let S = null, cur = -1, paused = false, loop = false, lintT = null;
let filt = "all", rowSeek = 0;

const send = (event, payload={}) => fetch("/api/event", {method:"POST",
  headers:{"Content-Type":"application/json"},
  body: JSON.stringify({ts_ms: Date.now(),
    row: cur >= 0 ? S.rows[cur].clip : "", event, payload})});

async function boot() {
  S = await (await fetch("/api/state")).json();
  document.title = `cv-review ${S.match} ${S.session} [${S.mode}]`;
  renderRows();
  openRow(S.rows.findIndex(r => !r.done) >= 0
          ? S.rows.findIndex(r => !r.done) : 0);
}

function renderRows() {
  const done = S.rows.filter(r => r.done).length;
  const ftag = filt === "all" ? "" : ` · filter:${filt}`;
  $("#rows").innerHTML =
    `<h1>${S.match} · ${S.session} · ${S.mode}</h1>` +
    `<div class="prog">${done}/${S.rows.length} done${ftag}</div>` +
    S.rows.map((r, i) => {
      if (filt !== "all" && r.confidence !== filt && i !== cur)
        return "";
      const conf = r.confidence === null ? "" :
        ` <span class="conf-${r.pt ? r.confidence : "none"}">` +
        `${r.confidence ?? ""}</span>`;
      return `<div class="row ${i===cur?"cur":""} ${r.done?"done":""}"
        data-i="${i}">${r.pt || "—"} · ${r.pts || "?"} · ${r.clip}` +
        `${conf}${r.skipped ? " ⏭" : ""}</div>`;
    }).join("");
  document.querySelectorAll(".row").forEach(el =>
    el.onclick = () => openRow(+el.dataset.i));
}

function openRow(i) {
  cur = i;
  const r = S.rows[i];
  v.src = `/clip/${r.clip}.mp4`;
  v.playbackRate = 1;
  const seekTo = r.serve_s !== null
    ? Math.max(0, parseFloat(r.serve_s) - 1.5) : 0;
  rowSeek = seekTo;
  v.onloadedmetadata = () => { v.currentTime = seekTo; v.play(); };
  $("#first").value = r.corrected_1st || r.first_draft || "";
  $("#second").value = r.corrected_2nd || "";
  $("#notes").value = r.notes || "";
  const conf = r.confidence !== null
    ? ` · <span class="hi">${r.confidence} ${r.conf_p}</span>` : "";
  const jump = r.serve_s !== null ? ` · serve @ ${r.serve_s}s` : "";
  $("#meta").innerHTML = `<b>${r.clip}</b> · Pt ${r.pt || "—"} · ` +
    `${r.pts || "?"} · Svr ${r.svr || "?"}${conf}${jump}`;
  $("#lint").textContent = "";
  renderRows();
  send("row_open", {i});
  lint();
}

function lint() {
  clearTimeout(lintT);
  lintT = setTimeout(async () => {
    const res = await (await fetch("/api/lint", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({first: $("#first").value,
                            second: $("#second").value})})).json();
    $("#lint").textContent = res.issues.map(i =>
      `${i.field}${i.pos !== null ? ":" + i.pos : ""} ${i.msg}`)
      .join("   ");
  }, 300);
}

async function accept(skipReason="") {
  const r = S.rows[cur];
  let res;
  try {
    res = await fetch("/api/accept", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({clip: r.clip,
        corrected_1st: $("#first").value.trim(),
        corrected_2nd: $("#second").value.trim(),
        notes: $("#notes").value.trim(), skip_reason: skipReason})});
  } catch (e) {
    $("#lint").textContent = "SAVE FAILED (network) — row NOT accepted";
    return;
  }
  if (!res.ok) {
    $("#lint").textContent =
      `SAVE FAILED (${res.status}) — row NOT accepted`;
    return;
  }
  Object.assign(r, {done: true, skipped: skipReason,
    corrected_1st: $("#first").value.trim(),
    corrected_2nd: $("#second").value.trim(),
    notes: $("#notes").value.trim()});
  const next = S.rows.findIndex((x, j) => j > cur && !x.done);
  openRow(next >= 0 ? next : cur);
}

["first", "second", "notes"].forEach(id =>
  $("#" + id).addEventListener("input", () => {
    send("edit", {field: id}); lint();
  }));
$("#accept").onclick = () => accept();
v.onplay = () => send("play", {t: v.currentTime});
v.onpause = () => send("pause", {t: v.currentTime});
v.onseeked = () => send("seek", {t: v.currentTime});
v.ontimeupdate = () => { if (loop && v.currentTime > v.duration - .05)
  v.currentTime = rowSeek; };

document.addEventListener("keydown", e => {
  if (paused && e.key !== "p") { e.preventDefault(); return; }
  const typing = ["INPUT"].includes(e.target.tagName);
  if (e.key === "Escape") { e.target.blur(); return; }
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); accept();
    return; }
  if (typing) return;            // p/c/l/n/x are legal MCP letters
  if (e.key === "p") { paused = !paused;
    $("#overlay").classList.toggle("on", paused);
    if (paused) { $("#cheat").classList.remove("on"); v.pause(); }
    send(paused ? "clock_pause" : "clock_resume");
    return; }
  const rate = SPEEDS.indexOf(v.playbackRate);
  const acts = {
    "j": () => openRow(Math.min(cur + 1, S.rows.length - 1)),
    "k": () => openRow(Math.max(cur - 1, 0)),
    " ": () => v.paused ? v.play() : v.pause(),
    "ArrowLeft":  () => { v.pause(); v.currentTime -= 1/30; },
    "ArrowRight": () => { v.pause(); v.currentTime += 1/30; },
    "[": () => { v.playbackRate = SPEEDS[Math.max(rate - 1, 0)];
                 send("rate", {r: v.playbackRate}); },
    "]": () => { v.playbackRate =
                   SPEEDS[Math.min(rate + 1, SPEEDS.length - 1)];
                 send("rate", {r: v.playbackRate}); },
    "l": () => { loop = !loop; },
    "f": () => { if (S.mode !== "review") return;
                 const o = ["all", "high", "low"];
                 filt = o[(o.indexOf(filt) + 1) % 3]; renderRows(); },
    "1": () => $("#first").focus(),
    "2": () => $("#second").focus(),
    "n": () => $("#notes").focus(),
    "x": () => { const why = prompt("skip reason:");
                 if (why) accept(why); },
    "c": () => { $("#cheat").classList.toggle("on"); send("cheat"); },
  };
  if (acts[e.key]) { e.preventDefault(); acts[e.key](); }
});
boot();
</script>
```

- [ ] **Step 2: Automated check that the server serves it**

Run: `uv run pytest tests/test_review_server.py::test_root_serves_ui -q`
(then also run the real file through Python's HTML sanity check)
Run: `uv run python -c "from courtvision.review import UI_PATH; t=UI_PATH.read_text(); assert 'cv-review' in t and '/api/accept' in t and len(t)>4000; print('ui ok', len(t), 'bytes')"`
Expected: `ui ok <N> bytes`

- [ ] **Step 3: Manual smoke (needs Task 5's CLI — do a module-level run now)**

Run: `uv run python -c "
from courtvision import config, review
review.run(config.load('t3'), 'review', 'smoke', port=8765,
           open_browser=True)"`

Checklist (then Ctrl-C; delete the session with `rm -rf outputs/t3/review/smoke`):
- [ ] page loads, 59 rows listed, HIGH rows green
- [ ] clicking a row plays its clip; auto-seeks near the serve
- [ ] `space`/arrows/`[`/`]`/`l` control playback; `j`/`k` move rows
- [ ] typing in 1st shows lint warnings live (try `6b2Z*` → unknown mark)
- [ ] `Enter` accepts and advances; row dims; progress counter ticks
- [ ] `p` shows CLOCK PAUSED overlay and freezes keys; `p` resumes
- [ ] `c` toggles the cheat sheet; `x` prompts for a skip reason
- [ ] `f` cycles the row filter all→high→low (current row stays visible)
- [ ] restart the same command: accepted row still shows corrected value

- [ ] **Step 4: Commit**

```bash
git add courtvision/review_ui.html
git commit -m "Review tool: single-page UI — three panes, keyboard-first, cheat sheet, clock pause"
```

---

### Task 5: CLI wiring — `review` subcommand

**Files:**
- Modify: `courtvision/cli.py`

**Interfaces:**
- Consumes: `review.run(cfg, mode, name, seed, n, port, open_browser)` (Task 3).
- Produces: `uv run python -m courtvision review <match> --mode review|cold --session NAME [--seed S] [--n 10] [--port 8765] [--no-browser]`.

- [ ] **Step 1: Add the subparser**

In `courtvision/cli.py`, after the `decompose` subparser line (`sub.add_parser("decompose", ...)`), insert:

```python
    p = sub.add_parser("review",
                       help="correct drafts against clips (local web UI)")
    p.add_argument("match")
    p.add_argument("--mode", choices=["review", "cold"], required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--seed", default=None)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
```

And at the bottom of the dispatch chain, before the final blank line, add:

```python
    elif args.cmd == "review":
        from . import review
        review.run(config.load(args.match), args.mode, args.session,
                   seed=args.seed, n=args.n, port=args.port,
                   open_browser=not args.no_browser)
```

Also add one line to the module docstring command list at the top:

```python
    uv run python -m courtvision review <match> --mode review --session r1
```

- [ ] **Step 2: Verify headless**

Run: `uv run python -m courtvision review t3 --mode cold --session cli-smoke --seed x --n 3 --no-browser & sleep 2 && curl -s http://127.0.0.1:8765/api/state | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['mode']=='cold' and len(d['rows'])==3 and d['rows'][0]['confidence'] is None; print('cli ok')" && kill %1 && rm -rf outputs/t3/review/cli-smoke`
Expected: `cli ok`

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add courtvision/cli.py
git commit -m "Review tool: CLI wiring — courtvision review <match> --mode --session"
```

---

### Task 6: `review_analysis.py` — timing, accuracy, anchoring, histogram

**Files:**
- Create: `courtvision/review_analysis.py`
- Create: `tests/test_review_analysis.py`
- Modify: `courtvision/cli.py`

**Interfaces:**
- Consumes: session dirs (Task 2 layout), export CSVs, MCP truth via `cfg.eval.mcp_map` (columns `clip,note,mcp_pt,svr,first,second,winner,status,gms,pts`; truth strings are `first`/`second`, `status=="matched"`); `courtvision/mcp.py` (`mcp_point_tokens`, `token_levenshtein`, `backtrace`, `classify_sub`, `tok_kind`).
- Produces:
  - `active_seconds(events) -> (dict clip->seconds, list idle_holes)` — events = parsed JSONL dicts; gap rule: accumulate `ts_ms` gaps to the current row (set by `row_open`) while not clock-paused; a gap > 180 000 ms becomes an idle hole `(clip, seconds)` instead of active time; `accept`/`skip` close the row (a later `row_open` on the same clip accumulates further).
  - `draft_point_tokens(draft) -> list` — tokenizer for OUR draft grammar (leading `s`, `?` letters kept as `?` shots).
  - `truth_played(map_row) -> str` — `second if second.strip() else first`.
  - `analyze(specs, out_path=None) -> str` — `specs = {"cold_a": (cfg, name), "review": (cfg, name), "cold_b": (cfg, name)}`; returns the report text, writes it to `out_path` (default `<review session dir>/analysis.md`).
  - CLI: `uv run python -m courtvision review-analyze --cold-a t6:NAME --review t6:NAME --cold-b t7:NAME`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_review_analysis.py`:

```python
from courtvision.review_analysis import (active_seconds,
                                         draft_point_tokens)


def ev(ts, event, row="c1"):
    return {"ts_ms": ts, "row": row, "event": event, "payload": {}}


def test_active_seconds_basic_and_pause():
    events = [ev(0, "row_open"), ev(10_000, "play"),
              ev(20_000, "clock_pause"), ev(80_000, "clock_resume"),
              ev(90_000, "accept")]
    active, holes = active_seconds(events)
    # 0->10s + 10->20s + resume 80->90s = 30s; the pause never counts
    assert active == {"c1": 30.0}
    assert holes == []


def test_active_seconds_idle_hole_excluded():
    events = [ev(0, "row_open"), ev(300_000, "accept")]
    active, holes = active_seconds(events)
    assert active.get("c1", 0.0) == 0.0
    assert holes == [("c1", 300.0)]


def test_active_seconds_two_rows_and_revisit():
    events = [ev(0, "row_open", "c1"), ev(5_000, "row_open", "c2"),
              ev(9_000, "accept", "c2"), ev(9_000, "row_open", "c1"),
              ev(15_000, "accept", "c1")]
    active, _ = active_seconds(events)
    assert active == {"c1": 11.0, "c2": 4.0}


def test_draft_tokens_strip_s_and_keep_unknowns():
    assert draft_point_tokens("s4b3f1w@") == ["s4", "b3", "f1", "w"]
    # '?' letters are real shots the tokenizer must not drop
    assert draft_point_tokens("s?f2?3?") == ["s?", "f2", "?3", "?"]
    assert draft_point_tokens("s6*") == ["s6", "*"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_review_analysis.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `courtvision/review_analysis.py`**

```python
"""cv-18 analysis — what the stopwatch and the corrections say.

Consumes review sessions (events.jsonl + corrected.csv), the draft
exports, and MCP truth (the *_mcp_map.csv 'first'/'second' columns —
this is the ONLY review-side module allowed to read truth). Produces
the cv-18 tables:

  timing     s/point: cold-A vs review(HIGH) vs review(low) vs cold-B
  accuracy   Trevor-vs-MCP token edits per arm (cold and corrected)
  anchoring  every HIGH row where the DRAFT was >5 edits from truth:
             did the correction escape the draft or rubber-stamp it?
  histogram  draft->corrected edit ops by class (what ate the time)
  triage     draft->corrected edits, HIGH vs low

Timing rule (from the spec): inter-event gaps over 180 s are idle
holes, reported but never counted; the clock_pause key stops the
clock entirely. Cold-A rows are excluded from review-pass timing by
manifest, not by hand.
"""

import csv
import json
from collections import Counter

from .mcp import (mcp_point_tokens, token_levenshtein, backtrace,
                  classify_sub, tok_kind)

MAX_GAP_MS = 180_000
FH = set("frvoulhjt")
BH = set("bszpymik")


def active_seconds(events):
    """{clip: active_s}, [(clip, hole_s), ...] per the timing rule."""
    active, holes = {}, []
    row, paused, last = None, False, None
    for e in sorted(events, key=lambda e: e["ts_ms"]):
        ts = e["ts_ms"]
        if row is not None and last is not None and not paused:
            gap = ts - last
            if gap > MAX_GAP_MS:
                holes.append((row, gap / 1000.0))
            else:
                active[row] = active.get(row, 0.0) + gap / 1000.0
        if e["event"] == "row_open":
            row = e["row"]
            active.setdefault(row, 0.0)
        elif e["event"] == "clock_pause":
            paused = True
        elif e["event"] == "clock_resume":
            paused = False
        elif e["event"] in ("accept", "skip"):
            row = None
        last = ts
    return {k: round(v, 3) for k, v in active.items()}, \
        [(c, round(s, 3)) for c, s in holes]


def draft_point_tokens(draft):
    """Tokenize OUR draft grammar: leading 's', '?' letters kept.

    Mirrors mcp.mcp_point_tokens but '?' is a legal shot letter here
    (the draft's way of saying 'a shot happened, side unknown'). The
    ending is peeled off FIRST — the final '?' of a string like
    's?f2?3?' is the unknown ending, not an extra shot. Ambiguous
    '?' runs (e.g. 'b??') resolve greedily; fine for histograms."""
    s = draft[1:] if draft.startswith("s") else draft
    core = s.rstrip("@#!")
    if core.endswith("*"):
        ending, body = "*", core[:-1]
    elif core and core[-1] in "nwdx?":
        ending, body = core[-1], core[:-1]
    else:
        ending, body = "?", core
    serve = body[0] if body and body[0] in "0456" else "?"
    toks = [f"s{serve if serve in '456' else '?'}"]
    i = 1
    shotset = FH | BH | {"?"}
    while i < len(body):
        c = body[i]
        if c in shotset:
            letter = "f" if c in FH else "b" if c in BH else "?"
            direction = "?"
            j = i + 1
            while j < len(body) and body[j] not in shotset:
                if body[j] in "123" and direction == "?":
                    direction = body[j]
                j += 1
            toks.append(f"{letter}{direction}")
            i = j
        else:
            i += 1
    toks.append(ending)
    return toks


def truth_played(map_row):
    return (map_row["second"] if map_row["second"].strip()
            else map_row["first"])


def corrected_played(corr_row):
    return (corr_row["corrected_2nd"]
            if corr_row["corrected_2nd"].strip()
            else corr_row["corrected_1st"])


def _load_session(cfg, name):
    d = cfg.out_dir / "review" / name
    man = json.loads((d / "manifest.json").read_text())
    events = [json.loads(l) for l in
              (d / "events.jsonl").read_text().splitlines()]
    corr = {}
    if (d / "corrected.csv").exists():
        with open(d / "corrected.csv") as f:
            corr = {r["clip"]: r for r in csv.DictReader(f)}
    with open(cfg.out_dir / "export" / f"{cfg.id}_mcp_draft.csv") as f:
        export = {r["clip"]: r for r in csv.DictReader(f)}
    truth = {}
    with open(cfg.eval.mcp_map) as f:
        truth = {r["clip"]: r for r in csv.DictReader(f)
                 if r["status"] == "matched"}
    return {"dir": d, "man": man, "events": events, "corr": corr,
            "export": export, "truth": truth}


def _med(xs):
    xs = sorted(xs)
    return 0.0 if not xs else round(xs[len(xs) // 2], 1)


def _fmt_arm(label, secs):
    if not secs:
        return f"{label:<22} n=0"
    return (f"{label:<22} n={len(secs):<4} median {_med(secs):>6}s  "
            f"mean {round(sum(secs)/len(secs), 1):>6}s")


def _edits_vs_truth(sess, clips):
    out = []
    for c in clips:
        t, k = sess["truth"].get(c), sess["corr"].get(c)
        if not t or not k or k["flags"].startswith("skipped"):
            continue
        d = token_levenshtein(mcp_point_tokens(truth_played(t)),
                              mcp_point_tokens(corrected_played(k)))
        out.append((c, d))
    return out


def analyze(specs, out_path=None):
    ca_cfg, ca = specs["cold_a"]
    rv_cfg, rv = specs["review"]
    cb_cfg, cb = specs["cold_b"]
    A = _load_session(ca_cfg, ca)
    R = _load_session(rv_cfg, rv)
    B = _load_session(cb_cfg, cb)
    contaminated = set(A["man"]["rows"])

    L = ["# cv-18 analysis", ""]
    L.append(f"sessions: cold_a={ca} review={rv} cold_b={cb}")
    L.append("")

    # -- timing --------------------------------------------------------
    L.append("## timing (active s/point; >180s gaps excluded)")
    all_holes = []
    secs = {}
    for tag, S in (("cold_a", A), ("review", R), ("cold_b", B)):
        act, holes = active_seconds(S["events"])
        secs[tag] = act
        all_holes += [(tag, c, s) for c, s in holes]
    hi = {c for c, r in R["export"].items()
          if r["confidence"] == "high"}
    rv_rows = [c for c in R["man"]["rows"] if c not in contaminated
               and c in R["corr"]]
    L.append(_fmt_arm("cold-A (t6)", [secs["cold_a"][c]
             for c in A["man"]["rows"] if c in secs["cold_a"]
             and c in A["corr"]]))
    L.append(_fmt_arm("review HIGH", [secs["review"][c]
             for c in rv_rows if c in hi and c in secs["review"]]))
    L.append(_fmt_arm("review low", [secs["review"][c]
             for c in rv_rows if c not in hi and c in secs["review"]]))
    L.append(_fmt_arm("cold-B (t7)", [secs["cold_b"][c]
             for c in B["man"]["rows"] if c in secs["cold_b"]
             and c in B["corr"]]))
    L.append(f"contaminated (cold-A rows in review timing): "
             f"{len(contaminated)} excluded")
    if all_holes:
        L.append("idle holes (excluded): " + ", ".join(
            f"{t}:{c} {s:.0f}s" for t, c, s in all_holes))
    L.append("")

    # -- accuracy vs MCP -----------------------------------------------
    L.append("## Trevor vs MCP (token edits on the played string)")
    for label, S, clips in (
            ("cold-A", A, A["man"]["rows"]),
            ("review (all corrected)", R, R["man"]["rows"]),
            ("cold-B", B, B["man"]["rows"])):
        ed = _edits_vs_truth(S, clips)
        if ed:
            ds = [d for _, d in ed]
            L.append(f"{label:<24} n={len(ds):<4} median {_med(ds)}  "
                     f"exact {sum(d == 0 for d in ds)}  "
                     f"<=1 {sum(d <= 1 for d in ds)}")
    L.append("")

    # -- fault agreement -----------------------------------------------
    L.append("## fault agreement (was there a 2nd serve?)")
    for label, S, clips in (("cold-A", A, A["man"]["rows"]),
                            ("review", R, R["man"]["rows"]),
                            ("cold-B", B, B["man"]["rows"])):
        tp = fp = fn = tn = 0
        for c in clips:
            t, k = S["truth"].get(c), S["corr"].get(c)
            if not t or not k or k["flags"].startswith("skipped"):
                continue
            th = bool(t["second"].strip())
            kh = bool(k["corrected_2nd"].strip())
            tp += th and kh; fp += (not th) and kh
            fn += th and (not kh); tn += (not th) and (not kh)
        L.append(f"{label:<10} both-fault {tp}  trevor-only {fp}  "
                 f"mcp-only {fn}  both-clean {tn}")
    L.append("")

    # -- anchoring -----------------------------------------------------
    L.append("## anchoring: HIGH rows whose DRAFT was >5 edits from "
             "truth")
    n_listed = 0
    for c in R["man"]["rows"]:
        r = R["export"][c]
        t = R["truth"].get(c)
        k = R["corr"].get(c)
        if r["confidence"] != "high" or not t or not k:
            continue
        tt = mcp_point_tokens(truth_played(t))
        dd = draft_point_tokens(r["1st"])
        if token_levenshtein(tt, dd) <= 5:
            continue
        kk = mcp_point_tokens(corrected_played(k))
        L.append(f"  {c}: draft->truth "
                 f"{token_levenshtein(tt, dd)}, corrected->truth "
                 f"{token_levenshtein(tt, kk)}, corrected->draft "
                 f"{token_levenshtein(dd, kk)}")
        n_listed += 1
    if not n_listed:
        L.append("  (none — no confidently-wrong drafts in this file)")
    L.append("")

    # -- correction histogram + triage ----------------------------------
    L.append("## what Trevor changed (draft -> corrected, review pass)")
    bins = Counter()
    edits_by_tier = {"high": [], "low": []}
    for c in R["man"]["rows"]:
        r, k = R["export"][c], R["corr"].get(c)
        if not k or k["flags"].startswith("skipped"):
            continue
        dd = draft_point_tokens(r["1st"])
        kk = mcp_point_tokens(corrected_played(k))
        dist, ops = backtrace(dd, kk)
        edits_by_tier[r["confidence"]].append(dist)
        for op, a, b in ops:
            if op == "sub":
                bins[classify_sub(b, a)] += 1
            elif op == "del":
                bins[f"del_{tok_kind(a)}"] += 1
            elif op == "ins":
                bins[f"ins_{tok_kind(b)}"] += 1
        if k["corrected_2nd"].strip():
            bins["fault_added"] += 1
    for name, n in bins.most_common():
        L.append(f"  {name:<18} {n}")
    L.append("")
    L.append("## triage honesty (draft->corrected edits)")
    for tier in ("high", "low"):
        ds = edits_by_tier[tier]
        if ds:
            L.append(f"  {tier:<5} n={len(ds):<4} median {_med(ds)}  "
                     f"mean {round(sum(ds)/len(ds), 2)}")

    report = "\n".join(L) + "\n"
    out = out_path or (R["dir"] / "analysis.md")
    with open(out, "w") as f:
        f.write(report)
    print(report)
    print(f"-> {out}")
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_review_analysis.py -q`
Expected: `4 passed`

- [ ] **Step 5: Wire the CLI**

In `courtvision/cli.py`, after the `review` subparser block, add:

```python
    p = sub.add_parser("review-analyze",
                       help="cv-18 tables from review sessions")
    p.add_argument("--cold-a", required=True, metavar="MATCH:SESSION")
    p.add_argument("--review", required=True, metavar="MATCH:SESSION")
    p.add_argument("--cold-b", required=True, metavar="MATCH:SESSION")
```

And in the dispatch chain, after the `review` branch:

```python
    elif args.cmd == "review-analyze":
        from . import review_analysis
        def _spec(s):
            mid, name = s.split(":", 1)
            return (config.load(mid), name)
        review_analysis.analyze({"cold_a": _spec(args.cold_a),
                                 "review": _spec(args.review),
                                 "cold_b": _spec(args.cold_b)})
```

- [ ] **Step 6: End-to-end dry run on synthetic sessions**

Run: `uv run pytest -q` (all green), then a live sanity pass — create a 2-row cold session on t3, accept both rows through the running server with curl, and run the analyzer against it three times over (standing in for all three blocks):

```bash
uv run python -m courtvision review t3 --mode cold --session an-dry --seed d --n 2 --no-browser &
sleep 2
for CLIP in $(curl -s http://127.0.0.1:8765/api/state | python3 -c "import json,sys; print(' '.join(r['clip'] for r in json.load(sys.stdin)['rows']))"); do
  curl -s -X POST http://127.0.0.1:8765/api/event -d "{\"ts_ms\": $(date +%s000), \"row\": \"$CLIP\", \"event\": \"row_open\", \"payload\": {}}" >/dev/null
  curl -s -X POST http://127.0.0.1:8765/api/accept -d "{\"clip\": \"$CLIP\", \"corrected_1st\": \"6b2f1*\", \"corrected_2nd\": \"\", \"notes\": \"\", \"skip_reason\": \"\"}" >/dev/null
done
kill %1
uv run python -m courtvision review-analyze --cold-a t3:an-dry --review t3:an-dry --cold-b t3:an-dry
rm -rf outputs/t3/review/an-dry
```

Expected: the report prints with all six sections, real numbers, no traceback.

- [ ] **Step 7: Commit**

```bash
git add courtvision/review_analysis.py tests/test_review_analysis.py courtvision/cli.py
git commit -m "Review analysis: timing/accuracy/anchoring/histogram -> cv-18 tables"
```

---

### Task 7: Protocol runbook + USAGE

**Files:**
- Create: `docs/cv18-protocol.md`
- Modify: `docs/USAGE.md`

**Interfaces:**
- Consumes: the CLI surface from Tasks 5–6.
- Produces: the exact commands Trevor runs, in order, with the frozen seeds.

- [ ] **Step 1: Write `docs/cv18-protocol.md`**

```markdown
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
```

- [ ] **Step 2: Add the review section to `docs/USAGE.md`**

Append after the existing `draft`/`export` section:

```markdown
## Correcting drafts: the review tool

`review` opens a local web UI for correcting a match's draft export
against its clips — the charter-assist surface, and the instrument
for the cv-18 stopwatch experiment (protocol: docs/cv18-protocol.md).

    uv run python -m courtvision review t6 --mode review --session r1
    uv run python -m courtvision review t6 --mode cold --session c1 \
        --seed cv18-a --n 10          # drafts hidden, gradeable rows

Sessions live in `outputs/<match>/review/<session>/` (manifest,
events.jsonl telemetry, corrected.csv) and resume if rerun. `cold`
mode hides everything machine-derived (draft string, confidence,
serve timestamp); `review` mode pre-fills the draft. Lint warnings
under the inputs are advisory MCP-legality checks and never block.

    uv run python -m courtvision review-analyze \
        --cold-a t6:c1 --review t6:r1 --cold-b t7:c2

writes the timing/accuracy/anchoring/histogram report to the review
session's directory.
```

- [ ] **Step 3: Final full-suite check and commit**

Run: `uv run pytest -q`
Expected: all tests pass

```bash
git add docs/cv18-protocol.md docs/USAGE.md
git commit -m "cv-18 protocol runbook + USAGE: review tool + analyzer"
```

---

## Self-Review (run after Task 7)

- [ ] Spec coverage: tool (Tasks 2–5), one-variable arms (cold strip test), telemetry (Task 2 events + Task 6 timing), Cold-A/review/Cold-B with contamination-by-manifest (Task 6 `contaminated`), anchoring/histogram/triage/fault metrics (Task 6), 180 s idle rule (Task 6), warm-up + frozen seeds (Task 7), resumable sessions (Task 2), Range video (Task 3), lint-never-blocks (Task 1), no-truth-in-tool (imports in `review.py` must not mention `mcp_map`/`mcp_points` — verify with `grep -n "mcp_" courtvision/review.py`).
- [ ] `uv run pytest -q` green; `git log --oneline` shows one commit per task.
- [ ] Manual smoke checklist from Task 4 Step 3 completed on t3.
