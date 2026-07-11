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
