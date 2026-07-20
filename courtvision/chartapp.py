"""The charting app — chart-along sessions + (Task 7) the HTTP app.

ChartSession is the ground-truth factory's memory: it stores RAW
inputs only (strings, winner override, boundary stamps, notes,
flags); every score column is recomputed by replaying score.Score
over the points, so an edit to point 5 automatically rescores point
50. Staged-match charting reuses the frozen review.ReviewSession —
this module never duplicates it.

Layout, under outputs/charting/<match_id>/:
  manifest.json   setup (players, format, first server, video name)
  points.csv      raw inputs, one row per point, in order
  events.jsonl    telemetry, same line contract as the bench
"""

import csv
import json
import time
from pathlib import Path

from .config import ROOT
from . import notation
from .score import Score, winner_from_strings

UI_PATH = Path(__file__).with_name("chart_ui.html")
CONFORMANCE_PATH = (Path(__file__).resolve().parent.parent / "tests"
                    / "fixtures" / "score_conformance.json")

CHARTING_ROOT = ROOT / "outputs" / "charting"
RAW_FIELDS = ["first", "second", "notes", "winner", "start_s",
              "end_s", "flags"]
EXPORT_FIELDS = ["match_id", "Pt", "Set1", "Set2", "Gm1", "Gm2",
                 "Pts", "Gm#", "TbSet", "Svr", "1st", "2nd", "Notes",
                 "PtWinner"]
SETUP_KEYS = {"player1", "player2", "best_of", "final_set",
              "first_server", "video"}


class ChartSession:
    def __init__(self, match_id, setup=None):
        self.match_id = match_id
        self.dir = CHARTING_ROOT / match_id
        man_p = self.dir / "manifest.json"
        if man_p.exists():
            man = json.loads(man_p.read_text())
            if setup is not None and setup != man["setup"]:
                raise ValueError(f"{match_id}: setup differs from "
                                 f"existing session")
            self.setup = man["setup"]
        else:
            if setup is None or set(setup) != SETUP_KEYS:
                raise ValueError(f"new session '{match_id}' needs "
                                 f"setup keys {sorted(SETUP_KEYS)}")
            self.dir.mkdir(parents=True, exist_ok=True)
            self.setup = setup
            man_p.write_text(json.dumps(
                {"setup": setup,
                 "created_ts_ms": int(time.time() * 1000),
                 "grammar_version": notation.GRAMMAR["version"]},
                indent=2))
        self.points = self._load_points()

    # -- storage --------------------------------------------------------

    def _points_path(self):
        return self.dir / "points.csv"

    def _load_points(self):
        p = self._points_path()
        if not p.exists():
            return []
        return list(csv.DictReader(open(p)))

    def _save_points(self):
        with open(self._points_path(), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            w.writeheader()
            w.writerows(self.points)

    # -- mutation -------------------------------------------------------

    def add_point(self, first, second, notes="", winner=None,
                  start_s=None, end_s=None, flags=""):
        if flags != "unseen" and winner is None:
            if winner_from_strings(first, second) is None:
                raise ValueError("winner required: ending is "
                                 "underivable from the string")
        row = {"first": first, "second": second, "notes": notes,
               "winner": "" if winner is None else str(winner),
               "start_s": "" if start_s is None else str(start_s),
               "end_s": "" if end_s is None else str(end_s),
               "flags": flags}
        self.points.append(row)
        self._save_points()
        return row

    def insert_unseen(self, winner):
        return self.add_point("", "", winner=winner, flags="unseen")

    def update_point(self, idx, **fields):
        bad = set(fields) - set(RAW_FIELDS)
        if bad:
            raise ValueError(f"unknown fields {sorted(bad)}")
        self.points[idx].update(
            {k: str(v) if v is not None else "" for k, v in
             fields.items()})
        self._save_points()

    # -- replay ---------------------------------------------------------

    def _winner_player(self, row, server):
        if row["winner"]:
            return int(row["winner"])
        rel = winner_from_strings(row["first"], row["second"])
        if rel is None:
            return None
        return server if rel == 1 else (2 if server == 1 else 1)

    def _replay(self):
        sc = Score(best_of=int(self.setup["best_of"]),
                   final_set=self.setup["final_set"],
                   first_server=int(self.setup["first_server"]))
        out = []
        for i, row in enumerate(self.points):
            server = sc.server
            w = self._winner_player(row, server)
            if w is None:
                raise ValueError(f"point {i + 1}: no winner stored "
                                 f"or derivable")
            ctx = sc.point(w)
            out.append((row, ctx, w))
        return sc, out

    def state(self):
        sc, replayed = self._replay()
        pts = []
        for i, (row, ctx, w) in enumerate(replayed):
            d = dict(row)
            d.update(ctx)
            d["pt"] = i + 1
            d["PtWinner"] = str(w)
            pts.append(d)
        return {"match_id": self.match_id, "setup": self.setup,
                "points": pts, "score_now": sc.display,
                "over": sc.over, "next_server": sc.server}

    def export_rows(self):
        rows = []
        for p in self.state()["points"]:
            notes = p["notes"]
            if p["flags"] == "unseen":
                notes = f"unseen;{notes}" if notes else "unseen;"
            rows.append({"match_id": self.match_id,
                         "Pt": str(p["pt"]), "Set1": p["Set1"],
                         "Set2": p["Set2"], "Gm1": p["Gm1"],
                         "Gm2": p["Gm2"], "Pts": p["Pts"],
                         "Gm#": p["Gm#"], "TbSet": p["TbSet"],
                         "Svr": p["Svr"], "1st": p["first"],
                         "2nd": p["second"], "Notes": notes,
                         "PtWinner": p["PtWinner"]})
        return rows

    def segments(self):
        out = []
        for p in self.state()["points"]:
            if p["start_s"] and p["end_s"]:
                out.append({"clip": f"{self.match_id}_point_"
                                    f"{p['pt']:03d}",
                            "start_s": float(p["start_s"]),
                            "end_s": float(p["end_s"])})
        return out

    def append_event(self, evt):
        evt = dict(evt)
        evt["session"], evt["mode"] = self.match_id, "chart"
        evt["server_ts_ms"] = int(time.time() * 1000)
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(evt) + "\n")
