"""The charting app — chart-along sessions + (Task 7) the HTTP app.

ChartSession is the ground-truth factory's memory: it stores RAW
inputs only (strings, the ATTESTED winner, boundary stamps, notes,
flags). The winner is what the charter watched happen — it is truth,
never re-derived. Every bookkeeping column (score, server, games) is
recomputed by replaying score.Score over the points, so an edit to
point 5 automatically rescores point 50's context; where the replay
then contradicts an attested winner (string says the server won, but
the replayed server isn't who the charter saw win), the point is
flagged `conflict` and export refuses until the chart is reconciled.
Staged-match charting reuses the frozen review.ReviewSession — this
module never duplicates it.

Layout, under outputs/charting/<match_id>/:
  manifest.json   setup (players, format, first server, video name)
  points.csv      raw inputs, one row per point, in order
  events.jsonl    telemetry, same line contract as the bench
"""

import csv
import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import ROOT
from . import httpkit, notation
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
        with open(p) as f:
            return list(csv.DictReader(f))

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
        old = dict(self.points[idx])
        self.points[idx].update(
            {k: str(v) if v is not None else "" for k, v in
             fields.items()})
        try:
            self._replay()          # an edit must never brick replay
        except ValueError:
            self.points[idx] = old
            raise
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
            # consistency audit: the stored winner is the charter's
            # attested fact; if the string derives a winner under the
            # REPLAYED server and they disagree, something in the
            # chart is wrong somewhere — flag it, never hide it.
            rel = winner_from_strings(row["first"], row["second"])
            conflict = False
            if rel is not None and row["flags"] != "unseen":
                svr = int(ctx["Svr"])
                derived = svr if rel == 1 else (2 if svr == 1 else 1)
                conflict = derived != w
            d["conflict"] = conflict
            pts.append(d)
        return {"match_id": self.match_id, "setup": self.setup,
                "points": pts, "score_now": sc.display,
                "over": sc.over, "next_server": sc.server,
                "conflicts": sum(p["conflict"] for p in pts)}

    def export_rows(self):
        st = self.state()
        if st["conflicts"]:
            raise ValueError(
                f"{st['conflicts']} point(s) contradict the score "
                f"replay - edit them (amber in the app) before export")
        rows = []
        for p in st["points"]:
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
            if (p["start_s"] and p["end_s"]
                    and float(p["start_s"]) < float(p["end_s"])):
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


# ---------------------------------------------------------------------------
# The HTTP half. Two flavors: the chart-along server (this session class)
# and the staged review flavor (reuses the frozen review.ReviewSession,
# serving this app's page instead of the bench's).
# ---------------------------------------------------------------------------


def _ui_bytes(server_mode, match_id=""):
    t = UI_PATH.read_text()
    inject = (f'<script>window.SERVER_MODE="{server_mode}";'
              f'window.MATCH_ID="{match_id}";</script>')
    return t.replace("<!doctype html>",
                     "<!doctype html>" + inject, 1).encode()


def make_chart_server(session, video_path, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/?") or self.path == "/":
                body = _ui_bytes("chart", session.match_id)
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/grammar.json":
                httpkit.send_file_ranged(
                    self, notation.GRAMMAR_PATH, "application/json")
            elif self.path == "/conformance.json":
                httpkit.send_file_ranged(
                    self, CONFORMANCE_PATH, "application/json")
            elif self.path == "/video":
                httpkit.send_file_ranged(self, video_path,
                                         "video/mp4")
            elif self.path == "/api/chart-state":
                httpkit.send_json(self, session.state())
            elif self.path == "/export/bundle":
                parts = []
                try:
                    rows = session.export_rows()
                except ValueError as e:
                    self.send_error(400, str(e))
                    return
                out = [",".join(EXPORT_FIELDS)]
                for r in rows:
                    out.append(",".join(
                        '"%s"' % r[c].replace('"', '""')
                        if any(x in r[c] for x in ',"\n') else r[c]
                        for c in EXPORT_FIELDS))
                parts.append(("points.csv", "\n".join(out) + "\n"))
                segs = ["clip,start_s,end_s"] + [
                    f"{s2['clip']},{s2['start_s']},{s2['end_s']}"
                    for s2 in session.segments()]
                parts.append(("segments.csv", "\n".join(segs) + "\n"))
                parts.append(("manifest.json", json.dumps(
                    {"match_id": session.match_id,
                     "setup": session.setup,
                     "points": len(session.points)}, indent=1)))
                body = "\n".join(f"--- FILE: {n} ---\n{c}"
                                 for n, c in parts).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            body = httpkit.read_json(self)
            if body is None:
                self.send_error(400, "invalid JSON")
                return
            if self.path == "/api/lint":
                httpkit.send_json(self, {"issues": notation.lint(
                    body.get("first", ""), body.get("second", ""))})
            elif self.path == "/api/event":
                session.append_event(body)
                httpkit.send_json(self, {"ok": True})
            elif self.path == "/api/point":
                try:
                    act = body.get("action")
                    if act == "add":
                        session.add_point(
                            body.get("first", ""),
                            body.get("second", ""),
                            notes=body.get("notes", ""),
                            winner=body.get("winner"),
                            start_s=body.get("start_s"),
                            end_s=body.get("end_s"))
                    elif act == "unseen":
                        session.insert_unseen(body["winner"])
                    elif act == "update":
                        idx = body.pop("idx")
                        body.pop("action")
                        session.update_point(idx, **body)
                    else:
                        raise ValueError(f"unknown action {act!r}")
                except (ValueError, KeyError, IndexError) as e:
                    httpkit.send_json(self, {"error": str(e)}, 400)
                    return
                httpkit.send_json(self, {"ok": True})
            else:
                self.send_error(404)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run_chart(match_id, video, setup=None, port=8766,
              open_browser=True):
    session = ChartSession(match_id, setup)
    httpd = make_chart_server(session, video, port)
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(f"chart '{match_id}' {len(session.points)} pts -> {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nsaved:", session.dir)


def run_staged(cfg, mode, name, seed=None, n=None, port=8766,
               open_browser=True):
    """The bench's session + routes, but the NEW page (review flavor).
    review.py stays frozen; we reuse its session object only."""
    from .review import ReviewSession
    session = ReviewSession(cfg, mode, name, seed=seed, n=n)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                body = _ui_bytes("review")
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/grammar.json":
                httpkit.send_file_ranged(
                    self, notation.GRAMMAR_PATH, "application/json")
            elif self.path == "/conformance.json":
                httpkit.send_file_ranged(
                    self, CONFORMANCE_PATH, "application/json")
            elif self.path == "/api/state":
                httpkit.send_json(self, session.state())
            elif self.path.startswith("/clip/"):
                stem = self.path[len("/clip/"):-len(".mp4")]
                if not stem.replace("_", "").isalnum():
                    self.send_error(404)
                    return
                httpkit.send_file_ranged(
                    self, session.cfg.clip_path(stem), "video/mp4")
            else:
                self.send_error(404)

        def do_POST(self):
            body = httpkit.read_json(self)
            if body is None:
                self.send_error(400, "invalid JSON")
                return
            if self.path == "/api/lint":
                httpkit.send_json(self, {"issues": notation.lint(
                    body.get("first", ""), body.get("second", ""))})
            elif self.path == "/api/event":
                session.append_event(body)
                httpkit.send_json(self, {"ok": True})
            elif self.path == "/api/accept":
                if "clip" not in body:
                    self.send_error(400, "missing clip")
                    return
                session.accept(body["clip"],
                               body.get("corrected_1st", ""),
                               body.get("corrected_2nd", ""),
                               body.get("notes", ""),
                               body.get("skip_reason", ""))
                session.append_event(
                    {"ts_ms": int(time.time() * 1000),
                     "row": body["clip"],
                     "event": ("skip" if body.get("skip_reason")
                               else "accept"), "payload": {}})
                httpkit.send_json(self, {"ok": True})
            else:
                self.send_error(404)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(f"{cfg.id} '{name}' [{mode}] -> {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nsession saved:", session.dir)


def emit_static(out_path):
    """One self-contained file: grammar + conformance baked in."""
    ui = UI_PATH.read_text()
    grammar = notation.GRAMMAR_PATH.read_text()
    conf = (CONFORMANCE_PATH.read_text()
            if CONFORMANCE_PATH.exists() else "[]")
    inject = ("<script>\n"
              f"window.GRAMMAR = {grammar};\n"
              f"window.CONFORMANCE = {conf};\n"
              "</script>\n")
    out_path = Path(out_path)
    result = ui.replace("<!doctype html>",
                        "<!doctype html>" + inject, 1)
    out_path.write_text(result)
    print(f"-> {out_path} ({out_path.stat().st_size} bytes, "
          f"open it from anywhere)")
