"""Auto-ingestion: a corpus match record -> found video -> staged, labeled shots.

The point of this module is to make honest which links of the chain scale
by machine and which still need a human. The chain, with its verdict:

  find_match      corpus index + points lookup by match_id        AUTO
  extract_labels  MCP notation -> per-shot label rows             AUTO   (video-free)
  find_video      yt-dlp search, prefer condensed/extended cuts   AUTO
  fetch_video     download + fps-normalize into clips/            AUTO
  scaffold_yaml   write data/matches/<id>.yaml                    AUTO   (knobs left TODO)
  stage           fitcourt -> probe -> boundaries -> extract ->   BLOCKED on 3 by-eye
                  align                                           knobs (see below)
  readiness       report which staging knobs are set / missing    AUTO

The label side of a training pair is free and effectively infinite: the
Match Charting Project has ~1.85M charted points, and extract_labels turns
any one match into per-shot rows with zero video. What is NOT free is the
FEATURE side (ball track, player boxes, court coordinates), which only
exists once the video is staged and each clip is joined to its chart row.
Staging is where scale dies, on three knobs no code here can fill:

  1. court_detect.hull_lo/hull_hi + fit_lo/fit_hi  (fitcourt, probe)
     per-broadcast HSV court-colour band + a hand-picked static court-view
     frame window. Wrong band -> the court hull grabs crowd/apron and the
     homography fit is garbage.
  2. boundaries CFG entry                          (boundaries.py)
     per-broadcast score-bug crop windows, per-set eras, and a ref frame.
     These are hardcoded in-module, measured by eye per feed.
  3. data/mcp/<id>_clip_alignment.csv              (align)
     by-eye transcription of each clip's score bug (the join key).

extract_labels and the MCP-string walk reuse courtvision.mcp's vocabulary
(FH_SIDE / BH_SIDE / SHOT_CHARS) so the shot decode matches the rest of
the package exactly.

Usage:
    uv run python -m courtvision.ingest labels 20240608-W-Roland_Garros-F-Iga_Swiatek-Jasmine_Paolini
    uv run python -m courtvision.ingest find-video 20240608-W-...   # metadata only
    uv run python -m courtvision.ingest ingest 20240608-W-... [--download] [--stage]
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .mcp import FH_SIDE, BH_SIDE, SHOT_CHARS

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
MCP_DIR = ROOT / "data" / "mcp"
MATCH_DIR = ROOT / "data" / "matches"
CLIPS = ROOT / "clips"

DIRECTIONS = set("123")
DEPTHS = set("789")
ENDING_MARKS = set("*@#")
ERROR_LETTERS = set("nwdx!e")


# ---------------------------------------------------------------------------
# find_match  — AUTO
# ---------------------------------------------------------------------------

def _index_path(match_id):
    return CORPUS / ("charting-m-matches.csv" if "-M-" in match_id
                     else "charting-w-matches.csv")


def _points_paths(match_id):
    """The points shards that could hold this match, newest era first.

    The MCP shards split by DATE, not by the id, so we don't try to be
    clever: return all three for the match's gender and let the caller
    scan until it finds the id (a match is wholly in one shard)."""
    g = "m" if "-M-" in match_id else "w"
    return [CORPUS / f"charting-{g}-points-{era}.csv"
            for era in ("2020s", "2010s", "to-2009")]


def find_match(match_id):
    """Return (index_record dict, [point rows]) for a corpus match_id.

    Raises SystemExit with a clear message if the id or its points are
    absent (the ~45-match index/points gap the survey found)."""
    idx_path = _index_path(match_id)
    record = None
    for r in csv.DictReader(open(idx_path)):
        if r["match_id"] == match_id:
            record = r
            break
    if record is None:
        raise SystemExit(f"match_id not in {idx_path.name}: {match_id}")

    point_rows = []
    for pp in _points_paths(match_id):
        if not pp.exists():
            continue
        hit = False
        for r in csv.DictReader(open(pp)):
            if r["match_id"] == match_id:
                point_rows.append(r)
                hit = True
            elif hit:
                break          # rows are contiguous per match; stop after
        if point_rows:
            break
    if not point_rows:
        raise SystemExit(f"no point rows for {match_id} in any shard "
                         "(indexed but unpointed — the corpus gap)")
    return record, point_rows


# ---------------------------------------------------------------------------
# extract_labels  — AUTO, video-free (the part that scales today)
# ---------------------------------------------------------------------------

def _side(letter):
    return "fh" if letter in FH_SIDE else "bh" if letter in BH_SIDE else "?"


def walk_shots(played):
    """Walk one PLAYED serve string into ordered shot dicts.

    shot 0 is the serve; rally shots follow. Each rally shot carries its
    letter, side, direction (1/2/3 or ?), depth (7/8/9 or ?), and whether
    it ended the point and how (* winner, n/w/d/x error kind)."""
    shots = []
    if not played:
        return shots
    serve = played[0] if played[0] in "0456" else "?"
    shots.append({"is_serve": True, "letter": "", "side": "serve",
                  "serve_zone": serve if serve in "456" else "?",
                  "dir": "", "depth": "", "ending": "", "err": ""})
    i = 1
    while i < len(played):
        c = played[i]
        if c in SHOT_CHARS:
            direction = depth = "?"
            j = i + 1
            while j < len(played) and played[j] not in SHOT_CHARS:
                if played[j] in DIRECTIONS and direction == "?":
                    direction = played[j]
                elif played[j] in DEPTHS and depth == "?":
                    depth = played[j]
                j += 1
            shots.append({"is_serve": False, "letter": c, "side": _side(c),
                          "serve_zone": "", "dir": direction, "depth": depth,
                          "ending": "", "err": ""})
            i = j
        else:
            i += 1
    # ending: the trailing mark and any error letter just before it
    end = ""
    err = ""
    tail = played.rstrip()
    if tail and tail[-1] in ENDING_MARKS:
        end = tail[-1]
        body = tail[:-1]
        if end in "@#" and body and body[-1] in ERROR_LETTERS:
            err = body[-1]
    if shots:
        shots[-1]["ending"] = end
        shots[-1]["err"] = err
    return shots


def extract_labels(match_id, point_rows, out_path=None):
    """MCP point rows -> per-shot label rows. AUTO, needs no video.

    One row per stroke: which point, shot index, which player struck it
    (server on even rally index, returner on odd), the serve zone or the
    shot letter/side/direction/depth, and the point-ending mark. These
    are the LABEL half of a training pair; the FEATURE half (ball, boxes,
    court coords) only exists after the video is staged and aligned."""
    out_path = out_path or (MCP_DIR / f"{match_id}_shots.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["match_id", "pt", "set1", "set2", "gm1", "gm2", "pts", "svr",
            "served", "shot_idx", "striker", "is_serve", "serve_zone",
            "letter", "side", "dir", "depth", "ending", "err", "pt_winner"]
    n_shots = 0
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in point_rows:
            first, second = r.get("1st", ""), r.get("2nd", "")
            played = second.strip() if second.strip() else first.strip()
            served = "2nd" if second.strip() else "1st"
            svr = r.get("Svr", "")
            for k, sh in enumerate(walk_shots(played)):
                # serve = shot 0 struck by server; rally alternates
                striker = svr if k % 2 == 0 else ("2" if svr == "1" else "1")
                wr.writerow({
                    "match_id": match_id, "pt": r.get("Pt", ""),
                    "set1": r.get("Set1", ""), "set2": r.get("Set2", ""),
                    "gm1": r.get("Gm1", ""), "gm2": r.get("Gm2", ""),
                    "pts": r.get("Pts", ""), "svr": svr, "served": served,
                    "shot_idx": k, "striker": striker,
                    "is_serve": sh["is_serve"], "serve_zone": sh["serve_zone"],
                    "letter": sh["letter"], "side": sh["side"],
                    "dir": sh["dir"], "depth": sh["depth"],
                    "ending": sh["ending"], "err": sh["err"],
                    "pt_winner": r.get("PtWinner", "")})
                n_shots += 1
    return out_path, n_shots


# ---------------------------------------------------------------------------
# find_video / fetch_video  — AUTO
# ---------------------------------------------------------------------------

def _queries(record):
    p1, p2 = record["Player 1"], record["Player 2"]
    yr = record["Date"][:4]
    tourn = record["Tournament"]
    rnd = record.get("Round", "")
    base = f"{p1} vs {p2} {yr} {tourn}"
    return [f"{base} condensed match",
            f"{base} extended highlights",
            f"{base} {rnd} highlights"]


def _score_candidate(entry):
    """Prefer condensed/full over short highlight, and longer durations —
    a chart's 150-400 points align far better to a 20-40 min condensed
    match than to a 2-5 min highlight reel."""
    title = (entry.get("title") or "").lower()
    dur = entry.get("duration") or 0
    s = 0.0
    if "condensed" in title or "full match" in title:
        s += 100
    if "extended" in title:
        s += 40
    if "highlight" in title:
        s += 10
    # duration sweet spot 20-45 min
    if 1200 <= dur <= 2700:
        s += 60
    elif 600 <= dur < 1200:
        s += 30
    elif dur > 2700:
        s += 45
    elif dur < 180:
        s -= 20
    return s


def find_video(record, per_query=4):
    """yt-dlp metadata search across condensed/extended/highlight queries.

    Returns candidates sorted best-first. Metadata only — no bytes."""
    seen, cands = set(), []
    for q in _queries(record):
        try:
            out = subprocess.run(
                ["yt-dlp", "--dump-json", "--flat-playlist",
                 "--no-warnings", f"ytsearch{per_query}:{q}"],
                capture_output=True, text=True, timeout=90)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  query failed ({q!r}): {e}", file=sys.stderr)
            continue
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = e.get("id")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            cands.append({
                "id": vid, "title": e.get("title"),
                "duration": e.get("duration"),
                "url": e.get("url") or f"https://www.youtube.com/watch?v={vid}",
                "channel": e.get("channel") or e.get("uploader"),
                "query": q, "score": 0.0})
    for c in cands:
        c["score"] = _score_candidate(c)
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def fetch_video(url, dest, quality="worst"):
    """Download a video into clips/ (gitignored). quality='worst' keeps
    a proof download tiny; use 'best' for a real staging run."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = "worst" if quality == "worst" else "bestvideo+bestaudio/best"
    subprocess.run(
        ["yt-dlp", "-f", fmt, "--no-warnings", "-o", str(dest), url],
        check=True)
    return dest


def normalize_fps(src, dst, fps=30):
    """Re-encode to a constant fps (the staging stages assume it)."""
    src, dst = Path(src), Path(dst)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-r", str(fps), "-an", str(dst)], check=True)
    return dst


# ---------------------------------------------------------------------------
# scaffold_yaml  — AUTO (manual knobs written as loud TODOs)
# ---------------------------------------------------------------------------

def scaffold_yaml(match_id, record, video_rel, short_id, out_path=None):
    """Write data/matches/<short_id>.yaml with everything the machine can
    fill, and the three by-eye knobs as TODOs. Never overwrites."""
    out_path = out_path or (MATCH_DIR / f"{short_id}.yaml")
    if out_path.exists():
        return out_path, False
    p1, p2 = record["Player 1"], record["Player 2"]
    surface = record.get("Surface", "?")
    text = f"""# {short_id} — {p1} vs {p2}, {record.get('Tournament','')} \
{record.get('Round','')} {record['Date'][:4]} ({surface}) [auto-scaffold]
id: {short_id}
title: "{short_id} {surface} {record.get('Tournament','')}"
match: "{p1}-{p2}, {record.get('Tournament','')} {record.get('Round','')} \
{record['Date'][:4]} ({surface})"
video: {video_rel}
clips_dir: clips/points_{short_id}
out_dir: outputs/{short_id}
ball_dir: ball_wasb
players_dir: players
charts_dir: charts_wasb

lefty: {{near: false, far: false}}   # TODO verify each player's hand

staging:
  lock_serve: true
  serve_zone_requires_side: true
  near_ending_fill: true
  coda_report: false

# TODO(MANUAL 1/3) court_detect — per-broadcast HSV court band + a static
# court-view fit window. fitcourt/probe cannot run until these are set by
# eye from real frames (see docs/USAGE.md). Without them staging is blocked.
# court_detect:
#   hull_lo: [H, S, V]
#   hull_hi: [H, S, V]
#   fit_lo: 0
#   fit_hi: 0

# TODO(MANUAL 2/3) a CFG entry keyed "{short_id}" in courtvision/boundaries.py
# (score-bug crop windows, per-set eras, ref_frame) — measured by eye.

serve_detect:
  variant: {"ball" if surface.lower() == "clay" else "stance"}
  center_tol_m: 4.0

players_detect:
  near_top_m: 11.885
  far_bottom_m: 11.885

eval:
  title: "{short_id}: {surface}, {record.get('Tournament','')}"
  mcp_points: data/mcp/{match_id}_points.csv
  mcp_match_id: "{match_id}"
  # TODO(MANUAL 3/3) mcp_map + alignment come from a by-eye score-bug
  # transcription per clip (data/mcp/{short_id}_clip_alignment.csv), then
  # `align`. start_end / set_priors / tiebreak_states need the parity pass.
  # mcp_map: data/mcp/{short_id}_mcp_map.csv
  # alignment: data/mcp/{short_id}_clip_alignment.csv
  start_end: far
  set_priors: {{}}
  tiebreak_states: []
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path, True


# ---------------------------------------------------------------------------
# staging readiness  — AUTO (reports the wall; does not run heavy compute)
# ---------------------------------------------------------------------------

def staging_readiness(short_id):
    """Inspect a scaffolded match: which of the three by-eye staging knobs
    are filled, i.e. how far `stage` could get before a human is needed."""
    import yaml as _yaml
    from . import boundaries
    yml_path = MATCH_DIR / f"{short_id}.yaml"
    report = {"court_detect": False, "boundaries_cfg": False,
              "clip_alignment": False}
    if not yml_path.exists():
        return report
    raw = _yaml.safe_load(yml_path.read_text())
    cd = raw.get("court_detect") or {}
    report["court_detect"] = all(k in cd for k in
                                 ("hull_lo", "hull_hi", "fit_lo", "fit_hi"))
    report["boundaries_cfg"] = short_id in boundaries.CFG
    ev = raw.get("eval") or {}
    align_csv = ev.get("alignment")
    report["clip_alignment"] = bool(align_csv and (ROOT / align_csv).exists())
    report["ready_to_stage"] = all(report.values())
    return report


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def ingest(match_id, short_id=None, download=False, video_url=None,
           video_quality="worst", stage=False):
    """Drive the automatable chain and print an honest readiness report."""
    short_id = short_id or ("g_" + match_id.split("-")[0])
    record, point_rows = find_match(match_id)
    print(f"[find_match] {match_id}: {len(point_rows)} charted points")

    # points CSV (align/eval input) + per-shot label rows
    pts_out = MCP_DIR / f"{match_id}_points.csv"
    pts_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pts_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
        wr.writeheader()
        wr.writerows(point_rows)
    shots_out, n_shots = extract_labels(match_id, point_rows)
    print(f"[extract_labels] {n_shots} shot-label rows -> {shots_out}")

    cands = find_video(record)
    print(f"[find_video] {len(cands)} candidates")
    for c in cands[:5]:
        d = int(c["duration"] or 0)
        dur = f"{d//60}:{d%60:02d}" if d else "?"
        print(f"    score {c['score']:>5.0f}  {dur:>6}  {c['title']}")
    best = cands[0] if cands else None

    video_rel = None
    if download and (video_url or best):
        url = video_url or best["url"]
        raw = CLIPS / f"{short_id}_raw.mp4"
        fetch_video(url, raw, quality=video_quality)
        norm = CLIPS / f"{short_id}_30fps.mp4"
        normalize_fps(raw, norm, fps=30)
        video_rel = f"clips/{norm.name}"
        print(f"[fetch_video] {url} -> {norm}")

    yml, wrote = scaffold_yaml(match_id, record,
                               video_rel or "clips/TODO_download.mp4",
                               short_id)
    print(f"[scaffold_yaml] {'wrote' if wrote else 'exists'} {yml}")

    rdy = staging_readiness(short_id)
    print(f"[readiness] {rdy}")
    if stage:
        print("[stage] BLOCKED: the 3 by-eye knobs above must be filled "
              "before fitcourt/probe/boundaries/align can run.")
    return {"match_id": match_id, "short_id": short_id,
            "n_points": len(point_rows), "n_shots": n_shots,
            "video_candidates": cands, "readiness": rdy}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="courtvision.ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("labels", help="MCP notation -> per-shot label rows")
    p.add_argument("match_id")
    p = sub.add_parser("find-video", help="yt-dlp search (metadata only)")
    p.add_argument("match_id")
    p = sub.add_parser("ingest", help="full automatable chain + readiness")
    p.add_argument("match_id")
    p.add_argument("--short-id", default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--url", default=None)
    p.add_argument("--quality", default="worst", choices=["worst", "best"])
    p.add_argument("--stage", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "labels":
        record, rows = find_match(args.match_id)
        out, n = extract_labels(args.match_id, rows)
        print(f"{n} shot-label rows -> {out}")
    elif args.cmd == "find-video":
        record, _ = find_match(args.match_id)
        for c in find_video(record):
            d = int(c["duration"] or 0)
            dur = f"{d//60}:{d%60:02d}" if d else "?"
            print(f"score {c['score']:>5.0f}  {dur:>6}  [{c['id']}]  {c['title']}")
    elif args.cmd == "ingest":
        ingest(args.match_id, short_id=args.short_id, download=args.download,
               video_url=args.url, video_quality=args.quality,
               stage=args.stage)


if __name__ == "__main__":
    main()
