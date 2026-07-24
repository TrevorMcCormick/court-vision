"""Map the (charted-match x full YouTube video) intersection — the real
bottleneck resource for scaling the pipeline.

For each marquee corpus match (data/corpus/stage_candidates.json), search
YouTube via yt-dlp and classify the best hit as a full match, condensed,
highlight, or none — by duration + title. Writes data/video_catalog.json
(tracked; the corpus itself is gitignored). Full matches (>90 min) are the
gold: they align to the whole MCP chart, not just a curated subset.

    PYTHONPATH=. uv run python experiments/video_survey_scan.py
"""

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDS = json.loads((ROOT / "data" / "corpus" / "stage_candidates.json").read_text())
OUT = ROOT / "data" / "video_catalog.json"

HL = re.compile(r"highlight|condensed|extended|preview|best of|top \d|reaction|"
                r"analysis|press", re.I)
FULL_HINT = re.compile(r"full match|full final|complete match|entire match", re.I)


def surname(name):
    return name.split()[-1].lower() if name else ""


def search(query, n=5):
    try:
        out = subprocess.run(
            ["yt-dlp", f"ytsearch{n}:{query}", "--dump-json",
             "--flat-playlist", "--no-warnings"],
            capture_output=True, text=True, timeout=90).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({"id": d.get("id"), "title": d.get("title") or "",
                     "dur": d.get("duration")})
    return rows


def classify(rows, s1, s2):
    """Best hit for this match: prefer full, then condensed, then highlight."""
    relevant = [r for r in rows
                if s1 in r["title"].lower() or s2 in r["title"].lower()]
    if not relevant:
        return {"type": "none", "video_id": None, "duration_s": None, "title": ""}
    best = None
    for r in relevant:
        dur = r["dur"] or 0
        is_hl = bool(HL.search(r["title"]))
        if dur > 5400 and not is_hl:
            t = "full"
        elif FULL_HINT.search(r["title"]) and dur > 3600:
            t = "full"
        elif dur > 900 and not is_hl:
            t = "condensed"
        else:
            t = "highlight"
        rank = {"full": 3, "condensed": 2, "highlight": 1}[t]
        key = (rank, dur)
        if best is None or key > best[0]:
            best = (key, {"type": t, "video_id": r["id"],
                          "duration_s": r["dur"], "title": r["title"]})
    return best[1]


def main():
    catalog = []
    n_full = 0
    for i, c in enumerate(CANDS):
        yr = c["date"][:4]
        q = f'{c["p1"]} {c["p2"]} {c["tournament"]} {yr} full match'
        rows = search(q)
        res = classify(rows, surname(c["p1"]), surname(c["p2"]))
        rec = {**c, **res}
        catalog.append(rec)
        if res["type"] == "full":
            n_full += 1
        mark = {"full": "★FULL", "condensed": "~cond", "highlight": "·hl",
                "none": " none"}[res["type"]]
        dur = f"{(res['duration_s'] or 0)//60}m" if res["duration_s"] else "  "
        print(f"[{i+1:3}/{len(CANDS)}] {mark} {dur:>4} {c['date']} "
              f"{c['p1'].split()[-1]} v {c['p2'].split()[-1]} "
              f"({c['tournament']} {c['round']})", flush=True)
        OUT.write_text(json.dumps(catalog, indent=1))   # incremental save
        time.sleep(1.0)                                   # be gentle to YouTube
    print(f"\n=== {n_full} FULL matches / {len(CANDS)} candidates -> {OUT}")


if __name__ == "__main__":
    main()
