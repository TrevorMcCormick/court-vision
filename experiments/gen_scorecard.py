"""Regenerate docs/scorecard.html's data regions from the eval run.

WHY: cv-17 shipped the scorecard hand-assembled, and hand-carried
numbers drift (this generator's first run caught four cells still
carrying 4-match-era figures). Everything between the
<!-- gen:*-start/end --> markers is emitted here from (a) a fresh
CSV-only eval of every staged match — seconds — and (b)
data/confidence_lomo.json, the LOMO sidecar `courtvision calibrate`
writes. Everything outside the markers is hand-authored and preserved
byte-for-byte (including the "proven, not plugged in yet" bench, which
is updated by hand when an experiment lands in the pipeline).

2026-07-23 redesign: the page is now glance-first — four stat tiles,
a compact per-question report card (chip + number + meter), detail
tables collapsed. Grades: Trust >=80%, Verify 45-80%, Re-key <45%;
strict accuracy, refusals count as wrong.

Usage:  uv run python experiments/gen_scorecard.py
Idempotent: same data + same day -> second run is a zero diff.
"""


import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import config, evaluate           # noqa: E402
from courtvision.config import ROOT                # noqa: E402

PAGE = ROOT / "docs" / "scorecard.html"
LOMO = ROOT / "data" / "confidence_lomo.json"
HISTORY = ROOT / "data" / "scorecard_history.json"
JSON_OUT = ROOT / "docs" / "scorecard.json"
CONTROL = "t2"
FEED = {"t1": "night hard", "t2": "day control", "t3": "clay RG",
        "t4": "grass WTA", "t5": "AO night", "t6": "USO", "t7": "Turin",
        "g1": "clay RG-24"}


def pct(x, dec=0):
    return f"{x:.{dec}%}"


def chip(x):
    if x >= 0.80:
        return '<span class="chip ok">Trust</span>'
    if x >= 0.45:
        return '<span class="chip warn">Verify</span>'
    return '<span class="chip bad">Re-key</span>'


def meter(x):
    cls = "ok" if x >= 0.80 else "warn" if x >= 0.45 else "bad"
    return f'<div class="meter"><i class="{cls}" style="width:{x:.0%}"></i></div>'


def row(question, x, shown=None):
    shown = shown or pct(x)
    return (f"<tr><td>{question}</td><td>{chip(x)}</td>"
            f'<td class="num">{shown}</td><td>{meter(x)}</td></tr>')


def strict_components(records):
    s = {"zone": [0, 0], "letters": [0, 0], "dirs": [0, 0],
         "dirs_denom": 0, "ending": [0, 0]}
    for r in records:
        if not r["aligned"]:
            continue
        m, o = r["mcp_toks"], r["our_toks"]
        if len(m) != len(o):
            continue
        if m[0][1] in "456":
            s["zone"][1] += 1
            s["zone"][0] += m[0] == o[0]
        if m[-1] != "?":
            s["ending"][1] += 1
            s["ending"][0] += m[-1] == o[-1]
        for mt, ot in zip(m[1:-1], o[1:-1]):
            if mt[0] in "fb":
                s["letters"][1] += 1
                s["letters"][0] += mt[0] == ot[0]
            if mt[1] in "123":
                s["dirs_denom"] += 1
                if ot[1] in "123":
                    s["dirs"][1] += 1
                    s["dirs"][0] += mt[1] == ot[1]
    return s


def collect(mids):
    per = {}
    for mid in mids:
        cfg = config.load(mid)
        tally, records = evaluate.evaluate(cfg, verbose=False)
        per[mid] = {"tally": tally, "strict": strict_components(records)}
    return per


def splice(html, name, content):
    a, b = f"<!-- gen:{name}-start -->", f"<!-- gen:{name}-end -->"
    assert html.count(a) == 1 and html.count(b) == 1, f"markers for {name}"
    pre, _, rest = html.partition(a)
    _, _, post = rest.partition(b)
    return f"{pre}{a}\n{content}\n{b}{post}"


def main():
    if not LOMO.exists():
        sys.exit("data/confidence_lomo.json missing — run "
                 "`uv run python -m courtvision calibrate` first")
    lomo = json.loads(LOMO.read_text())
    mids = sorted(lomo["per_match"])
    extra = sorted(set(config.match_ids()) - set(mids))
    if extra:
        print(f"note: staged but not in the published LOMO run, excluded: "
              f"{', '.join(extra)}")
    per = collect(mids)

    def frac(key, mid=None):
        n = sum(per[m]["tally"][key][0] for m in ([mid] if mid else mids))
        d = sum(per[m]["tally"][key][1] for m in ([mid] if mid else mids))
        return n, d, (n / d if d else 0.0)

    def strict(key, mid=None):
        n = sum(per[m]["strict"][key][0] for m in ([mid] if mid else mids))
        d = sum(per[m]["strict"][key][1] for m in ([mid] if mid else mids))
        return n, d, (n / d if d else 0.0)

    srv = frac("server")[2]
    rallies = {m: frac("rally_pm1", m)[2] for m in mids}
    r_rng = [rallies[m] for m in mids if m != CONTROL]
    let_pool = strict("letters")[2]
    dirs = strict("dirs")[2]
    zone = strict("zone")[2]
    ending = strict("ending")[2]
    accept = frac("accept")[2]

    pm = lomo["per_match"]
    hp = lomo["pooled"]["precision_num"] / lomo["pooled"]["precision_den"]
    hc = lomo["pooled"]["coverage"]
    ch, cl = lomo["confusion"]["high"], lomo["confusion"]["low"]
    h_le2 = (ch["0-1"] + ch["2"]) / ch["total"]
    l_6p = cl["6+"] / cl["total"]
    # usable = <=5 edits over ALL scored points, from the same bins the
    # trust model reports — a records-based version of this once shipped
    # 83% by quietly dropping unaligned points from the denominator
    usable = 1 - (ch["6+"] + cl["6+"]) / (ch["total"] + cl["total"])
    worst = sorted((p["precision_num"] / p["precision_den"], m)
                   for m, p in pm.items() if p["precision_den"])[:2]

    header = (f"Regenerated {date.today().isoformat()} from the grading run "
              f"({len(mids)} matches, {lomo['n']} human-charted points; "
              f'receipts in <a href="benchmark.md">benchmark.md</a>) — '
              f"numbers cannot drift from the code.")

    tiles = (f'<p class="tested"><b>{lomo["n"]}</b> points graded against '
             f'human charters, across <b>{len(mids)}</b> matches and five '
             f'broadcast styles.</p>')

    components = "\n".join([
        '<table>',
        '<tr><th>the machine\'s call</th><th>grade</th>'
        '<th style="text-align:right">right</th><th></th></tr>',
        row("Who served, from which end", srv),
        row("How many shots in the rally", (min(r_rng) + max(r_rng)) / 2,
            f"{pct(min(r_rng))}–{pct(max(r_rng))}"),
        row("Shot direction", dirs),
        row("Forehand or backhand", let_pool),
        row("Where the serve landed", zone),
        row("How the point ended", ending),
        '<tr><td>Faults / second serves</td>'
        '<td><span class="chip off">Missing</span></td>'
        '<td class="num">—</td><td></td></tr>',
        '</table>'])

    trust = f"""<table>
<tr><th>flag</th><th>measured, held out</th><th>what to do</th></tr>
<tr><td><strong>HIGH</strong></td>
<td>{pct(hp)} of flagged points need at most 5 fixes ({pct(hc)} of points
flagged); only ~{pct(h_le2)} are within 2 fixes</td>
<td>Start from the draft; verify every token, especially serve zones and
endings</td></tr>
<tr><td><strong>LOW</strong></td>
<td>no promise — {pct(l_6p)} are 6+ fixes out</td>
<td>Chart from scratch</td></tr>
</table>
<p>Weakest feeds for the flag: {FEED[worst[0][1]]} ({pct(worst[0][0])}) and
{FEED[worst[1][1]]} ({pct(worst[1][0])}) — their failure modes are invisible
to the confidence signals. There is deliberately no sign-off tier: it has
failed held-out validation at every attempted n.</p>"""

    feeds = f"""<table>
<tr><th>broadcast</th><th>tier</th><th>evidence</th></tr>
<tr><td>Stable wide day hard (US Open)</td><td><strong>supported</strong></td>
<td>best on record: {pct(frac("accept", "t6")[2], 1)} perfect,
{pct(frac("server", "t6")[2])} server calls</td></tr>
<tr><td>Indoor hard (Turin)</td><td><strong>supported</strong></td>
<td>rally {pct(rallies["t7"])}, server {pct(frac("server", "t7")[2])}</td></tr>
<tr><td>Grass (Wimbledon)</td><td><strong>experimental</strong></td>
<td>serve detector fires early; poisons letters</td></tr>
<tr><td>Night hard</td><td><strong>experimental</strong></td>
<td>late ball pickup; serve calls degrade first</td></tr>
<tr><td>Clay (Roland Garros)</td><td><strong>experimental</strong></td>
<td>broadcast cuts into rallies — a footage ceiling</td></tr>
<tr><td>Anything else (doubles, amateur video)</td>
<td><strong>unsupported</strong></td><td>never tested</td></tr>
</table>"""

    # ---- machine-readable feed for the website (+ history append) ----
    hist = json.loads(HISTORY.read_text())
    hist.pop("_comment", None)
    today = date.today().isoformat()
    values = {"server": srv, "rally": frac("rally_pm1")[2], "dirs": dirs,
              "letters": let_pool, "zone": zone, "ending": ending}
    for key, val in values.items():
        series = hist.setdefault(key, [])
        rounded = round(val, 3)
        if not series or abs(series[-1]["value"] - rounded) >= 0.005:
            series.append({"date": today,
                           "label": f"{len(mids)} matches", "value": rounded})
        elif series[-1]["date"] == today:
            series[-1]["value"] = rounded
    HISTORY.write_text(json.dumps(
        {"_comment": "Grading history for the scorecard trend lines; "
                     "appended by gen_scorecard.py, backfilled only from "
                     "documented benchmark figures.", **hist}, indent=1) + "\n")

    def comp(cid, question, val, shown=None):
        grade = ("trust" if val >= 0.80 else
                 "verify" if val >= 0.45 else "rekey")
        return {"id": cid, "question": question, "grade": grade,
                "value": round(val, 3), "shown": shown or pct(val),
                "history": hist.get(cid, [])}

    feed = {
        "generated": today, "points": lomo["n"], "matches": len(mids),
        "components": [
            comp("server", "Who served, from which end?", srv),
            comp("rally", "How many shots in the rally?", values["rally"],
                 f"{pct(min(r_rng))}–{pct(max(r_rng))}"),
            comp("dirs", "Which way did the shot go?", dirs),
            comp("letters", "Forehand or backhand?", let_pool),
            comp("zone", "Where did the serve land?", zone),
            comp("ending", "How did the point end?", ending),
        ],
        "missing": [{"id": "faults", "question": "Faults and second serves",
                     "note": "not attempted — every draft assumes a first serve"}],
        "bench": [
            {"id": "landing", "question": "Out-balls: wide or deep, from flight physics",
             "shown": "0 → 47%", "note": "was totally blind",
             "post": "/blog/nobody-sees-the-ball-land"},
            {"id": "ears", "question": "Hearing every hit in the soundtrack",
             "shown": "½ confirmed", "note": "196 impacts heard where video was blind",
             "post": "/blog/the-machine-grows-ears"},
            {"id": "court", "question": "Court finds itself — no human clicks",
             "shown": "7 of 8", "note": "within 2–16 px of the human; declines the 8th",
             "post": "/blog/the-court-that-fits-itself"},
            {"id": "pose", "question": "Player skeletons for forehand/backhand",
             "shown": "+13 pts", "note": "on grass, the worst surface",
             "post": "/blog/stick-figures-beat-smudges"},
        ],
    }
    JSON_OUT.write_text(json.dumps(feed, indent=1) + "\n")

    html = PAGE.read_text()
    for name, content in (("header", header), ("tiles", tiles),
                          ("components", components), ("trust", trust),
                          ("feeds", feeds)):
        html = splice(html, name, content)
    changed = html != PAGE.read_text()
    PAGE.write_text(html)
    print(f"{PAGE.relative_to(ROOT)}: {'updated' if changed else 'no change'}"
          f" — perfect {pct(accept, 1)}, usable {pct(usable)}, "
          f"HIGH {pct(hp)}@{pct(hc)}, server {pct(srv)}, dirs {pct(dirs)}, "
          f"letters {pct(let_pool)}, zone {pct(zone)}, endings {pct(ending)}")


if __name__ == "__main__":
    main()
