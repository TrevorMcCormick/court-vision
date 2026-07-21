"""Regenerate docs/scorecard.html's data-driven tables from the eval run.

WHY: cv-17 shipped the scorecard hand-assembled, and hand-carried
numbers drift (this generator's first run caught four cells that were
still the 4-match decomposition figures on a page claiming
benchmark-v2). Everything between the <!-- gen:*-start/end --> marker
comments is emitted here from (a) a fresh CSV-only eval of every
staged match — seconds — and (b) data/confidence_lomo.json, the LOMO
sidecar `courtvision calibrate` writes (NOT recomputed here: the LOMO
loop is minutes). Everything outside the markers is hand-authored
prose and is preserved byte-for-byte.

Strict accuracy = exact token match vs MCP at aligned positions on
length-matched points, refusals ('?') count as wrong — the same bar
the page's threshold legend declares.

Fixed prose inside the regions ("never above 50%", "best on record",
"control") is guarded by assertions: if the data stops supporting the
sentence, the run fails loudly instead of publishing a lie.

Usage:  uv run python experiments/gen_scorecard.py
Idempotent: same data + same day -> second run is a zero diff.
"""

import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision import config, evaluate           # noqa: E402
from courtvision.config import ROOT                # noqa: E402

PAGE = ROOT / "docs" / "scorecard.html"
LOMO = ROOT / "data" / "confidence_lomo.json"
CONTROL = "t2"                 # the 5-point day control — excluded from ranges
NEW_MATCHES = ("t5", "t6", "t7")     # staged after the 4-match calibration
FEED = {"t1": "night hard", "t2": "day control", "t3": "clay RG",
        "t4": "grass WTA", "t5": "AO night", "t6": "USO", "t7": "Turin"}


def pct(x, dec=0):
    return f"{x:.{dec}%}"


def status(x):
    """The page's declared thresholds: 🟢 ≥80%, 🟡 45–80%, 🔴 <45%."""
    return "🟢" if x >= 0.80 else "🟡" if x >= 0.45 else "🔴"


def strict_components(records):
    """Positional token comparison on length-matched points; skip the
    rare point whose token lists still differ (chart missed the serve
    -> 's?' inserted). Denominators = MCP-committed positions only."""
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


def collect():
    per = {}
    for mid in config.match_ids():
        cfg = config.load(mid)
        tally, records = evaluate.evaluate(cfg, verbose=False)
        n_chart = sum(1 for _ in csv.DictReader(
            open(cfg.charts_dir / "match_chart_v2.csv")))
        per[mid] = {"tally": tally, "strict": strict_components(records),
                    "n_chart": n_chart}
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
                 "`uv run python -m courtvision calibrate` first "
                 "(slow: the full LOMO loop, several minutes)")
    lomo = json.loads(LOMO.read_text())
    per = collect()
    mids = list(per)
    assert set(mids) == set(lomo["per_match"]), \
        "eval matches != calibrate matches — rerun calibrate"

    def frac(key, mid=None):
        n, d = 0, 0
        for m in ([mid] if mid else mids):
            a, b = per[m]["tally"][key]
            n, d = n + a, d + b
        return n, d, (n / d if d else 0.0)

    def strict(key, mid=None):
        n, d = 0, 0
        for m in ([mid] if mid else mids):
            a, b = per[m]["strict"][key]
            n, d = n + a, d + b
        return n, d, (n / d if d else 0.0)

    # ---- the numbers ----
    srv = frac("server")[2]
    srv_best = max(frac("server", m)[2] for m in mids)
    rallies = {m: frac("rally_pm1", m)[2] for m in mids}
    ctrl_n = per[CONTROL]["tally"]["rally_pm1"][1]
    assert rallies[CONTROL] == 1.0, "the 'control' phrasing needs 100%"
    r_rng = [rallies[m] for m in mids if m != CONTROL]
    let_pool = strict("letters")[2]
    lets = [strict("letters", m)[2] for m in mids if m != CONTROL]
    dirs = strict("dirs")[2]
    d_att = (sum(per[m]["strict"]["dirs"][1] for m in mids)
             / sum(per[m]["strict"]["dirs_denom"] for m in mids))
    zone = strict("zone")[2]
    zone_new = max(frac("serve_zone", m)[2] for m in NEW_MATCHES)
    assert zone_new < 0.50, "'never above 50% committed-only' broke"
    ending = strict("ending")[2]
    n_pts = sum(per[m]["n_chart"] for m in mids)
    acc = {m: frac("accept", m)[2] for m in mids}
    # "best on record" counts full condensed matches — t1/t2 are the
    # small-n highlights reels (benchmark.md's reading notes)
    assert max((a, m) for m, a in acc.items()
               if frac("accept", m)[1] >= 40)[1] == "t6", \
        "'t6 best on record' broke"

    pm = lomo["per_match"]
    hp = lomo["pooled"]["precision_num"] / lomo["pooled"]["precision_den"]
    hc = lomo["pooled"]["coverage"]
    ch, cl = lomo["confusion"]["high"], lomo["confusion"]["low"]
    h_le2 = (ch["0-1"] + ch["2"]) / ch["total"]
    l_6p = cl["6+"] / cl["total"]
    worst = sorted((p["precision_num"] / p["precision_den"], m)
                   for m, p in pm.items() if p["precision_den"])[:2]
    hi = {m: (p["precision_num"] / p["precision_den"], p["coverage"])
          for m, p in pm.items() if p["precision_den"]}

    # ---- the regions ----
    header = (
        f"Numbers: <strong>benchmark-v2</strong> ({len(mids)} matches, "
        f"{lomo['n']} scored\npoints, truth corrections of 2026-07-20 "
        f"applied) — the source of truth\nis <a href=\"benchmark.md\">"
        f"benchmark.md</a>; if this page disagrees,\nbenchmark.md wins and "
        f"this page has a bug. The marked tables were\nregenerated "
        f"{date.today().isoformat()} by <code>experiments/gen_scorecard.py"
        f"</code>\nfrom the eval run + <code>data/confidence_lomo.json</code>"
        f" — they cannot\ndrift from the run.")

    trust = f"""<table>
<thead>
<tr>
<th>flag</th>
<th>measured guarantee (benchmark-v2, held-out)</th>
<th>your action</th>
<th>NOT promised</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>HIGH</strong></td>
<td>{pct(hp)} of flagged points within ≤5 token edits, at {pct(hc)}
coverage</td>
<td>Start from the draft; verify every token against video, especially
zones and endings</td>
<td>NOT "the point is right" (~{pct(h_le2)} are within 2 edits) — and
the promise is per-point edit count, <strong>not per-component</strong>:
a HIGH point's serve zone is still untrustworthy</td>
</tr>
<tr>
<td><strong>LOW</strong></td>
<td>none — {pct(l_6p)} are 6+ edits out</td>
<td>Chart from scratch; use the draft as a hint at most</td>
<td>nothing</td>
</tr>
</tbody>
</table>
<p>Per-feed honesty: on {worst[0][1]} ({FEED[worst[0][1]]}) and \
{worst[1][1]} ({FEED[worst[1][1]]}) HIGH delivers
{pct(worst[0][0])} and {pct(worst[1][0])} instead of {pct(hp)} — those \
feeds' failures are invisible
to the confidence signals. There is deliberately <strong>no sign-off
tier</strong>: it has failed held-out validation at every attempted
n.</p>"""

    components = f"""<table>
<thead>
<tr>
<th>component</th>
<th>status · strict accuracy (benchmark-v2)</th>
<th>your action</th>
</tr>
</thead>
<tbody>
<tr>
<td>Who served, which end</td>
<td>{status(srv)} {pct(srv)} pooled ({pct(srv_best)} best feed)</td>
<td>Spot-check; distrust on night feeds</td>
</tr>
<tr>
<td>Rally length (±1 shot)</td>
<td>{status(frac("rally_pm1")[2])} {pct(min(r_rng))}–{pct(max(r_rng))} \
by match ({pct(rallies[CONTROL])} on the
{ctrl_n}-point control)</td>
<td>Verify — expect missing shots at night (track holes) and on clay
(editor cuts)</td>
</tr>
<tr>
<td>Forehand/backhand letters</td>
<td>{status(let_pool)} {pct(let_pool)} pooled ({pct(min(lets))}–\
{pct(max(lets))} by match; t4's low
end is mostly gap #2's index shift, not bad reads)</td>
<td>Verify each; wrong mostly where the player box is smeared or
absent</td>
</tr>
<tr>
<td>Shot direction (1/2/3)</td>
<td>{status(dirs)} {pct(dirs)} when attempted ({pct(d_att)} attempted)</td>
<td>Verify; the strongest annotation since the 2026-07-10 rebuild</td>
</tr>
<tr>
<td>Serve zone (4/5/6)</td>
<td>{status(zone)} {pct(zone)} strict (pooled); never above 50%
committed-only on any new match</td>
<td><strong>Re-key always</strong> — near coin-flip</td>
</tr>
<tr>
<td>Point ending type</td>
<td>{status(ending)} {pct(ending)} strict (pooled)</td>
<td><strong>Re-key always</strong> — wrong more often than right</td>
</tr>
<tr>
<td>Faults / second serves</td>
<td>⛔ not attempted</td>
<td>Every draft is written as a first-serve point — restore faults
yourself</td>
</tr>
<tr>
<td>Point boundaries &amp; replays</td>
<td>🟢 {n_pts} points/{len(mids)} matches; replays &amp; alternate angles
rejected structurally</td>
<td>Trust; one known open case (a replayed let charts as one long
point)</td>
</tr>
</tbody>
</table>"""

    feeds = f"""<table>
<thead>
<tr>
<th>feed / condition</th>
<th>tier</th>
<th>evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>Stable wide day hard (t6 USO)</td>
<td><strong>supported</strong> — best on record</td>
<td>{pct(acc["t6"], 1)} acceptance, {pct(frac("server", "t6")[2])} server \
end, HIGH {pct(hi["t6"][0])} @
{pct(hi["t6"][1])}</td>
</tr>
<tr>
<td>Indoor hard (t7 Turin)</td>
<td><strong>supported</strong> — structure 🟢, annotations 🟡</td>
<td>rally {pct(rallies["t7"])}, server end {pct(frac("server", "t7")[2])}\
</td>
</tr>
<tr>
<td>Grass WTA (t4 Wimbledon)</td>
<td><strong>experimental</strong></td>
<td>stance-serve early-fire poisons letters/insertions (gap #2)</td>
</tr>
<tr>
<td>Night hard (t1, t5)</td>
<td><strong>experimental</strong></td>
<td>late ball acquisition; serve components degrade first</td>
</tr>
<tr>
<td>Clay RG (t3)</td>
<td><strong>experimental</strong></td>
<td>editor cuts into rallies — a footage ceiling, not a pipeline
bug</td>
</tr>
<tr>
<td>Anything unbenchmarked (other feeds, doubles, amateur video)</td>
<td><strong>unsupported</strong></td>
<td>never tested — no numbers, no promises</td>
</tr>
</tbody>
</table>"""

    html = PAGE.read_text()
    for name, content in (("header", header), ("trust", trust),
                          ("components", components), ("feeds", feeds)):
        html = splice(html, name, content)
    changed = html != PAGE.read_text()
    PAGE.write_text(html)
    print(f"{PAGE.relative_to(ROOT)}: {'updated' if changed else 'no change'}"
          f" — HIGH {pct(hp)} @ {pct(hc)}, server {pct(srv)}, "
          f"zone {pct(zone)}, letters {pct(let_pool)}, dirs {pct(dirs)}, "
          f"endings {pct(ending)}, {n_pts} pts / {lomo['n']} scored")


if __name__ == "__main__":
    main()
