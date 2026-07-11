"""cv-18 analysis — what the stopwatch and the corrections say.

Consumes review sessions (events.jsonl + corrected.csv), the draft
exports, and MCP truth (the *_mcp_map.csv 'first'/'second' columns —
this is the ONLY review-side module allowed to read truth). Produces
the cv-18 tables:

  timing     s/point: cold-A vs review(HIGH) vs review(low) vs cold-B
  accuracy   Trevor-vs-MCP token edits per arm (cold and corrected)
  fault      Trevor-vs-MCP agreement on 2nd-serve faults (per arm)
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
                  classify_sub, tok_kind, FH_SIDE, BH_SIDE, SHOT_CHARS)

MAX_GAP_MS = 180_000


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
    shot-side projection (FH_SIDE/BH_SIDE/SHOT_CHARS) is imported
    from mcp.py so it cannot drift from the frozen benchmark's
    projection. The ending is peeled off FIRST — the final '?' of a
    string like 's?f2?3?' is the unknown ending, not an extra shot.
    Ambiguous '?' runs (e.g. 'b??') resolve greedily; fine for
    histograms."""
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
    shotset = SHOT_CHARS | {"?"}
    while i < len(body):
        c = body[i]
        if c in shotset:
            letter = "f" if c in FH_SIDE else "b" if c in BH_SIDE else "?"
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
    if not xs:
        return 0.0
    m = len(xs) // 2
    if len(xs) % 2:
        return round(xs[m], 1)
    return round((xs[m - 1] + xs[m]) / 2, 1)


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
                              draft_point_tokens(corrected_played(k)))
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
        if (r["confidence"] != "high" or not t or not k
                or k["flags"].startswith("skipped")):
            continue
        tt = mcp_point_tokens(truth_played(t))
        dd = draft_point_tokens(r["1st"])
        if token_levenshtein(tt, dd) <= 5:
            continue
        kk = draft_point_tokens(corrected_played(k))
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
        kk = draft_point_tokens(corrected_played(k))
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
