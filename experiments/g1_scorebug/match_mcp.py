"""Match the OCR'd score plateaus to the 88 MCP points, chronologically.

Both the plateau sweep and the MCP point list are chronological. We collapse
plateau fragments that share a score-key (camera cuts inside one point), then
greedy forward-match the observed key sequence onto the MCP key sequence,
resolving deuce recurrence by order (align.py's order_pass idea). Reports the
alignment hit rate and writes data/mcp/g1_clip_alignment.csv in the shipped
schema (clip,set1,set2,gm1,gm2,pts,server_bug,note).
"""
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
PLAT = Path(__file__).parent / "sweep_plateaus.csv"
MCP = ROOT / "data/mcp/20240608-W-Roland_Garros-F-Iga_Swiatek-Jasmine_Paolini_points.csv"

PTS_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


VALID = {"0", "15", "30", "40", "AD"}
# repair OCR slips seen in the sweep: "15"->"25"/"6"/"5"/"1", "30"->"320"/"35",
# "40"->"45"/"400", "15"->"150". Map any invalid token to its nearest legal pts.
REPAIR = {"25": "15", "26": "15", "20": "15", "6": "15", "5": "15", "1": "15",
          "150": "15", "10": "40", "35": "30", "320": "30", "300": "30",
          "3": "30", "45": "40", "400": "40", "48": "40", "4": "40", "00": "0"}


def _fix(tok):
    tok = (tok or "").strip().upper()
    if tok in VALID:
        return tok
    return REPAIR.get(tok, tok)


def norm_pts(a, b):
    """Canonical player-1-first points string from a raw (ptsA,ptsB) read,
    with OCR-slip repair to the legal tennis points vocabulary."""
    a = _fix(a); b = _fix(b)
    if a == "AD":
        return "AD-40"
    if b == "AD":
        return "40-AD"
    a = a if a in VALID else "0"
    b = b if b in VALID else "0"
    return f"{a}-{b}"


def mcp_key(r):
    setn = 1 if (int(r["Set1"]) + int(r["Set2"]) == 0) else 2
    pts = r["Pts"].upper()                    # server-first in MCP
    if r["Svr"] == "2" and "-" in pts:        # -> player-1-first
        l, _, rr = pts.partition("-"); pts = f"{rr}-{l}"
    return (setn, r["Gm1"], r["Gm2"], pts)


def obs_key(p):
    setn = 1 if not p["setA"] and not p["setB"] else 2
    gm1 = p["gmA"] or "0"; gm2 = p["gmB"] or "0"
    return (setn, gm1, gm2, norm_pts(p["ptsA"], p["ptsB"]))


def load_plateaus():
    rows = list(csv.DictReader(open(PLAT)))
    # collapse consecutive fragments sharing an obs_key; server by majority
    collapsed = []
    for p in rows:
        k = obs_key(p)
        if collapsed and collapsed[-1]["key"] == k:
            c = collapsed[-1]
            c["f1"] = p["f1"]; c["srv"].append(p["server"]); c["n"] += int(p["n"])
        else:
            collapsed.append({"key": k, "f0": p["f0"], "f1": p["f1"],
                              "srv": [p["server"]], "n": int(p["n"])})
    for c in collapsed:
        s = [x for x in c["srv"] if x in ("1", "2")]
        c["server"] = Counter(s).most_common(1)[0][0] if s else ""
    return collapsed


def greedy_match(obs, mcp):
    """LCS alignment between the MCP key-sequence and the observed key-sequence.

    Greedy-first-match is fragile: an OCR-glitch score equal to an earlier
    game-start key gets grabbed out of order and skips the correctly-ordered
    states after it. LCS finds the longest MONOTONIC matching, so noise
    insertions on either side are skipped and deuce recurrence resolves by
    order — exactly align.py's chronological-order principle, done optimally.
    """
    mk = [mcp_key(m) for m in mcp]
    ok = [o["key"] for o in obs]
    n, m = len(mk), len(ok)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if mk[i] == ok[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    matches = {}
    i = j = 0
    while i < n and j < m:
        if mk[i] == ok[j]:
            matches[i] = obs[j]; i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return matches


if __name__ == "__main__":
    mcp = list({r["Pt"]: r for r in csv.DictReader(open(MCP))}.values())
    obs = load_plateaus()
    # drop pre-match/intro obs: keep from first obs whose key matches MCP Pt<=3
    first_keys = {mcp_key(m) for m in mcp[:4]}
    start = next((i for i, o in enumerate(obs) if o["key"] in first_keys), 0)
    obs_live = obs[start:]
    matches = greedy_match(obs_live, mcp)

    print(f"MCP points: {len(mcp)}   collapsed obs states: {len(obs)} "
          f"(live from idx {start}: {len(obs_live)})")
    print(f"ALIGNED: {len(matches)}/{len(mcp)} MCP points matched to an OCR'd score state")
    # server agreement on matched points
    srv_ok = sum(1 for mi, o in matches.items() if o["server"] == mcp[mi]["Svr"])
    print(f"server-bug agreement on matched points: {srv_ok}/{len(matches)}")
    # show misses
    missed = [i for i in range(len(mcp)) if i not in matches]
    print(f"unmatched MCP Pts ({len(missed)}): {[mcp[i]['Pt'] for i in missed]}")

    # write alignment csv (shipped schema)
    out = ROOT / "data/mcp/g1_clip_alignment.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "set1", "set2", "gm1", "gm2", "pts", "server_bug", "note", "mcp_pt", "f0", "f1"])
        for mi in sorted(matches):
            m = mcp[mi]; o = matches[mi]
            setn, gm1, gm2, pts = o["key"]
            s1 = "1" if setn == 2 else "0"
            w.writerow([f"g1_point_{int(m['Pt']):02d}", s1, "0", gm1, gm2, pts,
                        o["server"], f"ocr-auto; mcp_pt {m['Pt']}", m["Pt"], o["f0"], o["f1"]])
    print(f"-> wrote {out}")
