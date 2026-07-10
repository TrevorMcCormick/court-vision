"""STEP 0 — direction-semantics calibration, before building anything.

Our zone() maps landing court-x into ABSOLUTE thirds (1 = image-left).
MCP's convention (MatchChart instructions + quick-start guide) is
RECEIVER-relative and handedness-aware: 1 = to a right-hander's
forehand side (a lefty's backhand), 2 = middle, 3 = to a right-hander's
backhand. In our court frame (x=0 image-left doubles sideline, y=0 far
baseline) that predicts:

    far righty receiver:  ascending x -> 1,2,3   (their right = image-left)
    near righty receiver: descending x -> 3,2,1  (their right = image-right)
    lefty receivers flip.

This script tests every plausible mapping EMPIRICALLY on all rally
shots at aligned positions on length-matched points across the 4
matches where our chart committed a direction (landing_x present).
t1 is both-lefty, t2/t3/t4 all-righty — jointly they disambiguate
handedness-relative vs purely geometric conventions.

Receiver end is tested under two definitions:
    geo   = the half the landing was measured in (ly vs NET_Y)
    chain = the other end from the chart's striker chain

Serve zones (digits 4/5/6, wide/body/T — geometric, NOT handedness-
relative per the instructions) get the same treatment: current
serve_zone() vs its 4<->6 swap, on committed serve zones.

Usage:
    uv run experiments/dir_calibrate.py
"""

import csv
from pathlib import Path

from mcp_accept import mcp_point_tokens

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mcp"

MATCHES = ["t1", "t2", "t3", "t4"]
LABELS = {"t1": "t1 night(L)", "t2": "t2 ctrl(R)", "t3": "t3 clay(R)",
          "t4": "t4 grass(R)"}
LEFTY = {"t1": True, "t2": False, "t3": False, "t4": False}

W_C, L_C = 10.97, 23.77
NET_Y = L_C / 2
CENTER_X = W_C / 2
INSET = 1.372


def thirds(x, x0, x1):
    """0/1/2 by thirds of [x0, x1], clipped."""
    t = (x - x0) / (x1 - x0)
    t = min(max(t, 0.0), 1.0 - 1e-9)
    return int(t * 3)


def load_rows(t):
    mapd = {r["clip"]: r for r in csv.DictReader(open(DATA / f"{t}_mcp_map.csv"))}
    chart = ROOT / "outputs" / t / "charts_wasb"
    match = {r["clip"]: r for r in csv.DictReader(open(chart / "match_chart_v2.csv"))}
    rally, serve = [], []
    for clip, mc in match.items():
        m = mapd[clip]
        if m["status"] != "matched":
            continue
        played = m["second"] if m["second"].strip() else m["first"]
        mcp = mcp_point_tokens(played)
        shots = list(csv.DictReader(open(chart / f"chart2_{clip}.csv")))
        # our token list, with a parallel row list (None for inserted s?)
        toks, rows = [], []
        for sh in shots:
            toks.append("x")
            rows.append(sh)
        if shots and shots[0]["is_serve"] != "True":
            toks.insert(0, None)
            rows.insert(0, None)
        toks.append("e")
        rows.append(None)
        if len(mcp) != len(toks):
            continue                      # length-matched points only
        for mt, sh in zip(mcp, rows):
            if sh is None:
                continue
            lx = sh["landing_x"]
            if lx in ("", "None"):
                continue
            lx, ly = float(lx), float(sh["landing_y"])
            if sh["is_serve"] == "True":
                if mt[1] in "456" and sh["zone"] in ("4", "5", "6"):
                    serve.append((clip, lx, ly, mc.get("side", ""),
                                  mc["server_used"], mt[1], sh["zone"]))
            else:
                if len(mt) == 2 and mt[1] in "123":
                    rally.append((clip, lx, ly, sh["striker"], mt[1],
                                  sh["zone"]))
    return rally, serve


def dir_map(lx, receiver, lefty, mode, x0=0.0, x1=W_C):
    """mode: abs / abs_rev / recv / recv_rev / hand / recv_hand /
    recv_hand_rev — returns '1'/'2'/'3'."""
    z = thirds(lx, x0, x1)                 # 0 = image-left third
    flip = False
    if mode == "abs_rev":
        flip = True
    elif mode == "recv":
        flip = receiver == "near"
    elif mode == "recv_rev":
        flip = receiver == "far"
    elif mode == "hand":
        flip = lefty
    elif mode == "recv_hand":
        flip = (receiver == "near") != lefty
    elif mode == "recv_hand_rev":
        flip = (receiver == "far") != lefty
    if flip:
        z = 2 - z
    return str(z + 1)


MODES = [
    ("abs asc (CURRENT zone())", "abs"),
    ("abs desc (mirror)", "abs_rev"),
    ("recv-end mirror, no hand", "recv"),
    ("recv-end mirror inverted", "recv_rev"),
    ("handedness only", "hand"),
    ("recv-end + handedness  <- MCP spec", "recv_hand"),
    ("recv-end + hand, inverted", "recv_hand_rev"),
]


def main():
    data = {t: load_rows(t) for t in MATCHES}
    n = {t: len(data[t][0]) for t in MATCHES}
    print("=== rally direction calibration — committed landings at aligned "
          "positions, length-matched points ===")
    print(f"    n = " + ", ".join(f"{LABELS[t]} {n[t]}" for t in MATCHES)
          + f", total {sum(n.values())}\n")

    for recv_def in ("geo", "chain"):
        print(f"receiver end = {'landing half (geo)' if recv_def == 'geo' else 'striker-chain other end'}:")
        print(f"{'mapping':38}" + "".join(f"{LABELS[t]:>13}" for t in MATCHES)
              + f"{'overall':>13}")
        for label, mode in MODES:
            row = f"{label:38}"
            num = den = 0
            for t in MATCHES:
                a = b = 0
                for clip, lx, ly, striker, md, oz in data[t][0]:
                    recv = (("far" if ly < NET_Y else "near") if recv_def == "geo"
                            else ("far" if striker == "near" else "near"))
                    a += dir_map(lx, recv, LEFTY[t], mode) == md
                    b += 1
                row += f"{f'{a}/{b}':>13}"
                num, den = num + a, den + b
            pct = f" ({100 * num / den:.0f}%)" if den else ""
            row += f"{f'{num}/{den}':>9}{pct}"
            print(row)
        print()

    # width variants on the winning axis question: full vs singles thirds
    print("width variants (receiver end = geo):")
    print(f"{'mapping':38}" + "".join(f"{LABELS[t]:>13}" for t in MATCHES)
          + f"{'overall':>13}")
    for label, mode, x0, x1 in [
            ("recv+hand, full-width thirds", "recv_hand", 0.0, W_C),
            ("recv+hand, singles-width thirds", "recv_hand", INSET, W_C - INSET),
            ("abs asc, singles-width thirds", "abs", INSET, W_C - INSET)]:
        row = f"{label:38}"
        num = den = 0
        for t in MATCHES:
            a = b = 0
            for clip, lx, ly, striker, md, oz in data[t][0]:
                recv = "far" if ly < NET_Y else "near"
                a += dir_map(lx, recv, LEFTY[t], mode, x0, x1) == md
                b += 1
            row += f"{f'{a}/{b}':>13}"
            num, den = num + a, den + b
        pct = f" ({100 * num / den:.0f}%)" if den else ""
        row += f"{f'{num}/{den}':>9}{pct}"
        print(row)

    # landing-half split under the CURRENT mapping — the mirror, directly
    print("\ncurrent zone() split by landing half (geo receiver):")
    for t in MATCHES:
        far = [r for r in data[t][0] if r[2] < NET_Y]
        near = [r for r in data[t][0] if r[2] >= NET_Y]
        fa = sum(dir_map(lx, "far", False, "abs") == md
                 for _, lx, ly, _, md, _ in far)
        na = sum(dir_map(lx, "near", False, "abs") == md
                 for _, lx, ly, _, md, _ in near)
        print(f"  {LABELS[t]:12} far-half {fa}/{len(far)}   near-half {na}/{len(near)}")

    # ---- serves ----
    print("\n=== serve zone calibration — committed serve zones, "
          "length-matched points ===")
    print(f"{'mapping':38}" + "".join(f"{LABELS[t]:>13}" for t in MATCHES)
          + f"{'overall':>13}")

    def serve_zone(lx, deuce, server, swap=False):
        left = deuce if server == "near" else not deuce
        x0, x1 = (INSET, CENTER_X) if left else (CENTER_X, W_C - INSET)
        tt = (lx - x0) / (x1 - x0)
        tt = min(max(tt, 0.0), 1.0)
        toward_center = tt if not left else 1 - tt
        z = 6 if toward_center < 1 / 3 else (5 if toward_center < 2 / 3 else 4)
        if swap:
            z = {6: 4, 5: 5, 4: 6}[z]
        return str(z)

    for label, swap in [("serve_zone() as shipped", False),
                        ("serve_zone() 4<->6 swapped", True)]:
        row = f"{label:38}"
        num = den = 0
        for t in MATCHES:
            a = b = 0
            for clip, lx, ly, side, server, md, oz in data[t][1]:
                if side not in ("deuce", "ad"):
                    continue
                a += serve_zone(lx, side == "deuce", server, swap) == md
                b += 1
            row += f"{f'{a}/{b}':>13}"
            num, den = num + a, den + b
        pct = f" ({100 * num / den:.0f}%)" if den else ""
        row += f"{f'{num}/{den}':>9}{pct}"
        print(row)


if __name__ == "__main__":
    main()
