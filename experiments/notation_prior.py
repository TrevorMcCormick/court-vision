"""TRACK B — the VIDEO-FREE notation prior.

How much of a tennis chart is predictable from CONTEXT ALONE — no pixels,
no ball track, no player boxes — just the Match Charting Project notation
of the rally so far?

We train two tabular models on ~1.85M-point MCP corpus (sampled) and score
them on a MATCH-DISJOINT held-out set (no match is ever in both train and
test — the honesty law for this repo):

  TASK 1  ending type of the point   -> winner(*) / net(n) / wide(w) / deep(d)
  TASK 2  the NEXT rally shot's side -> forehand(fh) / backhand(bh)
  TASK 3  the NEXT rally shot's direction -> 1 / 2 / 3 (bonus; feeds wide/deep)

Everything is video-free: features are notation context (prior shot sides
& directions, rally position, striker role, server, serve zone, score, tour,
surface). No pixel is touched. Deterministic (seed 42); match include/split
is a stable hash of match_id so a rerun reproduces the exact split.

Leakage control:
  * train/test are disjoint by match_id (hash split, verified in-code).
  * TASK 1 never sees the ending mark or the error letter as a feature; it
    may see the ending SHOT's side/direction (that IS the prior: "a backhand
    to the open court usually ends it").
  * TASK 2/3 predicting shot i see only shots 0..i-1.

Run:
  uv run --with scikit-learn --with pandas python experiments/notation_prior.py

Writes outputs/diag/notation_prior.json and outputs/diag/notation_prior.md.
Additive only — imports courtvision.ingest.walk_shots and
courtvision.mcp.mcp_ending_type, touches no shipped code or artifact.
"""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from courtvision.ingest import walk_shots            # noqa: E402
from courtvision.mcp import mcp_ending_type           # noqa: E402

CORPUS = ROOT / "data" / "corpus"
OUT_DIR = ROOT / "outputs" / "diag"
SEED = 42

# match sampling: include ~10% of matches; hold out ~20% of those as TEST.
INCLUDE_PCT = 10          # h % 100 < INCLUDE_PCT  -> in sample
TEST_MOD = 5              # (h // 7) % TEST_MOD == 0 -> test fold


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------

def _mid_hash(mid):
    return int(hashlib.md5(mid.encode()).hexdigest(), 16)


def _split_of(mid):
    """Return 'skip', 'train', or 'test' for a match id — deterministic."""
    h = _mid_hash(mid)
    if h % 100 >= INCLUDE_PCT:
        return "skip"
    return "test" if (h // 7) % TEST_MOD == 0 else "train"


def _surface_maps():
    """match_id -> surface, from both matches indices."""
    surf = {}
    for g in ("m", "w"):
        p = CORPUS / f"charting-{g}-matches.csv"
        for r in csv.DictReader(open(p)):
            surf[r["match_id"]] = (r.get("Surface") or "?").strip() or "?"
    return surf


def _pts_bucket(pts):
    """Coarse score-pressure bucket from the 'Pts' string (e.g. '30-40')."""
    pts = (pts or "").strip()
    if pts in ("40-AD", "AD-40", "30-40", "40-30", "AD-40", "40-AD"):
        return "pressure"
    if pts in ("0-0",):
        return "start"
    if "AD" in pts:
        return "pressure"
    return "mid"


def load_rows():
    """Single streaming pass -> (ending_rows, shot_rows) with split tags.

    ending_rows: one per point (point-level, TASK 1).
    shot_rows  : one per rally shot i>=1 (shot-level, TASK 2/3).
    """
    surf = _surface_maps()
    ending_rows, shot_rows = [], []
    seen_matches = {"train": set(), "test": set()}
    shards = [CORPUS / f"charting-{g}-points-{era}.csv"
              for g in ("m", "w")
              for era in ("2020s", "2010s", "to-2009")]
    for shard in shards:
        if not shard.exists():
            continue
        for r in csv.DictReader(open(shard)):
            mid = r.get("match_id", "")
            split = _split_of(mid)
            if split == "skip":
                continue
            played = (r.get("2nd") or "").strip() or (r.get("1st") or "").strip()
            if not played:
                continue
            shots = walk_shots(played)
            if not shots:
                continue
            seen_matches[split].add(mid)
            gender = "m" if "-M-" in mid else "w"
            surface = surf.get(mid, "?")
            svr = r.get("Svr", "?")
            ptsb = _pts_bucket(r.get("Pts"))
            served = "2nd" if (r.get("2nd") or "").strip() else "1st"
            serve_zone = shots[0]["serve_zone"]
            rally = shots[1:]                       # rally strokes only
            rlen = len(rally)

            # ---- TASK 1: point ending ----
            etype = mcp_ending_type(played)
            if etype in ("*", "n", "w", "d"):
                last = shots[-1]
                prev = shots[-2] if len(shots) >= 2 else {"side": "none"}
                first_rally_side = rally[0]["side"] if rally else "none"
                # striker of the last shot: server strikes even 0-based idx
                last_idx = len(shots) - 1
                last_role = "server" if last_idx % 2 == 0 else "returner"
                ending_rows.append({
                    "split": split, "match_id": mid, "y": etype,
                    "gender": gender, "surface": surface, "svr": svr,
                    "pts": ptsb, "served": served, "serve_zone": serve_zone,
                    "rlen": str(min(rlen, 12)),
                    "last_side": last["side"], "last_dir": last["dir"] or "?",
                    "prev_side": prev["side"],
                    "first_rally_side": first_rally_side,
                    "last_role": last_role,
                })

            # ---- TASK 2/3: next shot side / direction ----
            for i in range(1, len(shots)):
                cur = shots[i]
                if cur["side"] not in ("fh", "bh"):
                    continue
                p1 = shots[i - 1]
                p2 = shots[i - 2] if i >= 2 else {"side": "none", "dir": ""}
                role = "server" if i % 2 == 0 else "returner"
                shot_rows.append({
                    "split": split, "match_id": mid,
                    "y_side": cur["side"],
                    "y_dir": cur["dir"] if cur["dir"] in ("1", "2", "3") else "?",
                    "gender": gender, "surface": surface, "svr": svr,
                    "pts": ptsb, "serve_zone": serve_zone,
                    "pos": str(min(i, 12)), "role": role,
                    "prev_side": p1["side"], "prev_dir": p1["dir"] or "?",
                    "prev2_side": p2["side"], "prev2_dir": p2.get("dir") or "?",
                })
    return ending_rows, shot_rows, seen_matches


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _fit_eval(rows, feat_cols, y_col):
    """One-hot logistic regression; train on split=='train', test on 'test'.

    Returns dict with accuracy, base-rate (majority class in test), counts,
    macro per-class recall, and the confusion-ish top predictions.
    """
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]

    def X(rs):
        return [[r[c] for c in feat_cols] for r in rs]

    enc = OneHotEncoder(handle_unknown="ignore")
    Xtr = enc.fit_transform(X(tr))
    Xte = enc.transform(X(te))
    ytr = np.array([r[y_col] for r in tr])
    yte = np.array([r[y_col] for r in te])

    clf = LogisticRegression(max_iter=400, C=1.0, random_state=SEED)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)

    # base rate: predict the training-majority class everywhere
    maj = Counter(ytr).most_common(1)[0][0]
    base = float(np.mean(yte == maj))

    # per-class recall on test
    per_class = {}
    for cls in sorted(set(yte)):
        mask = yte == cls
        if mask.sum():
            per_class[cls] = {
                "n_test": int(mask.sum()),
                "recall": round(float(np.mean(pred[mask] == cls)), 3),
            }
    return {
        "n_train": len(tr), "n_test": len(te),
        "classes": sorted(set(ytr) | set(yte)),
        "base_rate_majority_class": round(base, 4),
        "majority_class": maj,
        "accuracy": round(float(acc), 4),
        "lift_abs": round(float(acc) - base, 4),
        "per_class": per_class,
        "test_class_dist": {k: v for k, v in
                            Counter(yte.tolist()).most_common()},
    }


def main():
    print("loading corpus (streaming, match-hash split)…", flush=True)
    ending_rows, shot_rows, seen = load_rows()

    # verify disjoint split
    overlap = seen["train"] & seen["test"]
    assert not overlap, f"LEAK: {len(overlap)} matches in both splits"

    result = {
        "seed": SEED,
        "include_pct": INCLUDE_PCT,
        "n_matches_train": len(seen["train"]),
        "n_matches_test": len(seen["test"]),
        "match_split_disjoint": True,
        "tasks": {},
    }

    print(f"matches: train={len(seen['train'])} test={len(seen['test'])} "
          f"(disjoint)", flush=True)
    print(f"ending rows: {len(ending_rows)}  shot rows: {len(shot_rows)}",
          flush=True)

    end_feats = ["gender", "surface", "svr", "pts", "served", "serve_zone",
                 "rlen", "last_side", "last_dir", "prev_side",
                 "first_rally_side", "last_role"]
    side_feats = ["gender", "surface", "svr", "pts", "serve_zone", "pos",
                  "role", "prev_side", "prev_dir", "prev2_side", "prev2_dir"]
    dir_feats = side_feats

    print("TASK 1: point ending type…", flush=True)
    result["tasks"]["ending_type"] = _fit_eval(ending_rows, end_feats, "y")

    print("TASK 2: next shot side…", flush=True)
    result["tasks"]["next_side"] = _fit_eval(shot_rows, side_feats, "y_side")

    print("TASK 3: next shot direction…", flush=True)
    dir_rows = [r for r in shot_rows if r["y_dir"] in ("1", "2", "3")]
    result["tasks"]["next_direction"] = _fit_eval(dir_rows, dir_feats, "y_dir")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "notation_prior.json").write_text(json.dumps(result, indent=2))
    _write_md(result)
    print("wrote", OUT_DIR / "notation_prior.json", flush=True)
    print(json.dumps({k: {"acc": v["accuracy"],
                          "base": v["base_rate_majority_class"],
                          "lift": v["lift_abs"]}
                      for k, v in result["tasks"].items()}, indent=2))


def _write_md(res):
    L = []
    L.append("# Track B — the video-free notation prior\n")
    L.append(f"Seed {res['seed']} · matches: {res['n_matches_train']} train / "
             f"{res['n_matches_test']} test (match-disjoint hash split, "
             f"~{res['include_pct']}% of the ~1.85M-point corpus sampled).\n")
    L.append("How predictable is each chart token from NOTATION CONTEXT "
             "alone — no pixels? Accuracy is on the held-out matches; "
             "base-rate is the always-guess-majority-class baseline.\n")
    L.append("| task | classes | n_test | base-rate | accuracy | lift |")
    L.append("|---|---|--:|--:|--:|--:|")
    names = {"ending_type": "point ending (win/net/wide/deep)",
             "next_side": "next shot side (fh/bh)",
             "next_direction": "next shot direction (1/2/3)"}
    for k, t in res["tasks"].items():
        L.append(f"| {names.get(k,k)} | {'/'.join(t['classes'])} | "
                 f"{t['n_test']:,} | {t['base_rate_majority_class']:.1%} | "
                 f"**{t['accuracy']:.1%}** | +{t['lift_abs']:.1%} |")
    L.append("")
    for k, t in res["tasks"].items():
        L.append(f"### {names.get(k,k)}")
        L.append(f"majority class `{t['majority_class']}`; per-class recall:")
        for cls, pc in t["per_class"].items():
            L.append(f"- `{cls}` — recall {pc['recall']:.0%} "
                     f"(n={pc['n_test']:,})")
        L.append("")
    (OUT_DIR / "notation_prior.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
