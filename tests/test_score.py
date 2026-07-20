"""score.py is the reference scorer. Unit tests pin the derivation
rules; the oracle tests replay three real MCP points files and demand
column-for-column agreement; a final test GENERATES the conformance
fixture the page's JS scorer must match (Task 6 self-test)."""

import csv
import json
from pathlib import Path

from courtvision.score import winner_from_strings, Score

MCP = Path(__file__).resolve().parent.parent / "data" / "mcp"
FIX = Path(__file__).resolve().parent / "fixtures"


def test_winner_ace_and_service_winner():
    assert winner_from_strings("6*", "") == 1          # ace
    assert winner_from_strings("4b2n#", "") == 1       # forced rtn err


# Parity worked by hand per case: shots = serve + rally letters;
# odd -> server hit last; '*' -> last hitter won; '@'/'#' -> lost.
def test_winner_rally_parities():
    assert winner_from_strings("5b2f1*", "") == 1   # 3 shots, *, server
    assert winner_from_strings("4f1*", "") == 2     # 2 shots, *, rtnr
    assert winner_from_strings("4b3f1w@", "") == 2  # 3 shots, err, srv errs -> rtnr
    assert winner_from_strings("6b2f3b2n@", "") == 1  # 4 shots, rtnr errs


def test_winner_second_serve_and_double_fault():
    assert winner_from_strings("4d", "5b2f1*") == 1   # played on 2nd
    assert winner_from_strings("6w", "4d") == 2       # double fault
    assert winner_from_strings("6w", "4g") == 2       # DF via footfault


def test_winner_unknown_ending_is_none():
    assert winner_from_strings("4b2f1?", "") is None
    assert winner_from_strings("0bbff?", "") is None


def test_other_marks_do_not_break_parity():
    # approach modifier and let cord are not shots or endings
    assert winner_from_strings("4b2+f1*", "") == 1


ORACLES = [
    # file, best_of, final_set  (first server read from row 1's Svr)
    ("points_20230611_djokovic_ruud.csv", 5, "tb7"),
    ("points_20240907_sabalenka_pegula.csv", 3, "tb7"),
    ("points_20231114_djokovic_sinner.csv", 3, "tb7"),
]

CHECK_COLS = ["Set1", "Set2", "Gm1", "Gm2", "Pts", "Svr", "TbSet"]


def _rows(fname):
    # t7's rows are filed starting mid-match (Pt 72..218 then wraps to
    # Pt 1..71, all 218 points present exactly once) -- sort by Pt to
    # replay chronologically; no-op for t3/t6, which are already sorted.
    with open(MCP / fname) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["Pt"]))
    return rows


def test_oracle_replays_match_real_points_files():
    for fname, best_of, fs in ORACLES:
        rows = _rows(fname)
        sc = Score(best_of=best_of, final_set=fs,
                   first_server=int(rows[0]["Svr"]))
        for r in rows:
            ctx = sc.point(int(r["PtWinner"]))
            for col in CHECK_COLS:
                assert ctx[col] == r[col], (
                    f"{fname} Pt {r['Pt']}: {col} "
                    f"engine={ctx[col]!r} file={r[col]!r}")
        assert sc.over, fname


def test_oracle_gm_number():
    # Gm# semantics discovered from data: assert engine matches file.
    fname, best_of, fs = ORACLES[1]
    rows = _rows(fname)
    sc = Score(best_of=best_of, final_set=fs,
               first_server=int(rows[0]["Svr"]))
    for r in rows:
        ctx = sc.point(int(r["PtWinner"]))
        assert ctx["Gm#"] == r["Gm#"], (
            f"Pt {r['Pt']}: Gm# engine={ctx['Gm#']} file={r['Gm#']}")


def test_winner_from_strings_against_ptwinner_column():
    """Where derivable, string-derived winners must match PtWinner.
    Report coverage so silent mass-None can't pass."""
    derived = checked = 0
    for fname, _, _ in ORACLES:
        for r in _rows(fname):
            w = winner_from_strings(r["1st"], r["2nd"])
            checked += 1
            if w is None:
                continue
            derived += 1
            svr = int(r["Svr"])
            player = svr if w == 1 else (2 if svr == 1 else 1)
            assert player == int(r["PtWinner"]), (
                f"{fname} Pt {r['Pt']}: {r['1st']!r}/{r['2nd']!r} "
                f"derived P{player}, file P{r['PtWinner']}")
    assert derived / checked > 0.95, (derived, checked)


def test_generate_conformance_fixture():
    FIX.mkdir(exist_ok=True)
    out = []
    for fname, best_of, fs in ORACLES:
        rows = _rows(fname)
        sc = Score(best_of=best_of, final_set=fs,
                   first_server=int(rows[0]["Svr"]))
        pts = []
        for r in rows:
            winner_player = int(r["PtWinner"])
            svr = int(r["Svr"])
            ctx = sc.point(winner_player)
            # winner_rel: 1=server won/2=returner won, same convention
            # as winner_from_strings, so the JS self-test (Task 6) can
            # check both the string derivation and the score replay
            # off one ground-truth field.
            winner_rel = 1 if winner_player == svr else 2
            pts.append({"winner_rel": winner_rel,
                        "first": r["1st"], "second": r["2nd"],
                        "expect": ctx})
        out.append({"match": fname, "best_of": best_of,
                    "final_set": fs,
                    "first_server": int(rows[0]["Svr"]),
                    "points": pts})
    (FIX / "score_conformance.json").write_text(
        json.dumps(out, indent=1))
    assert (FIX / "score_conformance.json").stat().st_size > 10000


def _force_games(sc, n_each):
    """Drive a set to n_each games all by trading service games
    (server wins 4 straight points each game)."""
    for _ in range(n_each * 2):
        srv = sc.server
        for _ in range(4):
            sc.point(srv)


def test_adv_final_set_has_no_tiebreak():
    sc = Score(best_of=3, final_set="adv", first_server=1)
    # split two tb7 sets 6-0 each way quickly
    for _ in range(6 * 4):
        sc.point(1)                    # P1 takes set 1: 6-0
    for _ in range(6 * 4):
        sc.point(2)                    # P2 takes set 2: 0-6
    assert sc.p1_sets == 1 and sc.p2_sets == 1
    _force_games(sc, 6)                # deciding set: 6-6 on serve
    ctx = sc.point(sc.server)
    assert ctx["TbSet"] == "False"     # advantage set: no TB
    assert sc.tb is False              # still normal games at 6-6
    # play on: two more traded games -> 7-7, still no tiebreak
    # (finish the game the point above started)
    srv = sc.server
    for _ in range(3):
        sc.point(srv)
    _force_games(sc, 0)                # no-op guard
    g = sc.sets[-1]
    assert max(g) >= 7 and sc.tb is False


def test_tb10_deciding_set_uses_ten_point_breaker():
    sc = Score(best_of=3, final_set="tb10", first_server=1)
    for _ in range(6 * 4):
        sc.point(1)
    for _ in range(6 * 4):
        sc.point(2)
    _force_games(sc, 6)                # deciding set 6-6
    assert sc.tb is True
    assert sc._tb_target == 10
    # winner needs 10 with 2 clear: 9-0 doesn't end it...
    for _ in range(9):
        sc.point(1)
    assert not sc.over
    sc.point(1)                        # 10-0 ends match
    assert sc.over and sc.p1_sets == 2


def test_tb7_nondeciding_sets_unaffected_by_tb10_preset():
    sc = Score(best_of=3, final_set="tb10", first_server=1)
    _force_games(sc, 6)                # set 1 at 6-6
    assert sc.tb is True and sc._tb_target == 7
