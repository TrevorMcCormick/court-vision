"""Lint is advisory: it returns issues, it never raises, and an empty
list means 'nothing to warn about'. Vocabulary follows the MCP
charting instructions; anything outside it warns with a position so
the UI can point at the character."""

from courtvision.notation import lint


def msgs(issues):
    return [i["msg"] for i in issues]


def test_legal_rally_string_is_clean():
    # serve T, bh crosscourt deep, fh down the line, fh winner
    assert lint("6b29f1f3*", "") == []


def test_legal_fault_then_second_serve_point():
    # 1st = wide fault; 2nd = body serve, bh into net, unforced
    assert lint("4w", "5b2n@") == []


def test_unknown_char_warns_with_position():
    issues = lint("6b2Z*", "")
    assert len(issues) == 1
    assert issues[0]["field"] == "1st"
    assert issues[0]["pos"] == 3
    assert "unknown" in issues[0]["msg"]


def test_orphan_digit_warns():
    # direction digit with no shot letter before it
    issues = lint("62b1*", "")
    assert any("digit" in m for m in msgs(issues))


def test_missing_serve_digit_warns():
    issues = lint("b2f1*", "")
    assert any("serve digit" in m for m in msgs(issues))


def test_no_ending_mark_warns():
    issues = lint("6b2f1", "")
    assert any("ending" in m for m in msgs(issues))


def test_second_filled_but_first_not_a_fault_warns():
    # if the point was played on the 2nd, the 1st should be a fault
    issues = lint("6b2f1*", "5b2n@")
    assert any("fault" in m for m in msgs(issues))


def test_empty_first_warns():
    issues = lint("", "")
    assert any("empty" in m for m in msgs(issues))


def test_error_ending_needs_error_letter():
    # '@' with no n/w/d/x/!/e before it
    issues = lint("6b2f1@", "")
    assert any("error letter" in m for m in msgs(issues))
