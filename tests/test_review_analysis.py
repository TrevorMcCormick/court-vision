from courtvision.review_analysis import (active_seconds,
                                         draft_point_tokens)


def ev(ts, event, row="c1"):
    return {"ts_ms": ts, "row": row, "event": event, "payload": {}}


def test_active_seconds_basic_and_pause():
    events = [ev(0, "row_open"), ev(10_000, "play"),
              ev(20_000, "clock_pause"), ev(80_000, "clock_resume"),
              ev(90_000, "accept")]
    active, holes = active_seconds(events)
    # 0->10s + 10->20s + resume 80->90s = 30s; the pause never counts
    assert active == {"c1": 30.0}
    assert holes == []


def test_active_seconds_idle_hole_excluded():
    events = [ev(0, "row_open"), ev(300_000, "accept")]
    active, holes = active_seconds(events)
    assert active.get("c1", 0.0) == 0.0
    assert holes == [("c1", 300.0)]


def test_active_seconds_two_rows_and_revisit():
    events = [ev(0, "row_open", "c1"), ev(5_000, "row_open", "c2"),
              ev(9_000, "accept", "c2"), ev(9_000, "row_open", "c1"),
              ev(15_000, "accept", "c1")]
    active, _ = active_seconds(events)
    assert active == {"c1": 11.0, "c2": 4.0}


def test_draft_tokens_strip_s_and_keep_unknowns():
    assert draft_point_tokens("s4b3f1w@") == ["s4", "b3", "f1", "w"]
    # '?' letters are real shots the tokenizer must not drop
    assert draft_point_tokens("s?f2?3?") == ["s?", "f2", "?3", "?"]
    assert draft_point_tokens("s6*") == ["s6", "*"]


def test_med_true_median_even_length():
    from courtvision.review_analysis import _med
    assert _med([3, 7]) == 5.0
    assert _med([1, 2, 3, 4]) == 2.5
    assert _med([5]) == 5


def test_rubber_stamp_accept_scores_zero_edits():
    # accepting the pre-filled s-prefixed draft unchanged must not
    # score a phantom serve edit vs MCP truth
    from courtvision.mcp import mcp_point_tokens, token_levenshtein
    truth = mcp_point_tokens("4b3f1w@")
    stamped = draft_point_tokens("s4b3f1w@")
    assert token_levenshtein(truth, stamped) == 0


def test_draft_tokens_mirror_mcp_ambiguous_shots():
    # h/i/j/k/t are side-ambiguous in the frozen eval projection;
    # corrected strings must project the same way truth does
    from courtvision.mcp import mcp_point_tokens
    assert draft_point_tokens("4h1n#") == mcp_point_tokens("4h1n#")
