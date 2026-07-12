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


def test_load_session_rejects_regenerated_export(cfg):
    import csv
    import json
    from courtvision.review_analysis import _load_session
    from tests.conftest import EXPORT_FIELDS, ROWS

    # Create a review session directory with manifest and empty events
    session_dir = cfg.out_dir / "review" / "sha-check"
    session_dir.mkdir(parents=True)
    with open(session_dir / "manifest.json", "w") as f:
        # Store the original SHA of the export
        export_path = cfg.out_dir / "export" / "tt_mcp_draft.csv"
        import hashlib
        original_sha = hashlib.sha256(export_path.read_bytes()).hexdigest()
        json.dump({"export_sha256": original_sha, "rows": []}, f)
    # Create empty events file
    with open(session_dir / "events.jsonl", "w") as f:
        pass
    # Regenerate the export with a different value
    export = cfg.out_dir / "export" / "tt_mcp_draft.csv"
    rows = [dict(r) for r in ROWS]
    rows[0]["1st"] = "s6*"                     # regenerate with a change
    with open(export, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    try:
        _load_session(cfg, "sha-check")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "export changed" in str(e)


def test_load_session_skips_corrupt_event_lines(cfg):
    from courtvision.review import ReviewSession
    from courtvision.review_analysis import _load_session

    s = ReviewSession(cfg, "cold", "bad-lines", seed="s", n=2)
    s.append_event({"ts_ms": 1, "row": "tt_point_01",
                    "event": "row_open", "payload": {}})
    with open(s.dir / "events.jsonl", "a") as f:
        f.write("{truncated garba\n\n")
    s.append_event({"ts_ms": 2, "row": "tt_point_01",
                    "event": "accept", "payload": {}})
    loaded = _load_session(cfg, "bad-lines")
    assert [e["event"] for e in loaded["events"]] == ["row_open", "accept"]


def test_analyze_unions_extra_contaminated_sessions(cfg):
    from courtvision.review import ReviewSession
    from courtvision.review_analysis import analyze

    def stamp(s, ts0):
        for k, clip in enumerate(s.manifest["rows"]):
            s.append_event({"ts_ms": ts0 + k * 10_000, "row": clip,
                            "event": "row_open", "payload": {}})
            s.append_event({"ts_ms": ts0 + k * 10_000 + 5_000,
                            "row": clip, "event": "accept",
                            "payload": {}})
            s.accept(clip, "6b2f1*", "", "")

    a = ReviewSession(cfg, "cold", "an-a", seed="s", n=2)
    stamp(a, 0)
    r = ReviewSession(cfg, "review", "an-r")
    stamp(r, 100_000)
    # practice session: whatever rows it drew, its rows must ALSO be
    # excluded from review timing
    p = ReviewSession(cfg, "cold", "an-p", seed="other", n=2)
    stamp(p, 200_000)

    base = analyze({"cold_a": (cfg, "an-a"), "review": (cfg, "an-r"),
                    "cold_b": (cfg, "an-a")},
                   out_path=cfg.out_dir / "an1.md")
    extra = analyze({"cold_a": (cfg, "an-a"), "review": (cfg, "an-r"),
                     "cold_b": (cfg, "an-a"),
                     "contaminated": [(cfg, "an-p")]},
                    out_path=cfg.out_dir / "an2.md")
    union = set(a.manifest["rows"]) | set(p.manifest["rows"])
    assert f"review timing: {len(set(a.manifest['rows']))}" in base
    assert f"review timing: {len(union)}" in extra
