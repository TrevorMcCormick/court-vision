"""MCP-notation legality lint — advisory, never blocking.

The review tool lints what the charter types; the same vocabulary
tables later validate the cv-19/20 strict export. Sources: the Match
Charting Project charting instructions (MatchChart 0.3.2). The lint
is deliberately humble: everything it doesn't recognize is a WARNING
with a position, never a rejection — MCP has edge vocabulary (lets,
time violations, challenges) that a charter may legitimately use.

    from courtvision.notation import lint
    issues = lint(first, second)   # [{"field","pos","msg"}, ...]

An empty list means nothing to warn about. `pos` is a 0-based index
into the offending field's string, or None for whole-string issues.
"""

import json
from pathlib import Path

GRAMMAR_PATH = Path(__file__).with_name("grammar.json")
GRAMMAR = json.loads(GRAMMAR_PATH.read_text())
_V = GRAMMAR["vocab"]

SERVE_DIGITS = set(_V["serve_digits"])   # 4 wide, 5 body, 6 T; 0 unk
SHOT_LETTERS = set(_V["shot_letters"])
DIRECTIONS = set(_V["directions"])
DEPTHS = set(_V["depths"])
ERROR_LETTERS = set(_V["error_letters"])
ENDING_MARKS = set(_V["ending_marks"])   # winner, unforced, forced
FAULT_LETTERS = set(_V["fault_letters"])
OTHER_MARKS = set(_V["other_marks"])


def _lint_field(s, field, is_fault_string):
    issues = []
    if not s:
        issues.append({"field": field, "pos": None,
                       "msg": f"{field} is empty"})
        return issues
    if s[0] not in SERVE_DIGITS:
        issues.append({"field": field, "pos": 0,
                       "msg": "no serve digit (0/4/5/6) at start"})
    known = (SERVE_DIGITS | SHOT_LETTERS | DIRECTIONS | DEPTHS |
             ERROR_LETTERS | ENDING_MARKS | FAULT_LETTERS |
             OTHER_MARKS)
    last_shot_idx = None
    for i, c in enumerate(s):
        if c not in known:
            issues.append({"field": field, "pos": i,
                           "msg": f"unknown mark '{c}'"})
        elif c in SHOT_LETTERS:
            last_shot_idx = i
        elif c in (DIRECTIONS | DEPTHS) and i > 0:
            # a direction/depth digit must trail a shot
            if last_shot_idx is None:
                issues.append({"field": field, "pos": i,
                               "msg": f"digit '{c}' follows no shot"})
    if is_fault_string:
        if not any(c in FAULT_LETTERS for c in s[1:]):
            issues.append({"field": field, "pos": None,
                           "msg": "2nd is filled, so 1st should be a "
                                  "fault (needs a fault letter)"})
        return issues
    if s[-1] not in ENDING_MARKS:
        issues.append({"field": field, "pos": len(s) - 1,
                       "msg": "no ending mark (* winner, @ unforced, "
                              "# forced)"})
    elif s[-1] in "@#":
        body = s[:-1]
        if not (body and body[-1] in ERROR_LETTERS):
            issues.append({"field": field, "pos": len(s) - 1,
                           "msg": "error ending without an error letter "
                                  "(n/w/d/x/!/e) before it"})
    return issues


def lint(first, second):
    """Lint an MCP 1st/2nd pair. Returns a list of warning dicts."""
    first, second = (first or "").strip(), (second or "").strip()
    issues = _lint_field(first, "1st", is_fault_string=bool(second))
    if second:
        issues += _lint_field(second, "2nd", is_fault_string=False)
    return issues
