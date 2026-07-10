"""Point-acceptance metric — the draft north star.

A charted point is compared to its MCP row at TOKEN level:

    [serve+zone] [letter+direction]* [ending]

e.g. MCP "4b28f1f3n#" -> ['s4', 'b2', 'f1', 'f3', 'n']
     ours "s4b2f1f3n@" -> ['s4', 'b2', 'f1', 'f3', 'n']

A point is ACCEPTED if the two token lists are within ONE token edit
(Levenshtein over whole tokens: one substitution, insertion, or
deletion). Token equality is strict string equality — a '?' letter or
zone matches nothing, so refusing to commit costs the same as being
wrong. Forced/unforced attribution (#/@) is charter judgment and is
not tokenized; MCP depth digits (7-9) and modifiers (+ - = ^ ; !) are
charting detail beyond the pipeline's claim and are dropped.

Lifted verbatim from experiments/mcp_accept.py (2026-07-10), plus the
edit backtrace/attribution helpers from experiments/mcp_decompose.py —
the alignment machinery the eval, decomposition, and confidence layers
share.

    from courtvision.mcp import mcp_point_tokens, chart_point_tokens
"""

FH_SIDE = set("frvoul")
BH_SIDE = set("bszpym")
SHOT_CHARS = FH_SIDE | BH_SIDE | set("hijkt")


def mcp_ending_type(played):
    """Ending TYPE only: * winner, or n/w/d/x error kind."""
    if played.endswith("*"):
        return "*"
    t = played.rstrip("@#!")
    return t[-1] if t and t[-1] in "nwdx" else "?"


def mcp_point_tokens(played):
    """Tokenize an MCP point string (the PLAYED serve's string)."""
    toks = []
    serve = played[0] if played and played[0] in "0456" else "?"
    toks.append(f"s{serve if serve in '456' else '?'}")
    i = 1
    while i < len(played):
        c = played[i]
        if c in SHOT_CHARS:
            letter = "f" if c in FH_SIDE else "b" if c in BH_SIDE else "?"
            direction = "?"
            j = i + 1
            while j < len(played) and played[j] not in SHOT_CHARS:
                if played[j] in "123" and direction == "?":
                    direction = played[j]
                j += 1
            toks.append(f"{letter}{direction}")
            i = j
        else:
            i += 1
    toks.append(mcp_ending_type(played))
    return toks


def chart_point_tokens(shot_rows, ending):
    """Tokenize our chart: shot_rows are chart2_*.csv DictReader rows,
    ending is the match_chart 'ending' column."""
    toks = []
    for sh in shot_rows:
        if sh["is_serve"] == "True":
            z = sh["zone"] if sh["zone"] in ("4", "5", "6") else "?"
            toks.append(f"s{z}")
        else:
            letter = sh["letter"] if sh["letter"] in ("f", "b") else "?"
            z = sh["zone"] if sh["zone"] in ("1", "2", "3") else "?"
            toks.append(f"{letter}{z}")
    if not toks or not toks[0].startswith("s"):
        toks.insert(0, "s?")     # chart never found the serve: wrong, visibly
    e = "*" if ending == "*" else (ending[0] if ending and ending[0] in "nwdx" else "?")
    toks.append(e)
    return toks


def token_levenshtein(a, b):
    """Plain Levenshtein over token lists (unit costs)."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def accepted(mcp_toks, our_toks, max_edits=1):
    return token_levenshtein(mcp_toks, our_toks) <= max_edits


# ---------------------------------------------------------------------------
# Shared MCP-string helpers (from the t*w eval scripts) and the edit
# backtrace/attribution machinery (from experiments/mcp_decompose.py).
# ---------------------------------------------------------------------------

ENDINGS = set("*nwdx?")


def parse_mcp(first, second):
    """MCP row -> (serve_digit, [stroke letters], played_string).
    The played string is the second serve's if the first faulted."""
    s = second if second.strip() else first
    serve = s[0] if s and s[0] in "0456" else "?"
    strokes = [c for c in s[1:] if c in SHOT_CHARS]
    return serve, strokes, s


def our_ending_type(ending):
    """Our chart's ending column -> comparable ending TYPE."""
    if ending == "*":
        return "*"
    return ending[0] if ending and ending[0] in "nwdx" else "?"


def tok_kind(t):
    if len(t) == 2 and t[0] == "s":
        return "serve"
    if len(t) == 1 and t in ENDINGS:
        return "ending"
    return "rally"


def backtrace(a, b):
    """Full-matrix token Levenshtein with backtrace.

    Returns (distance, ops) where ops is a list of
    (op, a_tok_or_None, b_tok_or_None); op in match/sub/del/ins.
    'del' = token in a (MCP) our draft lacks; 'ins' = extra token in
    b (ours) MCP lacks. Ties prefer diagonal, then del, then ins —
    deterministic, and diagonal-first keeps substitutions honest."""
    m, n = len(a), len(b)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i][j] = min(D[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                          D[i - 1][j] + 1, D[i][j - 1] + 1)
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub",
                        a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    ops.reverse()
    assert sum(op != "match" for op, _, _ in ops) == D[m][n]
    return D[m][n], ops


def classify_sub(mt, ot):
    """Bin a substitution by which component is wrong."""
    km, ko = tok_kind(mt), tok_kind(ot)
    if km != ko:
        return "sub_crosstype"
    if km == "serve":
        return "sub_serve_zone"
    if km == "ending":
        return "sub_ending"
    letter_ok = mt[0] == ot[0]
    dir_ok = mt[1] == ot[1]
    if letter_ok and not dir_ok:
        return "sub_direction"
    if dir_ok and not letter_ok:
        return "sub_letter"
    return "sub_letter+dir"


def structural(toks):
    """Shot count + letters only: serve -> 's', rally -> letter,
    ending dropped."""
    out = []
    for t in toks:
        k = tok_kind(t)
        if k == "serve":
            out.append("s")
        elif k == "rally":
            out.append(t[0])
    return out


def counterfactual(mcp, ours, fix):
    """Rebuild our token list with one component set to MCP truth at
    Levenshtein-aligned positions (sub or match pairs). Structure —
    insertions and deletions — is left exactly as charted.

    fix in: serve, letters, dirs_all, dirs_attempted, endings, and
    '+'-joined combos."""
    fixes = set(fix.split("+"))
    _, ops = backtrace(mcp, ours)
    out = []
    for op, mt, ot in ops:
        if ot is None:
            continue                      # del: we can't fix a missing shot
        if op in ("match", "sub") and mt is not None:
            km, ko = tok_kind(mt), tok_kind(ot)
            if km == ko == "serve" and "serve" in fixes:
                ot = mt
            elif km == ko == "ending" and "endings" in fixes:
                ot = mt
            elif km == ko == "rally":
                letter, d = ot[0], ot[1]
                if "letters" in fixes:
                    letter = mt[0]
                if "dirs_all" in fixes or ("dirs_attempted" in fixes
                                           and d != "?"):
                    d = mt[1]
                ot = letter + d
        out.append(ot)
    return out
