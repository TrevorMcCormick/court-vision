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

Shared by t1w/t2w/t3w/t4w eval scripts:

    from mcp_accept import mcp_point_tokens, chart_point_tokens, accepted
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
