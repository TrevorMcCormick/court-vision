"""The reference score engine — winner derivation + match state.

winner_from_strings() reads an MCP 1st/2nd pair and names the point
winner (1 = server, 2 = returner, None = underivable, e.g. the '?'
unknown ending a cut clip forces). Score replays a match point by
point and emits, for each point, the row-context the MCP points file
carries (Set1..Svr, Pts server-first, Gm#, TbSet) — always computed
by replay, never stored, so editing an earlier point can never leave
a stale score on a later one.

Every convention here is pinned by tests/test_score.py's oracle
replays of three real MCP points files (t3 best-of-5, t6 WTA
best-of-3, t7 where sets 2 and 3 both reach 6-6, tiebreaks
deciding 7-5 and 7-2).
"""

from .notation import (SHOT_LETTERS, ENDING_MARKS, FAULT_LETTERS,
                       OTHER_MARKS)

PTS = ["0", "15", "30", "40"]


def _core(s):
    return s.rstrip("".join(OTHER_MARKS))


def winner_from_strings(first, second):
    first = (first or "").strip()
    second = (second or "").strip()
    if second:
        c = _core(second)
        if (c and c[-1] in FAULT_LETTERS
                and not any(ch in ENDING_MARKS for ch in second)):
            return 2                          # double fault
    played = second if second else first
    if not played:
        return None
    c = _core(played)
    if not c:
        return None
    ending = c[-1]
    shots = 1 + sum(1 for ch in played[1:] if ch in SHOT_LETTERS)
    last_hitter = 1 if shots % 2 == 1 else 2
    if ending == "*":
        return last_hitter
    if ending in "@#":
        return 2 if last_hitter == 1 else 1
    return None


class Score:
    def __init__(self, best_of=3, final_set="tb7", first_server=1):
        assert final_set in ("tb7", "adv", "tb10")
        self.best_of, self.final_set = best_of, final_set
        self.sets = [[0, 0]]                 # games per set
        self.pts = [0, 0]                    # game points or TB points
        self.server = first_server
        self.game_no = 1                     # data proved: t6's set 2 opens
                                              # at Gm#13 not Gm#1 -- Gm# runs
                                              # continuously, never resets
        self.tb = False
        self._tb_opener = None
        self._tb_target = 7
        self.set_winners = []

    # -- helpers --------------------------------------------------------

    def _deciding(self):
        return len(self.sets) == self.best_of

    def _tb_this_set(self):
        if self.final_set == "adv" and self._deciding():
            return False
        return True

    def _pts_display(self):
        a, b = self.pts if self.server == 1 else self.pts[::-1]
        if self.tb:
            return f"{a}-{b}"
        if a >= 3 and b >= 3:
            if a == b:
                return "40-40"
            return "AD-40" if a > b else "40-AD"
        return f"{PTS[min(a, 3)]}-{PTS[min(b, 3)]}"

    def row_context(self):
        g = self.sets[-1]
        won1 = sum(1 for w in self.set_winners if w == 1)
        won2 = sum(1 for w in self.set_winners if w == 2)
        return {"Set1": str(won1), "Set2": str(won2),
                "Gm1": str(g[0]), "Gm2": str(g[1]),
                "Pts": self._pts_display(), "Gm#": str(self.game_no),
                "TbSet": str(self._tb_this_set()),
                "Svr": str(self.server)}

    @property
    def p1_sets(self):
        return sum(1 for w in self.set_winners if w == 1)

    @property
    def p2_sets(self):
        return sum(1 for w in self.set_winners if w == 2)

    @property
    def over(self):
        need = self.best_of // 2 + 1
        return (self.set_winners.count(1) >= need
                or self.set_winners.count(2) >= need)

    @property
    def display(self):
        s = " ".join(f"{a}-{b}" for a, b in self.sets)
        return f"{s} {self._pts_display()} (svr P{self.server})"

    # -- advancing ------------------------------------------------------

    def point(self, winner_player):
        """Apply one point (winner in PLAYER numbering, 1|2).
        Returns the row-context AS OF before this point."""
        ctx = self.row_context()
        i = winner_player - 1
        self.pts[i] += 1
        a, b = self.pts
        if self.tb:
            total = a + b
            if max(a, b) >= self._tb_target and abs(a - b) >= 2:
                self._game_won(winner_player, tiebreak=True)
            elif total % 2 == 1:
                self._swap_server()
        else:
            if max(a, b) >= 4 and abs(a - b) >= 2:
                self._game_won(winner_player)
        return ctx

    def _swap_server(self):
        self.server = 2 if self.server == 1 else 1

    def _game_won(self, winner_player, tiebreak=False):
        g = self.sets[-1]
        g[winner_player - 1] += 1
        self.pts = [0, 0]
        self.game_no += 1
        a, b = g
        set_done = False
        if tiebreak:
            set_done = True
            self.server = self._tb_opener      # then swap below
            self.tb, self._tb_opener = False, None
        elif max(a, b) >= 6 and abs(a - b) >= 2:
            set_done = True
        elif a == 6 and b == 6 and self._tb_this_set():
            self.tb = True
            self._tb_opener = (2 if self.server == 1 else 1)
            self.server = self._tb_opener
            self._tb_target = (10 if (self.final_set == "tb10"
                                      and self._deciding()) else 7)
            return
        self._swap_server()
        if set_done:
            self.set_winners.append(winner_player)
            if not self.over:
                self.sets.append([0, 0])
