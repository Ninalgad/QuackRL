"""A small two-player Scrabble environment on top of qenv (the Quackle bridge).

Pure Python, so you can read and change every rule decision here.

    env = ScrabbleEnv("/path/to/quackle/data")
    obs = env.reset(seed=0)
    while not env.done:
        i = my_policy(obs)                 # index into obs["moves"]
        obs, reward, done, info = env.step(i)

Rewards are zero-sum and given from the point of view of the player who just
moved: +1 win, -1 loss, 0 draw, and 0 on every non-final move.
"""
from copy import deepcopy
import numpy as np

import qenv

# Standard English tile values, indexed like qenv: 0 = blank, 1..26 = A..Z.
TILE_VALUES = np.array(
    [0, 1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10]
)

# Tournament rule: six scoreless turns in a row end the game.
# Quackle counts these but does not end the game itself, so we do it here.
MAX_SCORELESS_TURNS = 6

_initialized = False
PREMIUMS = None


def unpack_cross(mask):
    """(...,) uint32 cross-check masks -> (..., 26) bool, index k = letter A+k."""
    return ((mask[..., None] >> np.arange(26, dtype=np.uint32)) & 1).astype(bool)


class ScrabbleEnv:
    def __init__(self, data_dir, lexicon="nwl23", restrict_pass=True,
                 max_exchanges=5, exchange_only_if_best=False,
                 allow_exchanges=True, pass_only_if_forced=False):
        """restrict_pass: only offer 'pass' when the bag is empty or no other
        move exists. Prevents the 'pass forever' collapse for almost no cost.

        max_exchanges: offer only this many exchanges, the best by equity
        (None = all of them, often ~70 per turn). Keeps exchanges from flooding
        the candidate list while the best ones stay available.

        exchange_only_if_best: offer exchanges only when Quackle's top move by
        equity is an exchange (as Speedy would play one: ~2% of turns).
        Useful early in training to block exchange-stalling.

        allow_exchanges: False removes exchanges entirely.

        pass_only_if_forced: offer pass only when no other move exists, even
        with the bag empty (overrides restrict_pass)."""
        global _initialized, PREMIUMS
        if not _initialized:
            qenv.init(data_dir, lexicon)
            PREMIUMS = qenv.premiums()   # (15,15): 0 none, 1 DL, 2 TL, 3 DW, 4 TW
            _initialized = True
        self.restrict_pass = restrict_pass
        self.max_exchanges = max_exchanges
        self.exchange_only_if_best = exchange_only_if_best
        self.allow_exchanges = allow_exchanges
        self.pass_only_if_forced = pass_only_if_forced
        self._keep = None      # obs["moves"] row -> row in Quackle's full move list
        self.game = None
        self.done = True
        self.final_scores = None
        # Who placed each tile, and when. Quackle's board doesn't record this,
        # so the env tracks it from the moves it plays.
        self.owner = None      # (15,15) int8: -1 empty, else player id 0/1
        self.placed_ply = None # (15,15) int16: -1 empty, else ply it was placed
        self.ply = 0           # moves made so far, counting both players
        self._moves = None     # move arrays from the latest observation

    # ------------------------------------------------------------------ API
    def to_play(self):
        return deepcopy(self.game.to_move)

    def render(self, top=5):
        g = self.game
        print(g.board_string())
        s = g.scores()
        print(f"P0 {s[0]}  P1 {s[1]}  to move: P{g.to_move}  "
              f"rack: {g.rack_string()}  bag: {g.bag_size}")
        if not self.done:
            print("top moves:", self.move_strings()[:top])

    def move_strings(self):
        """Human-readable moves aligned with the current obs["moves"]."""
        names = self.game.move_strings()
        return [names[j] for j in self._keep]

    def reset(self, seed=None):
        if seed is not None:
            qenv.seed(seed)
        self.game = qenv.Game()
        self.done = False
        self.final_scores = None
        self.owner = np.full((15, 15), -1, dtype=np.int8)
        self.placed_ply = np.full((15, 15), -1, dtype=np.int16)
        self.ply = 0
        return self._observe()

    def step(self, i):
        assert not self.done, "call reset()"
        i = int(i)
        mover = self.game.to_move
        self._record_tiles(i, mover)
        self.game.play(int(self._keep[i]))
        self.ply += 1
        reward = 0.0
        if self.game.game_over or self.game.scoreless_turns >= MAX_SCORELESS_TURNS:
            self.done = True
            self.final_scores = self._final_scores()
            mine, theirs = self.final_scores[mover], self.final_scores[1 - mover]
            reward = float(np.sign(mine - theirs))
        obs = None if self.done else self._observe()
        return obs, reward, self.done, {"scores": self.final_scores or self.game.scores()}

    def speedy_action(self, obs):
        """Index of the move Quackle's Speedy Player would make (max static equity)."""
        return int(np.argmax(obs["moves"]["equity"]))

    def afterstates(self, obs, indices):
        """Stage-2 inputs: the position after each candidate move, BEFORE new
        tiles are drawn, from the mover's point of view.

        indices: positions in obs["moves"] (e.g. stage 1's top K). Must be
        called on the current obs, before step(). Returns a dict of arrays with
        a leading candidate axis K (see README)."""
        idx = np.asarray(indices, dtype=np.int64)
        mv = obs["moves"]
        a = self.game.afterstates(self._keep[idx].tolist())
        score = mv["score"][idx].astype(np.int32)
        a["premium"] = np.where(a["letters"] == 0, PREMIUMS, 0).astype(np.int8)  # uncovered only
        # The mover just placed the new tiles, so they're "mine" in its view.
        a["board_owner"] = np.where(a["placed"], 1, obs["board_owner"]).astype(np.int8)
        a["board_age"] = np.where(a["placed"], 0,
                                  np.where(obs["board_age"] >= 0, obs["board_age"] + 1, -1)).astype(np.int16)
        a["leave"] = mv["leave"][idx]
        a["kind"] = mv["kind"][idx]
        a["move_score"] = score
        a["equity"] = mv["equity"][idx]
        a["ntiles"] = mv["ntiles"][idx]
        a["bingo"] = mv["bingo"][idx]
        a["score_me"] = obs["score_me"] + score
        a["score_opp"] = np.full(len(idx), obs["score_opp"], dtype=np.int32)
        a["scoreless_turns"] = np.where(score == 0, obs["scoreless_turns"] + 1, 0)
        # Unchanged by our move until we draw, so shared by all candidates:
        a["unseen"] = obs["unseen"]
        a["bag_size"] = obs["bag_size"]
        return a

    # ------------------------------------------------------------ internals

    def _filter(self, moves, bag_size):
        """Rows of Quackle's move list to offer, in Quackle's (equity) order."""
        kind, eq = moves["kind"], moves["equity"]
        place = np.where(kind == 0)[0]
        exch = np.where(kind == 1)[0]
        passes = np.where(kind == 2)[0][-1:]   # one pass (Quackle may list its own too)
        if not self.allow_exchanges:
            exch = exch[:0]
        if self.exchange_only_if_best and len(exch) and kind[np.argmax(eq)] != 1:
            exch = exch[:0]
        if self.max_exchanges is not None and len(exch) > self.max_exchanges:
            exch = exch[np.argsort(-eq[exch], kind="stable")[: self.max_exchanges]]
        others = len(place) + len(exch)
        if self.pass_only_if_forced and others > 0:
            passes = passes[:0]                 # pass only when nothing else is legal
        elif self.restrict_pass and bag_size > 0 and others > 0:
            passes = passes[:0]                 # pass only when the bag is empty or forced
        keep = np.sort(np.concatenate([place, exch, passes]))
        if len(keep) == 0:          # safety net: never offer an empty list
            keep = np.arange(len(kind))
        return keep

    def _record_tiles(self, i, mover):
        """Mark the squares move i fills as placed by `mover` at this ply."""
        m = self._moves
        if m["kind"][i] != 0:          # exchange or pass: nothing placed
            return
        t = m["tiles"][i][: m["ntiles"][i]]
        rows, cols = t[:, 0].astype(int), t[:, 1].astype(int)
        self.owner[rows, cols] = mover
        self.placed_ply[rows, cols] = self.ply

    def _observe(self):
        g = self.game
        moves = g.legal_moves()
        self._keep = self._filter(moves, g.bag_size)
        moves = {k: v[self._keep] for k, v in moves.items()}
        self._moves = moves
        letters, blanks = g.board()
        cross_h, cross_v = g.cross_checks()
        me = g.to_move
        scores = g.scores()
        # Ownership relative to the player to move: 0 empty, 1 mine, 2 opponent's.
        owner_rel = np.where(self.owner < 0, 0, np.where(self.owner == me, 1, 2)).astype(np.int8)
        # Age in plies: 0 = placed by the previous move (the opponent's last play), -1 = empty.
        age = np.where(self.placed_ply < 0, -1, self.ply - 1 - self.placed_ply).astype(np.int16)
        return {
            "to_move": me,
            "board_letters": letters,          # (15,15) uint8, 0 = empty, 1..26 = A..Z
            "board_blanks": blanks,            # (15,15) bool
            "board_owner": owner_rel,          # (15,15) int8, 0 empty, 1 mine, 2 opponent's
            "board_age": age,                  # (15,15) int16, plies since placed, -1 empty
            "premium": np.where(letters == 0, PREMIUMS, 0).astype(np.int8),  # uncovered premiums
            "cross_h": cross_h,                # (15,15) uint32, bit k = letter A+k allowed
            "cross_v": cross_v,                #   (h: by the across word, v: by the down word)
            "rack": g.rack(),                  # (27,) counts, index 0 = blank
            "unseen": g.unseen(),              # (27,) counts: bag + opponent rack
            "bag_size": g.bag_size,
            "score_me": scores[me],
            "score_opp": scores[1 - me],
            "scoreless_turns": g.scoreless_turns,
            "moves": moves,                    # dict of arrays, one row per candidate
        }

    def _final_scores(self):
        s = list(self.game.scores())
        if not self.game.game_over:
            # Passed-out game: each player loses the value of their own rack.
            for p in (0, 1):
                s[p] -= int(TILE_VALUES @ self.game.rack(p))
        return tuple(s)


class PlacementOnlyEnv(ScrabbleEnv):
    """ScrabbleEnv with no exchanges, and pass only when no placement exists.

    Every offered move places tiles, except a forced pass when a player has no
    legal placement. If both players are stuck, six scoreless turns end the game.
    """

    def __init__(self, data_dir, lexicon="nwl23"):
        super().__init__(data_dir, lexicon, allow_exchanges=False, pass_only_if_forced=True)
