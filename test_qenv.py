"""Sanity checks and speed test for the qenv bridge.

Run from this folder:  python test_qenv.py /path/to/quackle/data
"""
import sys
import time

import numpy as np

import qenv

DATA = sys.argv[1] if len(sys.argv) > 1 else "../quackle/data"
qenv.init(DATA, "nwl23", seed=123)
rng = np.random.default_rng(0)


def check_move_arrays(g, m):
    """Every move's leave + used tiles must equal the rack."""
    rack = g.rack()
    n = len(m["kind"])
    assert m["kind"][-1] == 2, "last entry should be pass"
    for i in range(n):
        used = np.zeros(27, int)
        if m["kind"][i] == 0:
            k = m["ntiles"][i]
            for r, c, letter, blank in m["tiles"][i][:k]:
                used[0 if blank else letter] += 1
            assert (m["tiles"][i][k:] == -1).all()
        if m["kind"][i] != 1:  # exchanges: tiles array unused
            assert (used + m["leave"][i] == rack).all(), (i, g.move_strings()[i])


def play_game(policy):
    """policy(game, moves) -> index. Returns final scores and per-turn timings."""
    g = qenv.Game()
    gen_t, n_moves = [], []
    while not g.game_over:
        t0 = time.perf_counter()
        m = g.legal_moves()
        gen_t.append(time.perf_counter() - t0)
        n_moves.append(len(m["kind"]))
        i = policy(g, m)
        if m["kind"][i] == 0:  # placed tiles must appear on the board afterwards
            k = m["ntiles"][i]
            placed = m["tiles"][i][:k].copy()
            g.play(i)
            letters, blanks = g.board()
            for r, c, letter, blank in placed:
                assert letters[r, c] == letter and blanks[r, c] == bool(blank)
        else:
            g.play(i)
    return g.scores(), g.turn, gen_t, n_moves


speedy = lambda g, m: int(np.argmax(m["equity"]))           # same as Quackle's Speedy Player
random_play = lambda g, m: int(rng.integers(len(m["kind"]) - 1)) if len(m["kind"]) > 1 else 0

# 1. legality bookkeeping on a few positions
g = qenv.Game()
for _ in range(6):
    m = g.legal_moves()
    check_move_arrays(g, m)
    g.play(int(np.argmax(m["equity"])))
print("move arrays consistent with racks: OK")

# 2. argmax-equity from legal_moves() matches Quackle's own Speedy Player
# (Both games share one global RNG, so compare on a copy of the same position.)
a = qenv.Game()
for _ in range(20):
    if a.game_over:
        break
    m = a.legal_moves()
    i = int(np.argmax(m["equity"]))
    expect = a.move_strings()[i]
    got = a.copy().play_speedy()
    assert expect == got, (expect, got)
    a.play(i)
print("argmax(equity) == Quackle Speedy Player: OK")

# 3. copy() is independent
c = g.copy()
m = c.legal_moves(); c.play(0)
assert c.scores() != g.scores() and c.to_move != g.to_move
print("copy() independent: OK")

# 4. afterstates match the real board (incl. cross-checks) after playing the move
qenv.seed(11); g = qenv.Game()
checked = 0
for _ in range(12):
    if g.game_over:
        break
    m = g.legal_moves()
    idx = list(range(min(len(m["kind"]), 20))) + [len(m["kind"]) - 1]
    a = g.afterstates(idx)
    for k, i in enumerate(idx):
        c = g.copy(); c.play(i)
        (bl, bb), (ch, cv) = c.board(), c.cross_checks()
        assert (bl == a["letters"][k]).all() and (bb == a["blanks"][k]).all()
        assert (ch == a["cross_h"][k]).all() and (cv == a["cross_v"][k]).all()
        checked += 1
    g.play(int(np.argmax(m["equity"])))
print(f"afterstates match real boards ({checked} checked): OK")

# 5. full games + timing
for name, pol, n in [("speedy vs speedy", speedy, 30), ("random vs random", random_play, 10)]:
    t0 = time.perf_counter()
    res = [play_game(pol) for _ in range(n)]
    dt = time.perf_counter() - t0
    scores = np.array([r[0] for r in res])
    gen = np.concatenate([r[2] for r in res])
    nm = np.concatenate([r[3] for r in res])
    print(f"{name}: {n} games in {dt:.1f}s ({n/dt:.1f} games/s), "
          f"mean score {scores.mean():.0f}, mean turns {np.mean([r[1] for r in res]):.1f}, "
          f"move-gen {1e3*gen.mean():.2f} ms/turn (median {1e3*np.median(gen):.2f}), "
          f"moves/turn mean {nm.mean():.0f} max {nm.max()}")
