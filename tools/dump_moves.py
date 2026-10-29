"""Play seeded games and hash every turn's full legal-move set.

Used to check that move generation with a GADDAG gives exactly the same moves
as without one:  python tools/dump_moves.py DATA_DIR GAMES > out.txt
"""
import hashlib
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import qenv

data, games = sys.argv[1], int(sys.argv[2])
qenv.init(data, "nwl23", seed=2024)
print("gaddag", qenv.has_gaddag(), file=sys.stderr)
gen_time, turns = 0.0, 0
for k in range(games):
    qenv.seed(1000 + k)
    g = qenv.Game()
    while not g.game_over and g.scoreless_turns < 6:
        t0 = time.perf_counter()
        m = g.legal_moves()
        gen_time += time.perf_counter() - t0
        turns += 1
        s = g.move_strings()
        eq = m["equity"]
        print(k, turns, len(s), hashlib.md5("\n".join(sorted(s)).encode()).hexdigest())
        # deterministic choice independent of list order: best equity, then name
        best = max(range(len(s)), key=lambda i: (round(float(eq[i]), 4), s[i]))
        g.play(best)
print(f"{turns} turns, move-gen {1e3 * gen_time / turns:.2f} ms/turn", file=sys.stderr)
