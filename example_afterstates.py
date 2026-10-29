"""Stage-2 afterstates: the position after a candidate move, before drawing.

    python example_afterstates.py /path/to/quackle/data
"""
import sys

import numpy as np

from env import ScrabbleEnv, unpack_cross

DATA = sys.argv[1] if len(sys.argv) > 1 else "../quackle/data"
env = ScrabbleEnv(DATA)
obs = env.reset(seed=0)
obs, *_ = env.step(env.speedy_action(obs))   # one move in, so the board isn't empty

# Stage 1 would pick these; here, the top 8 by Quackle's equity.
top = np.argsort(-obs["moves"]["equity"])[:8]
a = env.afterstates(obs, top)

names = env.game.move_strings()
print("K =", len(top))
for k in range(len(top)):
    print(f"  {names[top[k]]:<16} score {a['move_score'][k]:>3}  "
          f"tiles placed {a['placed'][k].sum()}  my score after {a['score_me'][k]}")

print("\nshapes:")
for key, v in a.items():
    print(f"  {key:<16} {getattr(v, 'shape', ()) } {getattr(v, 'dtype', type(v).__name__)}")

# Cross-check masks unpack to (K, 15, 15, 26) booleans for the network.
print("\ncross_v unpacked:", unpack_cross(a["cross_v"]).shape)
