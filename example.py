"""Play agents against Quackle's Speedy Player through env.py.

    python example.py /path/to/quackle/data

Replace `random_agent` with your own network: it receives the observation dict
and must return an index into obs["moves"].
"""
import sys

import numpy as np

from env import ScrabbleEnv

DATA = sys.argv[1] if len(sys.argv) > 1 else "../quackle/data"
env = ScrabbleEnv(DATA)
rng = np.random.default_rng(0)


def random_agent(obs):
    return int(rng.integers(len(obs["moves"]["kind"])))


def speedy_agent(obs):
    return env.speedy_action(obs)


def match(agent_a, agent_b, games=20):
    """Agent A's win rate vs agent B, swapping who moves first each game."""
    wins = 0.0
    for k in range(games):
        obs = env.reset(seed=k)
        a_seat = k % 2                      # alternate the first move
        while True:
            agent = agent_a if obs["to_move"] == a_seat else agent_b
            mover = obs["to_move"]
            obs, reward, done, info = env.step(agent(obs))
            if done:
                a_reward = reward if mover == a_seat else -reward
                wins += (a_reward + 1) / 2      # win = 1, draw = 0.5, loss = 0
                break
    return wins / games


print("random vs speedy :", match(random_agent, speedy_agent))
print("speedy vs speedy :", match(speedy_agent, speedy_agent))

# The six-scoreless-turns rule ends a game of mutual passing.
env_np = ScrabbleEnv(DATA, restrict_pass=False)
obs = env_np.reset(seed=1)
turns = 0
while True:
    obs, r, done, info = env_np.step(len(obs["moves"]["kind"]) - 1)  # always pass
    turns += 1
    if done:
        print(f"all-pass game ended after {turns} turns, final scores {info['scores']}")
        break
