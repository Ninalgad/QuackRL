import numpy as np


def _encode(p):
    """p: dict of arrays sharing a leading shape B (e.g. () or (K,)).
    Returns board (*B, 15, 15, 90) and vec (*B, 65), float32."""
    L, age, own = p["letters"], p["age"], p["owner"]
    filled = L > 0
    board = np.concatenate([
        np.eye(27, dtype=np.float32)[L],                  # 27: one-hot letter, 0 = empty
        p["blanks"][..., None],                           #  1: blank tile
        (own == 1)[..., None], (own == 2)[..., None],     #  2: mine / opponent's
        (age == 0)[..., None], (age == 1)[..., None],     #  2: opp's last move / my last move
        (np.where(filled, np.clip(age, 0, 30), 0) / 30)[..., None],  # 1: scaled age
        np.eye(5, dtype=np.float32)[p["premium"]][..., 1:],  # 4: DL TL DW TW (uncovered)
        p["placed"][..., None],                           #  1: just placed (afterstates)
        unpack_cross(p["cross_h"]),                       # 26: letters allowed, across word
        unpack_cross(p["cross_v"]),                       # 26: letters allowed, down word
    ], axis=-1).astype(np.float32)

    B = L.shape[:-2]
    def col(x):                                           # scalar or (K,) -> (*B, 1)
        return np.broadcast_to(np.asarray(x, np.float32), B)[..., None]
    vec = np.concatenate([
        p["rack"] / 7,                                    # 27: tiles I hold
        np.broadcast_to(p["unseen"] / 12, B + (27,)),     # 27: bag + opponent's rack
        col(p["bag_size"] / 100),
        col(p["score_me"] / 500), col(p["score_opp"] / 500),
        col((p["score_me"] - p["score_opp"]) / 200),      # spread
        col(p["scoreless_turns"] / 6),
        col(p["move_score"] / 100), col(p["equity"] / 100),  # move features:
        col(p["ntiles"] / 7), col(p["bingo"]),            #   zero for plain states
        col(p["kind"] == 1), col(p["kind"] == 2),         #   exchange / pass
    ], axis=-1).astype(np.float32)
    return board, vec


def encode_state(obs):
    """Stage-1 encoder input: board (15,15,90), vec (65,)."""
    return _encode(dict(
        letters=obs["board_letters"], blanks=obs["board_blanks"],
        owner=obs["board_owner"], age=obs["board_age"], premium=obs["premium"],
        placed=np.zeros((15, 15), bool),
        cross_h=obs["cross_h"], cross_v=obs["cross_v"],
        rack=obs["rack"], unseen=obs["unseen"], bag_size=obs["bag_size"],
        score_me=obs["score_me"], score_opp=obs["score_opp"],
        scoreless_turns=obs["scoreless_turns"],
        move_score=0, equity=0, ntiles=0, bingo=0, kind=-1,
    ))


def encode_afterstates(a):
    """Stage-2 input from env.afterstates(): board (K,15,15,90), vec (K,65)."""
    return _encode(dict(
        letters=a["letters"], blanks=a["blanks"],
        owner=a["board_owner"], age=a["board_age"], premium=a["premium"],
        placed=a["placed"], cross_h=a["cross_h"], cross_v=a["cross_v"],
        rack=a["leave"], unseen=a["unseen"], bag_size=a["bag_size"],
        score_me=a["score_me"], score_opp=a["score_opp"],
        scoreless_turns=a["scoreless_turns"],
        move_score=a["move_score"], equity=a["equity"], ntiles=a["ntiles"],
        bingo=a["bingo"], kind=a["kind"],
    ))


def create_action_input(obs):
    moves = obs['moves']
    moves = [
        moves['row'] / 15, moves['col'] / 15, moves['horizontal'], moves['ntiles'] / 7,
        moves['score'] / 30, moves['bingo'], moves['kind'],
    ]
    moves = np.concatenate([np.vstack(moves, dtype='float32').T,
                            obs['moves']['leave']], dtype=np.float32, axis=-1)
    return moves


def unpack_cross(mask):
    """(...,) uint32 cross-check masks -> (..., 26) bool, index k = letter A+k."""
    return ((mask[..., None] >> np.arange(26, dtype=np.uint32)) & 1).astype(bool)


def create_checkpoint_name_(config, model_idx):
    base_dir = config.get('train', 'dir_name')
    run_name = config.get('train', 'run_name')
    return f"{base_dir}/{run_name}-{model_idx}.pt"


def create_running_name(config):
    base_dir = config.get('train', 'dir_name')
    run_name = config.get('train', 'run_name')
    return f"{base_dir}/{run_name}.pt"
