from typing import (Any, Dict, Callable, List, Tuple, Union, NamedTuple,
                    Optional, Sequence, Set)
import numpy as np
from configparser import SectionProxy
import torch
from src.utils import *


Config = SectionProxy
Pos = Tuple[int, int]
Mat = np.array
Player = int


class NetworkInput(NamedTuple):
    """Data for a net forward."""
    board: torch.tensor
    vec: torch.tensor
    action: torch.tensor


class Observation:
    def __init__(self, obs, is_afterstate=False):
        if is_afterstate:
            self.board, self.vec = encode_afterstates(obs)
            self.action = None
        else:
            self.board, self.vec = encode_state(obs)
            self.action = create_action_input(obs)
        self.is_afterstate = is_afterstate

    def to_frame(self, frame_idx):
        if self.is_afterstate:
            self.board, self.vec = self.board[frame_idx], self.vec[frame_idx]

    def get_inputs(self, device):
        if self.is_afterstate:
            action = None
        else:
            action = torch.tensor(self.action).float().to(device)

        return NetworkInput(
            board=torch.tensor(self.board).float().to(device),
            vec=torch.tensor(self.vec).float().to(device),
            action=action
        )


class State(NamedTuple):
    """Data for a single state."""
    state_observation: Observation
    afterstate_observation: Observation
    reward: float
    player: Player
    action: np.array


Trajectory = List[State]
