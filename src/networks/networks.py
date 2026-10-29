import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def __str__(self):
        return "Network"

    def __repr__(self):
        return str(self)

    def forward_representation(self, board, vec) -> torch.tensor:
        pass

    def forward_prediction(self, x: torch.tensor) -> torch.tensor:
        pass

    def forward_actions(self, actions: torch.tensor, r: torch.tensor) -> torch.tensor:
        pass

    def prune_actions(self, actions: torch.tensor, r: torch.tensor):
        pass

    def forward(self, board_state, vec_state, action, board_afterstate, vec_afterstate) -> torch.tensor:
        pass

    @torch.no_grad()
    def prediction(self, obs) -> list[int]:
        pass

    @torch.no_grad()
    def afterstate_prediction(self, obs) -> int:
        pass

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name, device):
        ckpt = torch.load(file_name, device)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        self.load_state_dict(ckpt)

    @torch.no_grad()
    def predict(self, obs):
        pass
