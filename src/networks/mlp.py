import torch.nn as nn
from src.typers import *
from src.networks.networks import Network


class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff, output_dim=None):
        super().__init__()
        output_dim = d_model if (output_dim is None) else output_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, f_x):
        return x + self.alpha * f_x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()

        self.mlp = PointWiseFeedForwardNetwork(d_model, dff)
        self.rezero = ReZero()

    def forward(self, x):
        return self.rezero(x, self.mlp(x))


class MLPNetwork(Network):
    def __init__(self, config: Config):
        super().__init__(config)
        num_blocks = config.getint('arc', 'num_blocks')
        d_model = config.getint('arc', 'd_model')
        dff = config.getint('arc', 'dff')
        self.top_k = config.getint('arc', 'top_k')
        board_dim = config.getint('constants', 'board_dim')
        vec_dim = config.getint('constants', 'vec_dim')
        action_dim = config.getint('constants', 'action_dim')
        self.temp = config.getfloat('train', 'temp')

        self.proj_b = PointWiseFeedForwardNetwork(15*15*board_dim, dff, d_model)
        self.proj_v = nn.Linear(vec_dim, d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, dff)
            for _ in range(num_blocks)
        ])
        self.state_value_layer = PointWiseFeedForwardNetwork(d_model, dff, 1)
        self.action_value_layer = PointWiseFeedForwardNetwork(action_dim + d_model, d_model, 1)
        self.d_model = d_model
        self.num_blocks = num_blocks

    def __str__(self):
        return f"MLPNetwork(dim={self.d_model}, blocks={self.num_blocks})"

    def forward_representation(self, board, vec):
        b = torch.flatten(board, start_dim=1)
        b, v = self.proj_b(b), self.proj_v(vec)
        x = (b + v) / (2*self.d_model ** 0.5)

        for layer in self.encoder_layers:
            x = layer(x)
        return x / self.num_blocks

    def forward_prediction(self, r):
        # r: (B, d_model) the output of forward_representation
        return self.state_value_layer(r).tanh()

    def forward_actions(self, actions: torch.tensor, r: torch.tensor):
        assert r.shape[0] == actions.shape[0], (r.shape, actions.shape)
        x = torch.cat((actions, r), dim=-1)
        return self.action_value_layer(x).tanh()

    def prune_actions(self, action, r):
        assert r.shape[0] == 1, r.shape
        # r: (1, d_model) the output of forward_representation
        # action: (num_actions, d_model)
        if action.shape[0] <= self.top_k:
            return list(range(action.shape[0]))

        q = self.forward_actions(action, r.repeat(action.shape[0], 1))

        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(q) + 1e-20) + 1e-20)
            q = q + self.temp * gumbel_noise.to(q.device)
        topk_values, topk_indices = torch.topk(q.squeeze(-1), k=self.top_k)
        topk_indices = topk_indices.cpu().numpy()
        return topk_indices

    def forward(self, board_state, vec_state, action, board_afterstate, vec_afterstate):
        # for training only
        h = self.forward_representation(board_state, vec_state)
        q0 = self.forward_prediction(h)
        q1 = self.forward_actions(action, h.detach())

        q2 = self.forward_prediction(
            self.forward_representation(board_afterstate, vec_afterstate)
        )
        return q0, q1, q2, h

    @torch.no_grad()
    def prediction(self, obs: Observation):
        # stage 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inp = obs.get_inputs(device)
        b, v = inp.board.unsqueeze(0), inp.vec.unsqueeze(0)
        a = inp.action

        h = self.forward_representation(b, v)
        top_action_idxs = self.prune_actions(a, h)
        return top_action_idxs

    @torch.no_grad()
    def afterstate_prediction(self, obs) -> int:
        # stage 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inp = obs.get_inputs(device)
        h = self.forward_representation(inp.board, inp.vec)
        q = self.forward_prediction(h)

        idx = np.argmax(q.squeeze(-1).squeeze(-1).cpu().numpy()).item()
        return idx
