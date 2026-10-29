from src.typers import *
from src.networks.networks import Network


class Agent:
    def __init__(self, player_id: int, network: Network, config: Config, training=False):
        self.player_id = player_id
        self.config = config
        self.network = network
        self.epsilon = config.getfloat('train', 'epsilon_greedy')
        self.training = training

    def __str__(self):
        return f"Agent(player_id={self.player_id}, network={self.network})"

    def __repr__(self):
        return str(self)

    def select_topk(self, obs: Observation) -> tuple[list[int], Any]:
        act_topk = self.network.prediction(obs)
        act_feat = obs.action[act_topk]
        return act_topk, act_feat

    def select_action(self, obs: Observation) -> int:
        if self.training and (np.random.uniform() < self.epsilon):
            act_idx = np.random.choice(len(obs.board))
        else:
            act_idx = self.network.afterstate_prediction(obs)
        return act_idx
