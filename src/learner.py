from src.typers import *
from src.replay import ReplayBuffer
from src.networks.networks import Network


def value_or_reward_loss(prediction, target):
    assert prediction.shape == target.shape, (prediction.shape, target.shape)
    return ((prediction - target) ** 2).mean()


class Learner:
    """A learner to update the network weights."""

    def __init__(self, net: Network,
                 config: Config,
                 replay_buffer: ReplayBuffer):
        self.config = config
        self.replay_buffer = replay_buffer
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = config.getfloat('train', 'lr')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.num_steps = 0

    def sample_batch(self):
        traj_batch = self.replay_buffer.sample()
        training_batch = {'board_state': [], 'vec_state': [], 'action': [],
                          'board_afterstate': [], 'vec_afterstate': [],
                          'value': []}

        for state in traj_batch:
            training_batch['board_state'].append(state.state_observation.board)
            training_batch['vec_state'].append(state.state_observation.vec)
            training_batch['board_afterstate'].append(state.afterstate_observation.board)
            training_batch['vec_afterstate'].append(state.afterstate_observation.vec)
            training_batch['action'].append(state.action)

            training_batch['value'].append(state.reward)

        training_batch = {k: torch.tensor(np.array(v, "float32")).float().to(self.device)
                          for (k, v) in training_batch.items()}
        training_batch['value'] = training_batch['value'].unsqueeze(-1)

        return training_batch

    def train_step(self, batch):
        """Applies a single training step."""
        self.optimizer.zero_grad()
        tar = batch['value']
        del batch['value']
        q0, q1, q2, _ = self.net(**batch)
        loss =  value_or_reward_loss(q0, tar)
        loss += value_or_reward_loss(q1, tar)
        loss += value_or_reward_loss(q2, tar)
        loss.backward()
        self.optimizer.step()

        self.num_steps += 1
        return loss.detach().cpu().numpy()

    def learn(self):
        losses = []
        # steps = self.config.getint('train', 'steps')
        sps = self.config.getfloat('train', 'steps_per_sample')
        bs = self.config.getint('train', 'batch_size')
        steps = int(sps * self.replay_buffer.new_samples / bs) + 1
        for _ in range(steps):
            batch = self.sample_batch()
            loss = self.train_step(batch)
            losses.append(loss)
        self.replay_buffer.new_samples = 0
        return np.mean(losses).item()

    def export(self, save_file, meta: Dict = None):
        state = {
            'steps': self.num_steps,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if meta is not None:
            state.update(meta)
        torch.save(state, save_file)

    def load(self, path: str) -> Dict:
        ckpt = torch.load(path, self.device)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.num_steps = ckpt['steps']
        del ckpt['steps'], ckpt['model_state_dict'], ckpt['optimizer_state_dict']
        return ckpt
