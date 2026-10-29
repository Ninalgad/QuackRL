from src.typers import *
import json


class NetworkCacher:
    """An object to share the network between the self-play and training jobs."""
    def __init__(self):
        self._networks = {}

    def save_network(self, step: int, network: Any):
        self._networks[step] = network

    def load_network(self, step) -> Tuple[int, Any]:
        steps = list(self._networks.keys())
        training_step = steps[np.argmin([abs(k - step) for k in steps]).item()]
        return training_step, self._networks[training_step]

    def export(self, save_dir):
        with open(save_dir + 'cacher.json', 'w') as f:
            json.dump(self._networks, f)

    def import_from_dir(self, save_dir):
        with open(save_dir + 'cacher.json', 'r') as f:
            self._networks = json.load(f)
