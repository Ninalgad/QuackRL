from src.networks.mlp import MLPNetwork


def get_network(name, config):
    net = None
    if name == 'mlp':
        net = MLPNetwork(config)

    if net is None:
        raise ValueError(f"'{name}' is not a registered network [mlp]")
    return net
