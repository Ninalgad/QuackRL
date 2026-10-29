from env import PlacementOnlyEnv
from src.agent import Agent
from src.replay import ReplayBuffer
from src.typers import *
from src.learner import Learner
from src.networks import get_network
import click
import configparser


def run_game(player_0: Agent, player_1: Agent, replay_buffer: ReplayBuffer):
    agents = [player_0, player_1]

    env = PlacementOnlyEnv("../quackle/data")
    obs = env.reset()
    reward = 0.0
    episode = []
    while not env.done:
        actor_id = env.to_play()
        agent = agents[actor_id]
        state_obs = Observation(obs, is_afterstate=False)
        action_topk, action_feature_topk = agent.select_topk(state_obs)

        a = env.afterstates(obs, action_topk)
        afterstate_obs = Observation(a, is_afterstate=True)
        i = agent.select_action(afterstate_obs)
        action_idx = action_topk[i]
        afterstate_obs.to_frame(i)
        state = State(
            state_observation=state_obs,
            afterstate_observation=afterstate_obs,
            reward=reward,
            player=actor_id,
            action=action_feature_topk[i],
        )
        episode.append(state)
        obs, reward, done, info = env.step(action_idx)

    # Send the episode to the replay.
    winner = np.argmax(env.final_scores).item()
    replay_buffer.save(episode, winner, abs(reward))
    return len(episode)


def create_players(net, config: Config):
    player_1 = Agent(0, network=net, config=config, training=True)
    player_2 = Agent(1, network=net, config=config, training=True)

    return player_1, player_2


@click.command()
@click.option('--num-rounds', type=int, help="Number of training iterations")
@click.option('--config-file', type=str, default="", help="Path to .ini configuration file.")
@click.option('--resume', is_flag=True, help="Resume training.")
def main(num_rounds, config_file, resume):
    config = configparser.ConfigParser()
    config.read(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_network(config.get('arc', 'network'), config).to(device)
    replay = ReplayBuffer(config)

    learner = Learner(net, replay_buffer=replay, config=config)
    round_num = 0
    games_per_round = config.getint('train', 'games_per_round')
    rounds_per_checkpoint = config.getint('train', 'rounds_per_checkpoint')
    if resume:
        # load and populate the buffer
        replay.load()
        # load model and optimizer weights
        meta = learner.load(create_running_name(config))
        round_num = meta['round_num'] + 1

    for _ in range(round_num, num_rounds):
        for _ in range(games_per_round):
            player_1, player_2 = create_players(learner.net, config)
            run_game(player_1, player_2, replay_buffer=replay)

        loss = learner.learn()

        if (round_num % rounds_per_checkpoint) == 0:
            print('Exporting, do not interrupt')
            learner.export(create_running_name(config), meta={'round_num': round_num})
            replay.export()
        print(f"Round: {round_num}, Loss: {loss:.4f}")

        round_num += 1

    learner.export(create_running_name(config), meta={'round_num': round_num})


if __name__ == '__main__':
    main()
