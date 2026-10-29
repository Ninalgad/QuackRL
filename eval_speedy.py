from env import ScrabbleEnv, PlacementOnlyEnv
from src.agent import Agent
from src.typers import *
from src.networks import get_network
import click
import configparser


SPEEDY = "[SPEEDY]"


def run_game(player_0, player_1):
    agents = [player_0, player_1]

    env = PlacementOnlyEnv("../quackle/data")
    obs = env.reset()

    n = 0
    while not env.done:
        actor_id = env.to_play()
        agent = agents[actor_id]
        if agent == SPEEDY:
            action_idx = env.speedy_action(obs)
        else:
            state_obs = Observation(obs, is_afterstate=False)
            action_topk, action_feature_topk = agent.select_topk(state_obs)

            a = env.afterstates(obs, action_topk)
            afterstate_obs = Observation(a, is_afterstate=True)
            i = agent.select_action(afterstate_obs)
            action_idx = action_topk[i]

        obs, reward, done, info = env.step(action_idx)
        n += 1

    s = env.final_scores
    winner = -1 if s[0] == s[1] else int(np.argmax(s))
    final_scores = s
    return n, winner, final_scores


def results_report(stats):
    # {'speedy_wins': [], 'num_turns': [], 'speedy_score': [], 'player_score': []}
    num_games = len(stats['speedy_wins'])
    wr_s = 100. * np.mean(stats['speedy_wins']).item()
    wr_p = 100. - wr_s
    w_s = np.sum(stats['speedy_wins']).item()
    w_p = num_games - w_s

    sr_s = np.mean(stats['speedy_score']).item()
    sr_p = np.mean(stats['player_score']).item()
    tpg = np.mean(stats['num_turns']).item()
    return (f"speedy wr: {wr_s:.4f}, plr wr: {wr_p:.4f},"
            f"speedy wins: {w_s}, plr wins: {w_p},"
            f" avg speedy score: {sr_s}, avg plr score: {sr_p} avg turns per game: {tpg} ")


@click.command()
@click.option('--num-rounds', type=int, help="Number of training iterations")
@click.option('--config-file', type=str, default="", help="Path to .ini configuration file.")
def main(num_rounds, config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_network(config.get('arc', 'network'), config).to(device)
    net.eval()

    net.load(create_running_name(config), device)

    stats = {'speedy_wins': [], 'num_turns': [], 'speedy_score': [], 'player_score': []}
    for i in range(num_rounds):
        if (i % 2) == 0:
            player_1, player_2 = Agent(0, network=net, config=config), SPEEDY
            speedy_id = 1
        else:
            player_1, player_2 = SPEEDY, Agent(0, network=net, config=config)
            speedy_id = 0
        num_turns, winner_id, scores = run_game(player_1, player_2)
        stats['speedy_wins'].append(int(speedy_id == winner_id))
        stats['num_turns'].append(num_turns)
        stats['speedy_score'].append(scores[speedy_id])
        stats['player_score'].append(scores[(speedy_id + 1) % 2])

        summary = {f"avg {k}": round(np.mean(v).item(), 2) for k, v in stats.items()}
        print(f"Game: {i}, Stats: {summary}")

    print("RESULTS")
    print(results_report(stats))


if __name__ == '__main__':
    main()
