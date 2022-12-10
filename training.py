from __future__ import annotations
from csv import reader
from itertools import count

import matplotlib.pyplot as plt
import pandas as pd

from q_learning import *
from simulations import *
from test_teams import *
from utils.pokemon import *
from utils.team import *

def main():

    pokemon_df = pd.read_csv("pokemon.csv")
    fast_moves_df = pd.read_csv("fastmoves.csv")
    charged_moves_df = pd.read_csv("chargedmoves.csv")
    charged_moves_df = charged_moves_df[charged_moves_df.Effects.isnull()]


    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)

    # config for full state
    # agent = QLearningTeam(input_dim=72, n_actions=6, hidden_size=32, batch_size=32, lr=1e-5, weight_decay=1e-7)

    # config for hidden state
    agent = QLearningTeam(input_dim=36, n_actions=6, team_class=HiddenTeam, hidden_size=32, batch_size=32, lr=1e-5, weight_decay=1e-7)
    losses, moves, rewards = [], [], []
    classes = [HiddenTeam]
    for team_class in classes:
        l, m , r = agent.buffer_learn(pokemon_df, fast_moves_df, charged_moves_df, team_class, 200)
        losses.extend(l)
        moves.extend(m)
        rewards.extend(r)
    plt.title("Losses")
    plt.plot(losses)
    plt.yscale("log")
    plt.show()
    plt.title("Moves")
    plt.hist(moves)
    plt.show()
    plt.title("Rewards")
    plt.plot(rewards)
    plt.show()
    torch.save(agent.dqn.state_dict(), "hidden_model.pt")

    simulate_team_battle(agent, make_team(HiddenTeam), make_team(HiddenTeam))


if __name__ == "__main__":
    main()
