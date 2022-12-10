from __future__ import annotations
from csv import reader
from itertools import count

import matplotlib.pyplot as plt
import pandas as pd

from q_learning import *
from simulations import *
from test_teams import *
from utils.attacks import FastAttack, ChargedAttack
from utils.battle import Battle
from utils.pokemon import *
from utils.team import *

pokemon_df = pd.read_csv("pokemon.csv")
fast_moves_df = pd.read_csv("fastmoves.csv")
charged_moves_df = pd.read_csv("chargedmoves.csv")
charged_moves_df = charged_moves_df[charged_moves_df.Effects.isnull()]

seed = 1
random.seed(seed)
torch.manual_seed(seed)
agent = QLearningTeam(input_dim=72, n_actions=6, hidden_size=128, batch_size=32, lr=1e-5, weight_decay=1e-5)
losses, moves, rewards = [], [], []
# classes = [PokemonNoAttack, PokemonFastAttack, PokemonRandomAction, PokemonChargedAttack]
classes = [Pokemon]
for pokemon_class in classes:
    l, m , r = agent.buffer_learn(pokemon_df, fast_moves_df, charged_moves_df, pokemon_class, 200)
    losses.extend(l)
    moves.extend(m)
    rewards.extend(r)
# losses, moves, rewards = agent.learn(pokemon_df.iloc[:1], fast_moves_df, charged_moves_df, PokemonNoAttack, 100)
# losses, moves = agent.learn(pokemon_df, fast_moves_df, charged_moves_df, 100)
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

simulate_team_battle(agent, team1, team2)


# stunfisk1 = Pokemon(
#     *pokemon_df.iloc[0].tolist()[:6],
#     dict(zip(pokemon_df.columns[6:], pokemon_df.iloc[0].values[6:])),
#     mud_shot,
#     rock_slide,
#     earthquake,
# )
# print(stunfisk1)
# print(stunfisk1.get_state(stunfisk1))
# stunfisk2 = PokemonNoAttack(
#     *pokemon_df.iloc[0].tolist()[:6],
#     dict(zip(pokemon_df.columns[6:], pokemon_df.iloc[0].values[6:])),
#     mud_shot,
#     rock_slide,
#     earthquake,
# )
# stunfisk3 = PokemonFastAttack(
#     *pokemon_df.iloc[0].tolist()[:6],
#     dict(zip(pokemon_df.columns[6:], pokemon_df.iloc[0].values[6:])),
#     mud_shot,
#     rock_slide,
#     earthquake,
# )
# stunfisk4 = PokemonChargedAttack(
#     *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
# )
# stunfisk5 = PokemonRandomAction(
#     *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
# )
# simulate_battle(agent, stunfisk2, stunfisk3)
# stunfisk1 = Pokemon(
#     *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
# )
# simulate_battle(agent, stunfisk1, stunfisk3)
# stunfisk1 = Pokemon(
#     *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
# )
# simulate_battle(agent, stunfisk1, stunfisk4)
# stunfisk1 = Pokemon(
#     *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
# )
# simulate_battle(agent, stunfisk1, stunfisk5)

# battle = Battle(stunfisk1, stunfisk2)
# action = agent.select_action(*battle.get_state(), stunfisk3, 0)
# print(action)


# simulate_battle(agent, stunfisk3, stunfisk4)

# for i in count():
#     torch.manual_seed(i)
#     agent = QLearning(hidden_size=64)
#     agent.learn([stunfisk1, stunfisk2], 20)
#     battle = Battle(stunfisk3, stunfisk4)
#     battle.get_initial_state()
#     action = agent.select_action(battle.get_initial_state(), stunfisk3, 0)
#     print(action)
#     if action.item() == 1:
#         torch.save(agent.dqn.state_dict(), "model.pt")
