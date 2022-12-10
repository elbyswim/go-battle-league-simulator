from csv import reader
from itertools import count

import matplotlib.pyplot as plt
import pandas as pd

from q_learning import *
from utils.attacks import FastAttack, ChargedAttack
from utils.battle import Battle
from utils.pokemon import *


# read data
pokemon_df = pd.read_csv("pokemon.csv")
fast_moves_df = pd.read_csv("fastmoves.csv")
charged_moves_df = pd.read_csv("chargedmoves.csv")
mud_shot = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Mud Shot"].values[0, :4])
rock_slide = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Rock Slide"].values[0, :3])
earthquake = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Earthquake"].values[0, :3])
# lock_on = FastAttack(*fast_moves_dict["Lock On"][0:3])
# focus_blast = ChargedAttack(*charged_moves_dict["Focus Blast"][0:2])
# flash_cannon = ChargedAttack(*charged_moves_dict["Flash Cannon"][0:2])
Stunfisk = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
# Registeel = Pokemon(*pokemon_df_dict["Registeel"][0:3], lock_on, focus_blast, flash_cannon)


def simulate_battle(agent: QLearning, pokemon_1: Pokemon, pokemon_2: Pokemon, max_steps: int = 1000) -> None:
    battle = Battle(pokemon_1, pokemon_2)
    hp_1 = []
    hp_2 = []
    for i in range(max_steps):
        hp_1.append(battle.pokemon_1.hp)
        hp_2.append(battle.pokemon_2.hp)
        if battle.done():
            break
        action = agent.select_action(*battle.get_state(), pokemon_1, 0)
        battle.update(action, torch.arange(4).reshape(1, 4)[pokemon_2.get_actions()][0].item())

    plt.plot(hp_1, label="Pokemon 1 HP")
    plt.plot(hp_2, label="Pokemon 2 HP")
    plt.legend()
    plt.show()


torch.manual_seed(0)
agent = QLearning(hidden_size=16, batch_size=256, lr=1e-5, weight_decay=1e-7)
losses, moves, rewards = [], [], []
# classes = [PokemonNoAttack, PokemonFastAttack, PokemonRandomAction, PokemonChargedAttack]
classes = [Pokemon]
for pokemon_class in classes:
    l, m , r = agent.buffer_learn(pokemon_df.iloc[:10], fast_moves_df, charged_moves_df, pokemon_class, 200)
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

stunfisk1 = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
print(stunfisk1)
print(stunfisk1.get_state(stunfisk1))
stunfisk2 = PokemonNoAttack(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
stunfisk3 = PokemonFastAttack(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
stunfisk4 = PokemonChargedAttack(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
stunfisk5 = PokemonRandomAction(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
simulate_battle(agent, stunfisk1, stunfisk2)
stunfisk1 = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
simulate_battle(agent, stunfisk1, stunfisk3)
stunfisk1 = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
simulate_battle(agent, stunfisk1, stunfisk4)
stunfisk1 = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :4], mud_shot, rock_slide, earthquake
)
simulate_battle(agent, stunfisk1, stunfisk5)

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
