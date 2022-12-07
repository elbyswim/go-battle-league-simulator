from csv import reader
from itertools import count

import matplotlib.pyplot as plt

from q_learning import *
from utils import *

with open("fastmoves.csv", newline="") as csv_file:
    csv_read = reader(csv_file)
    header = next(csv_read)
    fast_moves_dict = {rows[0]: [int(rows[1]), int(rows[2]), int(rows[3]), rows[4]] for rows in csv_read}
# outputs a dictionary of {name: [damage, energy, turns, type]}

with open("chargedmoves.csv", newline="") as csv_file:
    csv_read = reader(csv_file)
    header = next(csv_read)
    charged_moves_dict = {rows[0]: [int(rows[1]), int(rows[2]), rows[3]] for rows in csv_read}
# outputs a dictionary of {name: [damage, energy, type]}

with open("pokemon.csv", newline="") as csv_file:
    csv_read = reader(csv_file)
    types = next(csv_read)[6:24]
    pokemon_dict = {}
    for row in csv_read:
        type_dict = {types[i]: row[6 + i] for i in range(len(types))}
        characteristics = [int(row[1]), float(row[2]), float(row[3])]
        characteristics.append(row[4])
        characteristics.append(row[5])
        characteristics.append(type_dict)
        pokemon_dict.update({row[0]: characteristics})
# outputs a dictionary of {name: [hp, attack, def, type1, type2, {type: effectiveness}]}

mud_shot = FastAttack(*fast_moves_dict["Mud Shot"][0:3])
rock_slide = ChargedAttack(*charged_moves_dict["Rock Slide"][0:2])
earthquake = ChargedAttack(*charged_moves_dict["Earthquake"][0:2])
lock_on = FastAttack(*fast_moves_dict["Lock On"][0:3])
focus_blast = ChargedAttack(*charged_moves_dict["Focus Blast"][0:2])
flash_cannon = ChargedAttack(*charged_moves_dict["Flash Cannon"][0:2])

Stunfisk = Pokemon(*pokemon_dict["Stunfisk (Galarian)"][0:3], mud_shot, rock_slide, earthquake)
Registeel = Pokemon(*pokemon_dict["Registeel"][0:3], lock_on, focus_blast, flash_cannon)

stunfisk1 = Pokemon(*pokemon_dict["Stunfisk (Galarian)"][0:3], mud_shot, rock_slide, earthquake)
stunfisk2 = Pokemon(*pokemon_dict["Stunfisk (Galarian)"][0:3], mud_shot, rock_slide, earthquake)
stunfisk3 = Pokemon(*pokemon_dict["Stunfisk (Galarian)"][0:3], mud_shot, rock_slide, earthquake)
stunfisk4 = Pokemon(*pokemon_dict["Stunfisk (Galarian)"][0:3], mud_shot, rock_slide, earthquake)


def simulate_battle(agent: QLearning, pokemon_1: Pokemon, pokemon_2: Pokemon, max_steps: int = 1000) -> None:
    battle = Battle(pokemon_1, pokemon_2)
    state = battle.get_initial_state()
    for i in range(max_steps):
        print(f"Step {i}")
        print(f"Current state: {state}")
        action = agent.select_action(state, pokemon_1, 0)
        print(f"Selected action: {action}")
        state = battle.get_next_state(state, action, torch.tensor(0.0))
        if state[0] <= 0 and state[6] <= 0:
            break

torch.manual_seed(0)
agent = QLearning(hidden_size=64)
# agent.learn([stunfisk1, stunfisk2], 20)
losses = agent.learn([stunfisk1, stunfisk2], 20)
plt.plot(losses)
plt.show()
battle = Battle(stunfisk3, stunfisk4)
battle.get_initial_state()
action = agent.select_action(battle.get_initial_state(), stunfisk3, 0)
print(action)



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
