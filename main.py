from q_learning import *
from utils import *
from torch import concatenate, zeros, rand, argmax, tensor
from csv import reader

dqn = DQN()

with open('fastmoves.csv', newline='') as csvfile:
    csvread = reader(csvfile)
    header = next(csvread)
    fastmoves_dict = {rows[0]:[int(rows[1]), int(rows[2]), int(rows[3]), rows[4]] for rows in csvread}
#outputs a dictionary of {name: [damage, energy, turns, type]}


with open('chargedmoves.csv', newline='') as csvfile:
    csvread = reader(csvfile)
    header = next(csvread)
    chargedmoves_dict = {rows[0]:[int(rows[1]), int(rows[2]), rows[3]] for rows in csvread}
#outputs a dictionary of {name: [damage, energy, type]}


with open('pokemon.csv', newline='') as csvfile:
    csvread = reader(csvfile)
    types = next(csvread)[6:24]
    pokemon_dict = {}
    for row in csvread:
        type_dict = {types[i]:row[6+i] for i in range(len(types))}
        characteristics = [int(row[1]), float(row[2]), float(row[3])]
        characteristics.append(row[4])
        characteristics.append(row[5])
        characteristics.append(type_dict)
        pokemon_dict.update({row[0]: characteristics})
#outputs a dictionary of {name: [hp, attack, def, type1, type2, {type: effectiveness}]}


mud_shot = FastAttack(*fastmoves_dict["Mud Shot"][0:3])
rock_slide = ChargedAttack(*chargedmoves_dict["Rock Slide"][0:2])
earthquake = ChargedAttack(*chargedmoves_dict["Earthquake"][0:2])
lock_on = FastAttack(*fastmoves_dict["Lock On"][0:3])
focus_blast = ChargedAttack(*chargedmoves_dict["Focus Blast"][0:2])
flash_cannon = ChargedAttack(*chargedmoves_dict["Flash Cannon"][0:2])

Stunfisk = Pokemon(*pokemon_dict["Stunfisk (Galarian)"][0:3], mud_shot, rock_slide, earthquake)
Registeel = Pokemon(*pokemon_dict["Registeel"][0:3], lock_on, focus_blast, flash_cannon)

agent = QLearning()

agent.learn([Stunfisk, Registeel])
