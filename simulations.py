import matplotlib.pyplot as plt
import random
import torch

from q_learning import QLearning
from utils.battle import Battle
from utils.pokemon import Pokemon
from utils.team import Team
from utils.team_battle import TeamBattle

def simulate_battle(agent: QLearning, pokemon_1: Pokemon, pokemon_2: Pokemon, max_steps: int = 1000) -> None:
    battle = Battle(pokemon_1, pokemon_2)
    hp_1 = []
    hp_2 = []
    for i in range(max_steps):
        hp_1.append(battle.pokemon_1.hp)
        hp_2.append(battle.pokemon_2.hp)
        if battle.done():
            break
        # action = agent.select_action(*battle.get_state(), pokemon_1, 0)
        action = random.choice(torch.arange(4).reshape(1, 4)[pokemon_1.get_actions()]).item()
        action_2 = random.choice(torch.arange(4).reshape(1, 4)[pokemon_2.get_actions()]).item()

        battle.update(action, action_2)

    plt.plot(hp_1, label="Pokemon 1 HP")
    plt.plot(hp_2, label="Pokemon 2 HP")
    plt.legend()
    plt.show()

def simulate_team_battle(agent: QLearning, team_1: Team, team_2: Team, max_steps: int = 600) -> None:
    battle = TeamBattle(team_1, team_2)
    hp = torch.zeros((6, max_steps))

    for i in range(max_steps):
        hp[0, i] = battle.team_1.pokemon_1.hp
        hp[1, i] = battle.team_1.pokemon_2.hp
        hp[2, i] = battle.team_1.pokemon_3.hp
        hp[3, i] = battle.team_2.pokemon_1.hp
        hp[4, i] = battle.team_2.pokemon_2.hp
        hp[5, i] = battle.team_2.pokemon_3.hp
        if battle.done():
            end = i + 5
            break
        action_1 = agent.select_action(*battle.get_state(), team_1, 0)
        action_2 = random.choice(torch.arange(6).reshape(1, 6)[team_2.get_actions()]).item()
        battle.update(action_1, action_2)

    plt.plot(hp[0, :end], label="Team 1 Poke 1")
    plt.plot(hp[1, :end], label="Team 1 Poke 2")
    plt.plot(hp[2, :end], label="Team 1 Poke 3")
    plt.plot(-hp[3, :end], label="Team 2 Poke 1")
    plt.plot(-hp[4, :end], label="Team 2 Poke 2")
    plt.plot(-hp[5, :end], label="Team 2 Poke 3")
    plt.hlines(0, 0, end, 'k')
    plt.legend()
    plt.show()