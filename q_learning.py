from itertools import count
from typing import Any

import numpy as np
import torch
from torch import concatenate, zeros, rand, argmax
import torch.nn as nn
from torch.nn.functional import one_hot
from tqdm import trange

from utils import *

INPUT_DIM = 12
N_ACTIONS = 4

class DQN(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_size: int = 16, n_actions: int = 4) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class QLearning():
    def __init__(self, hidden_size: int = 16, loss_function: Any = nn.SmoothL1Loss(reduction='none'), optimizer: Any = torch.optim.Adam, gamma: float = 0.999) -> None:
        self.dqn = DQN(hidden_size=hidden_size)
        # self.target = DQN()
        # self.target.load_state_dict(self.dqn.state_dict())
        self.loss_function = loss_function
        self.optimizer = optimizer(self.dqn.parameters(), weight_decay=1e-5)
        self.gamma = gamma

    def select_action(self, state, pokemon: Pokemon, epsilon: float = 0.5) -> int:
        possible_actions = pokemon.get_actions(state) # subset of [0, 1, 2, 3]
        # exploration
        e = rand(1)
        if e <= epsilon:
            action = torch.tensor(np.random.choice(possible_actions)).float()
            # print(f"exploration action picked: {action}")
        # exploitation
        else:
            # with torch.no_grad():
            q_values = self.dqn(state) * one_hot(possible_actions, N_ACTIONS).sum(axis=0)
            action = argmax(q_values)    
            # print(f"exploitation action picked: {action}")
        return action.long() #.item()


    # def optimize(self, state_1, state_2, battle_1: Battle, battle_2: Battle):
    #     action_1 = self.select_action(state_1, battle_1.pokemon_1)
    #     value_1 = self.dqn(state_1)[action_1]
    #     action_2 = self.select_action(state_2, battle_2.pokemon_2)
    #     value_2 = self.dqn(state_2)[action_2]
    #     next_state_1 = battle_1.get_next_state(state_1, action_1, action_2)
    #     next_state_2 = battle_2.get_next_state(state_2, action_2, action_1)
    #     reward_1 = battle_1.get_reward(next_state_1)
    #     reward_2 = battle_2.get_reward(next_state_2)
    #     expected_value_1 = reward_1 + self.gamma * torch.amax(self.dqn(next_state_1) * one_hot(battle_1.pokemon_1.get_actions(next_state_1)).sum(axis=0))
    #     expected_value_2 = reward_2 + self.gamma * torch.amax(self.dqn(next_state_2) * one_hot(battle_2.pokemon_2.get_actions(next_state_2)).sum(axis=0))

    #     loss = self.loss_function(value_1, expected_value_1) + self.loss_function(value_2, expected_value_2)
    #     self.optimizer.zero_grad()
    #     loss.backwards()
    #     self.optimizer.step()

    #     return next_state_1, next_state_2


    def learn(self, list_of_pokemon, max_epochs=100):
        losses = []
        for epoch in trange(max_epochs):
            pokemon_1 = np.random.choice(list_of_pokemon)
            pokemon_2 = np.random.choice(list_of_pokemon)
            battle_1 = Battle(pokemon_1, pokemon_2)
            battle_2 = Battle(pokemon_2, pokemon_1)
            state_1 = battle_1.get_initial_state()
            state_2 = battle_2.get_initial_state()
            for t in count():
                action_1 = self.select_action(state_1, battle_1.pokemon_1, 1 / (epoch + 1))
                value_1 = self.dqn(state_1)[action_1]
                action_2 = self.select_action(state_2, battle_2.pokemon_2, 1 / (epoch + 1))
                # with torch.no_grad():
                value_2 = self.dqn(state_2)[action_2]
                next_state_1 = battle_1.get_next_state(state_1, action_1, action_2)
                next_state_2 = battle_2.get_next_state(state_2, action_2, action_1)
                reward_1 = battle_1.get_reward(action_1, next_state_1)
                reward_2 = battle_2.get_reward(action_2, next_state_2)
                expected_value_1 = reward_1 + self.gamma * torch.amax(self.dqn(next_state_1) * one_hot(battle_1.pokemon_1.get_actions(next_state_1), N_ACTIONS).sum(axis=0))
                expected_value_2 = reward_2 + self.gamma * torch.amax(self.dqn(next_state_2) * one_hot(battle_2.pokemon_2.get_actions(next_state_2), N_ACTIONS).sum(axis=0))
                loss = self.loss_function(value_1, expected_value_1) + self.loss_function(value_2, expected_value_2)
                losses.append(loss.detach())

                self.dqn.zero_grad()
                loss.backward()
                self.optimizer.step()

                if next_state_1[0] <= 0 or next_state_1[3] <= 0:
                    break

                state_1 = next_state_1
                state_1, state_2 = next_state_1, next_state_2
                # self.target.load_state_dict(self.dqn.state_dict())
        return losses
