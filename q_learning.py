from itertools import count
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import concatenate, zeros, rand, argmax
import torch.nn as nn
from torch.nn.functional import one_hot
from tqdm import trange

from utils.attacks import FastAttack, ChargedAttack
from utils.battle import Battle
from utils.pokemon import *

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

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2])
        return self.model(x)


class QLearning:
    def __init__(
        self,
        hidden_size: int = 16,
        loss_function: Any = nn.SmoothL1Loss(reduction="none"),
        optimizer: Any = torch.optim.Adam,
        gamma: float = 0.999,
    ) -> None:
        self.dqn = DQN(hidden_size=hidden_size)
        # self.target = DQN()
        # self.target.load_state_dict(self.dqn.state_dict())
        self.loss_function = loss_function
        self.optimizer = optimizer(self.dqn.parameters(), lr=1e-5, weight_decay=1e-7)
        self.gamma = gamma

    def select_action(
        self, state_1: torch.Tensor, state_2: torch.Tensor, pokemon: Pokemon, epsilon: float = 0.5
    ) -> int:
        possible_actions = pokemon.get_actions()  # subset of [0, 1, 2, 3]
        # exploration
        e = rand(1)
        if e <= epsilon:
            action = torch.tensor(np.random.choice(possible_actions)).float()
            # print(f"exploration action picked: {action}")
        # exploitation
        else:
            # with torch.no_grad():
            q_values = self.dqn(state_1, state_2) * one_hot(possible_actions, N_ACTIONS).sum(axis=0)

            action = argmax(q_values)
            # print(f"exploitation action picked: {action}")
        return action.long().item()

    # def optimize(self, state_1, state_2, battle: Battle, battle_2: Battle):
    #     action_1 = self.select_action(state_1, battle.pokemon_1)
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

    def initialize_pokemon(self, pokemon_df: pd.DataFrame, fast_moves_df: pd.DataFrame, charged_moves_df: pd.DataFrame):
        choice_1 = np.random.choice(len(pokemon_df))
        choice_2 = np.random.choice(len(pokemon_df))
        mud_shot = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Mud Shot"].values[0, :4])
        rock_slide = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Rock Slide"].values[0, :3])
        earthquake = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Earthquake"].values[0, :3])
        pokemon_1 = Pokemon(*pokemon_df.iloc[choice_1].tolist()[:4], mud_shot, rock_slide, earthquake)
        pokemon_2 = PokemonNoAttack(*pokemon_df.iloc[choice_2].tolist()[:4], mud_shot, rock_slide, earthquake)
        return pokemon_1, pokemon_2

    def learn(
        self,
        pokemon_df: pd.DataFrame,
        fast_moves_df: pd.DataFrame,
        charged_moves_df: pd.DataFrame,
        max_epochs: int = 100,
    ):
        losses = []
        for epoch in trange(max_epochs):
            pokemon_1, pokemon_2 = self.initialize_pokemon(pokemon_df, fast_moves_df, charged_moves_df)
            battle = Battle(pokemon_1, pokemon_2)
            state_1, state_2 = battle.get_state()
            for t in count():
                action_1 = self.select_action(state_1, state_2, battle.pokemon_1, 1 / (epoch + 1))
                value = self.dqn(state_1, state_2)[action_1]
                action_2 = self.select_action(state_2, state_1, battle.pokemon_2, 1 / (epoch + 1))
                battle.update(action_1, action_2)
                next_state_1, next_state_2 = battle.get_state()
                reward = battle.get_reward(action_1) / 200
                expected_value = reward + self.gamma * torch.amax(
                    self.dqn(next_state_1, next_state_2)
                    * one_hot(battle.pokemon_1.get_actions(), N_ACTIONS).sum(axis=0)
                )
                loss = self.loss_function(value, expected_value)
                # if loss.item() > 1000:
                #     print(value_1)
                #     print(state_1)
                #     print(reward_1)
                #     print(expected_value_1)
                #     breakpoint()
                losses.append(loss.detach())

                self.dqn.zero_grad()
                loss.backward()
                self.optimizer.step()

                if battle.done():
                    break

        return losses
