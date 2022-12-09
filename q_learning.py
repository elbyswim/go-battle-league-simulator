from itertools import count
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import concatenate, zeros, rand, argmax
from torch.masked import masked_tensor
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
        # breakpoint()
        e = rand(1)
        if e < epsilon:
            action = torch.tensor(np.random.choice(possible_actions)).float()
            # print(f"exploration action picked: {action}")
        # exploitation
        else:
            with torch.no_grad():
            # q_values = self.dqn(state_1, state_2) * one_hot(possible_actions, N_ACTIONS).sum(axis=0)
                q_values = masked_tensor(self.dqn(state_1, state_2), one_hot(possible_actions, N_ACTIONS).sum(axis=0).bool())

            action = argmax(q_values)
            # breakpoint()
            # print(f"exploitation action picked: {action}")
        return action.long().item()

    def initialize_pokemon(self, pokemon_df: pd.DataFrame, fast_moves_df: pd.DataFrame, charged_moves_df: pd.DataFrame):
        choice_1 = np.random.choice(len(pokemon_df))
        choice_2 = np.random.choice(len(pokemon_df))
        mud_shot = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Mud Shot"].values[0, :4])
        rock_slide = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Rock Slide"].values[0, :3])
        earthquake = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Earthquake"].values[0, :3])
        pokemon_1 = Pokemon(*pokemon_df.iloc[choice_1].tolist()[:4], mud_shot, rock_slide, earthquake)
        pokemon_2 = PokemonRandomAction(*pokemon_df.iloc[choice_2].tolist()[:4], mud_shot, rock_slide, earthquake)

        # fast_move_1 = FastAttack(*fast_moves_df.iloc[np.random.choice(len(fast_moves_df))].tolist()[:4])
        # fast_move_2 = FastAttack(*fast_moves_df.iloc[np.random.choice(len(fast_moves_df))].tolist()[:4])
        # charged_move_1 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        # charged_move_2 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        # charged_move_3 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        # charged_move_4 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        # pokemon_1 = Pokemon(*pokemon_df.iloc[choice_1].tolist()[:4], fast_move_1, charged_move_1, charged_move_2)
        # pokemon_2 = PokemonRandomAction(*pokemon_df.iloc[choice_2].tolist()[:4], fast_move_2, charged_move_3, charged_move_4)

        return pokemon_1, pokemon_2

    def learn(
        self,
        pokemon_df: pd.DataFrame,
        fast_moves_df: pd.DataFrame,
        charged_moves_df: pd.DataFrame,
        max_epochs: int = 100,
    ):
        losses = []
        actions = []
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
                losses.append(loss.detach())
                actions.append(action_1)

                self.dqn.zero_grad()
                loss.backward()
                self.optimizer.step()

                if battle.done():
                    break

        return losses, actions
