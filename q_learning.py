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

from replay_memory import ReplayMemory
from utils.attacks import FastAttack, ChargedAttack
from utils.battle import Battle
from utils.pokemon import *

INPUT_DIM = 12
N_ACTIONS = 4

BATCH_SIZE = 128


class DQN(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_size: int = 16, n_actions: int = 4) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], axis=1)
        return self.model(x)


class QLearning:
    def __init__(
        self,
        hidden_size: int = 16,
        # loss_function: Any = nn.SmoothL1Loss(reduction="none"),
        # optimizer: Any = torch.optim.Adam,
        gamma: float = 0.999,
    ) -> None:
        self.dqn = DQN(hidden_size=hidden_size)
        # self.target = DQN()
        # self.target.load_state_dict(self.dqn.state_dict())
        self.loss_function = nn.SmoothL1Loss(reduction="none")
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=1e-5, weight_decay=1e-7)
        self.gamma = gamma
        self.memory = ReplayMemory(10_000)

    def select_action(
        self, state_1: torch.Tensor, state_2: torch.Tensor, pokemon: Pokemon, epsilon: float = 0.5
    ) -> int:
        possible_actions = pokemon.get_actions()  # subset of [0, 1, 2, 3]
        # exploration
        e = rand(1)
        if e < epsilon:
            action = torch.tensor(np.random.choice(possible_actions)).float()
        # exploitation
        else:
            with torch.no_grad():
                q_values = masked_tensor(
                    self.dqn(state_1, state_2), one_hot(possible_actions, N_ACTIONS).sum(axis=0).reshape(1, -1).bool()
                )

            action = argmax(q_values)
        return action.long().item()

    def initialize_pokemon(self, pokemon_df: pd.DataFrame, fast_moves_df: pd.DataFrame, charged_moves_df: pd.DataFrame):
        # choice_1 = np.random.choice(len(pokemon_df))
        # choice_2 = np.random.choice(len(pokemon_df))
        # mud_shot = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Mud Shot"].values[0, :4])
        # rock_slide = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Rock Slide"].values[0, :3])
        # earthquake = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Earthquake"].values[0, :3])
        # pokemon_1 = Pokemon(*pokemon_df.iloc[choice_1].tolist()[:4], mud_shot, rock_slide, earthquake)
        # pokemon_2 = PokemonRandomAction(*pokemon_df.iloc[choice_2].tolist()[:4], mud_shot, rock_slide, earthquake)

        fast_move_1 = FastAttack(*fast_moves_df.iloc[np.random.choice(len(fast_moves_df))].tolist()[:4])
        fast_move_2 = FastAttack(*fast_moves_df.iloc[np.random.choice(len(fast_moves_df))].tolist()[:4])
        charged_move_1 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        charged_move_2 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        charged_move_3 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        charged_move_4 = ChargedAttack(*charged_moves_df.iloc[np.random.choice(len(charged_moves_df))].tolist()[:3])
        pokemon_1 = Pokemon(
            *pokemon_df.iloc[np.random.choice(len(pokemon_df))].tolist()[:4],
            fast_move_1,
            charged_move_1,
            charged_move_2,
        )
        pokemon_2 = PokemonRandomAction(
            *pokemon_df.iloc[np.random.choice(len(pokemon_df))].tolist()[:4],
            fast_move_2,
            charged_move_3,
            charged_move_4,
        )

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
                value = self.dqn(state_1, state_2)[0, action_1]
                action_2 = self.select_action(state_2, state_1, battle.pokemon_2, 1 / (epoch + 1))
                battle.update(action_1, action_2)
                next_state_1, next_state_2 = battle.get_state()
                reward = battle.get_reward(action_1) / 200
                with torch.no_grad():
                    next_values = self.dqn(next_state_1, next_state_2)
                expected_value = reward + self.gamma * torch.amax(
                    masked_tensor(
                        next_values,
                        one_hot(battle.pokemon_1.get_actions(), N_ACTIONS).sum(axis=0).reshape(1, -1).bool(),
                    )
                )
                expected_value = torch.tensor(expected_value.item())
                loss = self.loss_function(value, expected_value)
                losses.append(loss.detach())
                actions.append(action_1)

                self.dqn.zero_grad()
                loss.backward()
                self.optimizer.step()

                if battle.done():
                    break

        return losses, actions

    def optimize_model(self):

        from replay_memory import Transition

        if len(self.memory) < BATCH_SIZE:
            return 1
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), dtype=torch.bool)
        non_final_next_states_1 = torch.cat([s for s, d in zip(batch.next_state_1, batch.done) if not d])
        non_final_next_states_2 = torch.cat([s for s, d in zip(batch.next_state_2, batch.done) if not d])
        win_mask = torch.tensor(tuple(map(lambda s: s, batch.win)), dtype=torch.bool)
        state_1_batch = torch.cat(batch.state_1)
        state_2_batch = torch.cat(batch.state_2)
        action_batch = torch.tensor(batch.action).reshape(1, -1)
        reward_batch = torch.tensor(batch.reward).reshape(1, -1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.dqn(state_1_batch, state_2_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.dqn(non_final_next_states_1, non_final_next_states_2).max(1)[0]
        next_state_values[win_mask] = 1
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def buffer_learn(
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
                action_2 = self.select_action(state_2, state_1, battle.pokemon_2, 1 / (epoch + 1))
                battle.update(action_1, action_2)
                next_state_1, next_state_2 = battle.get_state()
                reward = battle.get_reward(action_1) / 200
                done = battle.done()
                win = battle.win()

                # Store the transition in memory
                self.memory.push(state_1, state_2, action_1, next_state_1, next_state_2, reward, done, win)

                # Move to the next state
                state_1, state_2 = battle.get_state()

                # Perform one step of the optimization (on the policy network)
                loss = self.optimize_model()

                losses.append(loss)

                if done:
                    break
        return losses
