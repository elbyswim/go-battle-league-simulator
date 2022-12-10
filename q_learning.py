from itertools import count
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import zeros, rand, argmax
import torch.nn as nn
from tqdm import trange

from replay_memory import ReplayMemory
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
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        gamma: float = 0.999,
    ) -> None:
        self.dqn = DQN(hidden_size=hidden_size)
        # self.target = DQN()
        # self.target.load_state_dict(self.dqn.state_dict())
        self.loss_function = nn.SmoothL1Loss(reduction="none")
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr, weight_decay=weight_decay)
        self.gamma = gamma
        self.memory = ReplayMemory(10_000)

    def select_action(
        self, state_1: torch.Tensor, state_2: torch.Tensor, pokemon: Pokemon, epsilon: float = 0.5
    ) -> int:
        possible_actions = pokemon.get_actions()  # tensor[bool]
        # exploration
        e = rand(1)
        # if True:
        if e < epsilon:
            action = np.random.choice(torch.arange(4).reshape(1, 4)[possible_actions])
        # exploitation
        else:
            with torch.no_grad():
                q_values = self.dqn(state_1, state_2)
            q_values[~possible_actions] = -float("inf")
            action = argmax(q_values).long().item()
        return action

    def initialize_pokemon(self, pokemon_df: pd.DataFrame, fast_moves_df: pd.DataFrame, charged_moves_df: pd.DataFrame, opponent_class):
        # choice_1 = np.random.choice(len(pokemon_df))
        # choice_2 = np.random.choice(len(pokemon_df))
        # mud_shot = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Mud Shot"].values[0, :4])
        # rock_slide = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Rock Slide"].values[0, :3])
        # earthquake = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Earthquake"].values[0, :3])
        # pokemon_1 = Pokemon(*pokemon_df.iloc[choice_1].tolist()[:4], mud_shot, rock_slide, earthquake)
        # pokemon_2 = PokemonNoAttack(*pokemon_df.iloc[choice_2].tolist()[:4], mud_shot, rock_slide, earthquake)

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
        pokemon_2 = opponent_class(
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
        opponent_class = PokemonRandomAction,
        max_epochs: int = 100,
    ):
        losses = []
        actions = []
        rewards = []
        # with torch.no_grad():
            # print(self.dqn(torch.tensor([[177.,   0.,   0.,   2.,  39.,  62.]]), torch.tensor([[177.,   0.,   0.,   2.,  39.,  62.]])))
        for epoch in trange(max_epochs):
            # print(f"################################# Epoch {epoch} ######################################")
            pokemon_1, pokemon_2 = self.initialize_pokemon(pokemon_df, fast_moves_df, charged_moves_df, opponent_class)
            battle = Battle(pokemon_1, pokemon_2)
            state_1, state_2 = battle.get_state()
            reward_episode = 0
            for t in count():
                action_1 = self.select_action(state_1, state_2, pokemon_1, 1 - epoch / max_epochs)
                assert pokemon_1.get_actions()[0, action_1]
                values = self.dqn(state_1, state_2)
                value = values[0, action_1]
                action_2 = self.select_action(state_2, state_1, pokemon_2, 1 - epoch / max_epochs)
                battle.update(action_1, action_2)
                next_state_1, next_state_2 = battle.get_state()
                reward = battle.get_reward(action_1)
                # reward_episode = reward + self.gamma * reward_episode
                reward_episode += reward
                with torch.no_grad():
                    next_values = self.dqn(next_state_1, next_state_2)
                next_values[~pokemon_1.get_actions()] = -float("inf")
                expected_value = reward + self.gamma * torch.amax(next_values)
                # print(action_1, values, expected_value)
                loss = self.loss_function(value, expected_value)
                # print(action_1, action_2)
                # print(battle)
                # print(reward)
                # print(values)
                # print(next_values)
                # print(expected_value)
                # print(loss)
                # breakpoint()
                losses.append(loss.detach())
                actions.append(action_1)

                self.dqn.zero_grad()
                loss.backward()
                self.optimizer.step()

                if battle.done():
                    break

                state_1, state_2 = battle.get_state()

            rewards.append(reward_episode)

        return losses, actions, rewards

    def optimize_model(self, epoch):

        from replay_memory import Transition

        if len(self.memory) < self.batch_size:
            return 1
        transitions = self.memory.sample(self.batch_size)
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
        action_batch = torch.tensor(batch.action).reshape(-1, 1)
        reward_batch = torch.tensor(batch.reward).reshape(-1, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.dqn(state_1_batch, state_2_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros_like(state_action_values)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.dqn(non_final_next_states_1, non_final_next_states_2).max(1, keepdim=True).values
        next_state_values[win_mask] = 1
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

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
        opponent_class = PokemonRandomAction,
        max_epochs: int = 100,
    ):
        losses = []
        actions = []
        rewards = []
        for epoch in trange(max_epochs):
            pokemon_1, pokemon_2 = self.initialize_pokemon(pokemon_df, fast_moves_df, charged_moves_df, opponent_class)
            battle = Battle(pokemon_1, pokemon_2)
            state_1, state_2 = battle.get_state()
            reward_episode = 0
            for t in range(50):
                action_1 = self.select_action(state_1, state_2, battle.pokemon_1, 1 / (epoch + 1))
                actions.append(action_1)
                action_2 = self.select_action(state_2, state_1, battle.pokemon_2, 1)
                battle.update(action_1, action_2)
                next_state_1, next_state_2 = battle.get_state()
                reward = battle.get_reward(action_1)
                reward_episode += reward
                done = battle.done()
                win = battle.win()

                # Store the transition in memory
                self.memory.push(state_1, state_2, action_1, next_state_1, next_state_2, reward, done, win)

                # Move to the next state
                state_1, state_2 = battle.get_state()

                # Perform one step of the optimization (on the policy network)
                loss = self.optimize_model(epoch)

                if done:
                    break
            losses.append(loss)
            rewards.append(reward_episode)

        return losses, actions, rewards
