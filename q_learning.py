import numpy as np
from torch import concatenate, zeros, rand, argmax
import torch.nn as nn

import utils

N_ACTIONS = 3

class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

def action_to_tensor(action):
    t = zeros(N_ACTIONS)
    if action > 0:
        t[action - 1] += 1
    return t


def select_action(state, pokemon: utils.Pokemon, dqn: DQN, epsilon: float = 0.1):
    # return [0, 0, 0] or [1, 0, 0]
    actions = pokemon.get_actions(state) # subset of [0, 1, 2, 3]
    # exploration
    e = rand(1)
    if e <= epsilon:
        print('exploration')
        action = np.random.sample(actions) 
    # exploitation
    else:
        print("exploitation")
        print(state, )
        q_values = [dqn(concatenate((state, action_to_tensor(a)), dim=1)) for a in actions]
        print('q_values')
        print(q_values)
        action = argmax(q_values)
    return action_to_tensor(action)