import random
import numpy as np

# class ReplayMemory:
#     def __init__(self, max_size, input_shape) -> None:
#         self.memory_size = max_size
#         self.memory_counter = 0

#         self.memory_obs_1 = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
#         self.memory_obs_2 = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
#         self.memory_action = np.zeros((self.memory_size, *input_shape), dtype=np.long)
#         self.memory_reward = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
#         self.memory_next_obs_1 = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
#         self.memory_next_obs_2 = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
#         self.memory_terminal = np.zeros((self.memory_size, *input_shape), dtype=np.bool)

#     def store_transition(self, obs_1, obs_2, action, reward, next_obs_1, next_obs_2, done):
#         index = self.memory_counter % self.memory_size

#         self.memory_obs_1[index] = obs_1
#         self.memory_obs_2[index] = obs_2
#         self.memory_action[index] = action
#         self.memory_reward[index] = reward
#         self.memory_next_obs_1[index] = next_obs_1
#         self.memory_next_obs_2[index] = next_obs_2
#         self.memory_terminal[index] = done

#         self.memory_counter += 1

#     def sample_buffer(self, batch_size):
#         max_memory = min(self.memory_counter, self.memory_size)
#         batch = np.random.choice(max_memory, batch)

#         obs_1s = self.memory_obs_1[batch]
#         obs_2s = self.memory_obs_2[batch]
#         actions = self.memory_action[batch]
#         rewards = self.memory_reward[batch]
#         next_obs_1s = self.memory_next_obs_1[batch]
#         next_obs_2s = self.memory_next_obs_2[batch]
#         dones = self.memory_terminal[batch]

#         return obs_1s, obs_2s, actions, rewards, next_obs_1s, next_obs_2s, dones


from collections import namedtuple, deque


Transition = namedtuple(
    "Transition", ("state_1", "state_2", "action", "next_state_1", "next_state_2", "reward", "done", "win")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
