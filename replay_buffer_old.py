#%%
import numpy as np
from collections import deque
import random
import torch as T

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))

        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

#%%
if __name__ == "__main__":
    STATE_DIMS = 2
    ACTION_DIMS = 1
    MEM_FACTOR = 100
    N_PATHS = 1

    replay_buffer = ReplayBuffer(MEM_FACTOR)
    print(replay_buffer.buffer)

    # Generate arbitrary state transition
    states = np.arange(N_PATHS * STATE_DIMS).reshape((N_PATHS, STATE_DIMS))
    next_states = np.arange(N_PATHS * STATE_DIMS).reshape((N_PATHS, STATE_DIMS))
    actions = np.arange(N_PATHS * ACTION_DIMS).reshape((N_PATHS, ACTION_DIMS))
    rewards = np.arange(N_PATHS)
    dones = np.zeros(N_PATHS)

    replay_buffer.push(states, actions, rewards, next_states, dones)
    replay_buffer.sample(1)
# %%
