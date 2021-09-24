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

        # create batch dictionary which is going to be cast into tensors
        # batch = dict(
        #     states=state,
        #     next_states=next_state,
        #     actions=action,
        #     rewards=reward,
        #     dones=done,
        # )
        # The sample_batch method is called when learning. Because the
        # values that are taken from sample buffer are directly fed into
        # the respective network, they need to be transformed into a
        # torch tensor
        # return {k: T.as_tensor(v, dtype=T.float32) for k, v in batch.items()}
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)