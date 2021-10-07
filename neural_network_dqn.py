#%%
import torch.nn as nn
import math, random
import torch
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import os
import logging
# import logging.config

# Setup logger
logger = logging.getLogger(__name__)

import numpy as np

def combined_shape(length, shape=None):
    """
    This function takes the size of the replay buffer and the shape of a
    matrix to return a tuple with specific dimensions for matrix
    initialization.

    Args:
        length (int): Number of sample slots in the replay buffer.
        shape (tuple, optional): Defaults to None. Typical inputs are
            state_dims or action_dims

    Returns:
        tuple: Returns a tuple with dimensions for matrix initialization
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class NetworkUtils:
    """
    Provides useful functionalities for neural network classes.  Inherit
    from this class when creating a new neural network to use its
    functionalities.

    Functionalities include:
        * saving and loading network parameters
    """

    def save_checkpoint(self):
        if hasattr(self, "checkpoint_dir") and hasattr(self, "name"):
            checkpoint_file_path = os.path.join(self.checkpoint_dir, self.name)
            logger.info("--- Saving Checkpoint ---")
            torch.save(self.state_dict(), checkpoint_file_path)
        else:
            logger.error(
                "--- Could not save checkpoint, some attributes are missing ---"
            )

    def load_checkpoint(self):
        if hasattr(self, "checkpoint_dir") and hasattr(self, "name"):
            checkpoint_file_path = os.path.join(self.checkpoint_dir, self.name)
            logger.info("--- Loading Checkpoint ---")
            self.load_state_dict(torch.load(checkpoint_file_path))
            # optional: self.eval()
            # Remember that you must call model.eval() to set dropout and
            # batch normalization layers to evaluation mode before running
            # inference. Failing to do this will yield inconsistent
            # inference results.
        else:
            logger.error(
                "--- Could not load checkpoint, some attributes are missing ---"
            )
class DQN(nn.Module, NetworkUtils):
    def __init__(self, 
                name:str, 
                state_dims: int,
                action_dims: int,
                env,
                checkpoint_dir: str = "models/tmp/dqn",):
        super(DQN, self).__init__()

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.env = env

        self.layers = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.shape[0])
        )

        # Initialize device to which the network should be passed to
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Pass the Critic network to said device
        self.to(self.device)
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if self.env.rng.rand() > epsilon:
            # print("greedy")
            
            q_value = self.forward(torch.tensor(state).to(self.device))
            # print(f"{q_value=}")
            action = q_value
            action  = q_value.argmax().item()
            # print(f"{action=}")

        else:
            # print("random")
            action = self.env.action_space.sample()
            # print(f"{action=}")
        return action

#%%
if __name__ == "__main__":
    from buffer import ReplayBuffer
    import numpy as np

    dqn = DQN("test_dqn", 2, 1)
    dqn.save_checkpoint()

    STATE_DIMS = 2
    ACTION_DIMS = 1
    MEM_FACTOR = 100
    N_PATHS = 5

    replay_buffer = ReplayBuffer(STATE_DIMS, ACTION_DIMS, MEM_FACTOR, N_PATHS)
    # Generate arbitrary state transition
    states = np.arange(N_PATHS * STATE_DIMS).reshape((N_PATHS, STATE_DIMS))
    next_states = np.arange(N_PATHS * STATE_DIMS).reshape((N_PATHS, STATE_DIMS))
    actions = np.arange(N_PATHS * ACTION_DIMS).reshape((N_PATHS, ACTION_DIMS))
    rewards = np.arange(N_PATHS)
    dones = np.zeros(N_PATHS)

    replay_buffer.store_transition(states, actions, rewards, next_states, dones)
    memory_sample = replay_buffer.sample_batch(1)

    states = memory_sample["states"].to(dqn.device)
    actions = memory_sample["actions"].to(dqn.device)
    rewards = memory_sample["rewards"].to(dqn.device)
    next_states = memory_sample["next_states"].to(dqn.device)
    dones = memory_sample["dones"].to(dqn.device)

    action = dqn.act(states, 0.0001)
# %%
