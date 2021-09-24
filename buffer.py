# %%
import numpy as np
from numpy import ndarray

import torch as T

import neural_network_dqn as core

# Import logging module
import logging
import logging.config

# Setup logger
logger = logging.getLogger(__name__)
# logging.config.fileConfig("logging.conf")
# logging.basicConfig(level=logging.INFO)


class ReplayBuffer:
    """
    Stores set amount of past experiences for learning (exploitation).

    The ReplayBuffer stores tuples with the structure: [states,
    next_states, actions, rewards, dones]. The replay buffer can also be 
    referred to as replay memory.
    These tuples contain:
     
        1.  The states passed from the environment to the ActorNetwork.
        2.  The actions that were chosen from the ActorNetwork for the
            passed states.
        3.  The resulting states for the chosen actions.
        4.  The reward received for the chosen actions.
        5.  The information whether the states were terminal states or not.

    Args:
        state_dims (int): Number of dimensions of a typical state.
            Example: states = (path_index, [age, p_income_factor, wealth])
            then state_dims = 3
        action_dims (int): Number of dimensions of the action space.
            Example : action = [bonds, consumption, stocks], action_dims = 3
        mem_factor (int): Hyperparameter used to provide a constant
            relative scaling between maximum buffer size and number of paths
        n_paths (int, optional): Number of games that are played
            simultaneously . Defaults to 1.
    """

    def __init__(
        self, state_dims: int, action_dims: int, mem_factor: int, n_paths: int = 1,
    ):

        logger.info("""Initializing replay memory...""")
        logger.info(f"""Memory factor set to {mem_factor}.""")
        logger.info(f"""Number of paths set to {n_paths}.""")

        # The multiplication of n_paths with mem_factor ensures both:
        # first: When storing new state-action pairs into the buffer, we
        # will not exceed the range of the buffer.
        # second: Constant relative scaling between n_paths and the
        # overall buffer size with mem_factor as a hyperparameter.
        size = mem_factor * n_paths
        logger.info(f"""Memory size set to {size}.""")
        self.n_paths = n_paths
        self.max_size = size
        # The memory counter is used to track how much of the replay
        # buffer was filled. It is later used for indexing when storing
        # new memories/ experiences
        self.mem_cntr = 0
        logger.info(f"""Memory counter set to {self.mem_cntr}.""")

        # Preallocate the buffers for all relevant variables
        # State memory contains the current observed state
        self.state_memory = np.zeros(
            core.combined_shape(size, state_dims), dtype=np.float32
        )
        # Next state memory contains the state that resulted from the
        # action chosen in the prior state
        self.next_state_memory = np.zeros(
            core.combined_shape(size, state_dims), dtype=np.float32
        )
        # Action contains the chosen actions at given states (state_memory)
        self.action_memory = np.zeros(
            core.combined_shape(size, action_dims), dtype=np.float32
        )
        # Contains the reward which the player/ agent gets after
        # performing an action from a given state
        self.reward_memory = np.zeros(size, dtype=np.float32)
        # Terminal memory contains 0s or 1s, depending if the
        # corresponding state (action, reward, next state pair) result
        # in the termination of the game/ an end game condition was met.n
        self.terminal_memory = np.zeros(size, dtype=np.float32)

        logger.info("""Initializing replay memory finished...""")

    def store_transition(
        self,
        states: ndarray,
        actions: ndarray,
        rewards: ndarray,
        next_states: ndarray,
        dones: ndarray,
    ):
        """
        Stores a given transition or a set of transitions in their 
        respective buffer array.

        Whether a set of transitions or a single transition is saved in
        the array depends on the n_paths attribute. If n_paths = 1, then
        a single transition is saved in the array. Else each entry of
        the n_paths is saved in a respective row of their arrays. 

        Args:
            states (ndarray): Array of vectors holding information of
                all dimensions of a state.
            actions (ndarray): Array of action vectors which are used to
                step from state to next_state.
            rewards (ndarray): Array of vectors of scalars which describe the
                rewards gained from doing an action at a given state.
            next_states (ndarray): Array of vectors of states that result from
                applying chosen actions at a given prior states. 
            dones (ndarray): Vector of 0s and 1s that serve as flags
                indicating when the end of the game is reached.
        """
        # Check up to which point the memory/ the replay buffer was
        # filled. E.g. mem_cntr = 10; max_size (size of buffer) = 100
        # --> index = 10. Therefore the next time something is stored in
        # the replay buffer, it should be placed at index 10
        start_idx = self.mem_cntr % self.max_size
        end_idx = start_idx + self.n_paths
        logger.info(
            f"""Writing into replay memory from start index: {start_idx} to end index: {end_idx}."""
        )

        # Fill the respective replay memory from the start to end index
        self.state_memory[start_idx:end_idx] = states
        self.action_memory[start_idx:end_idx] = actions
        self.reward_memory[start_idx:end_idx] = rewards
        self.next_state_memory[start_idx:end_idx] = next_states
        self.terminal_memory[start_idx:end_idx] = dones

        # The memory counter is updated by adding the number of paths.
        # Usually only a single experience would be stored in replay
        # memory. However, when multiple games/ paths are played at the
        # same time, the replay memory gets more than a single
        # experience, namely as many experiences as there are paths
        self.mem_cntr += self.n_paths
        logger.info(f"""Memory counter set to {self.mem_cntr}.""")

    def sample_batch(self, batch_size: int):# -> dict:
        """
        Randomly sampled subset of the ReplayBuffer used for one step of
        gradient ascent/ descent of the learning process, since using 
        the totality of the ReplayBuffer proved to be inefficient.

        Args:
            batch_size (int): Defines the amount of tuples sampled from 
            the ReplayBuffer for the learning process.

        Returns:
            dict: Dictionary containing pytorch tensors.
                The dictionary contains tensors for states, next states,
                actions, rewards, dones (termination flags).
        """
        logger.info(f"Sampling batch with size {batch_size} from replay memory.")
        # max_mem defines the upper limit of the sampling interval. It
        # prevents the sampling of null values, when the replay_buffer
        # is not completely filled.
        max_mem = min(self.mem_cntr, self.max_size)
        logger.info(f"""Max memory size is {max_mem}.""")
        # random.choice samples multiple integer values within a given
        # range. For our purposes it is equivalent to randint.
        idxs = np.random.choice(max_mem, batch_size)

        # create batch dictionary which is going to be cast into tensors
        batch = dict(
            states=self.state_memory[idxs],
            next_states=self.next_state_memory[idxs],
            actions=self.action_memory[idxs],
            rewards=self.reward_memory[idxs],
            dones=self.terminal_memory[idxs],
        )

        # The sample_batch method is called when learning. Because the
        # values that are taken from sample buffer are directly fed into
        # the respective network, they need to be transformed into a
        # torch tensor
        logger.info(f"""Finished sampling batch from replay memory.""")
        return {k: T.as_tensor(v, dtype=T.float32) for k, v in batch.items()}


# %%
if __name__ == "__main__":
    STATE_DIMS = 2
    ACTION_DIMS = 1
    MEM_FACTOR = 100
    N_PATHS = 1

    replay_buffer = ReplayBuffer(STATE_DIMS, ACTION_DIMS, MEM_FACTOR, N_PATHS)
    # Generate arbitrary state transition
    states = np.arange(N_PATHS * STATE_DIMS).reshape((N_PATHS, STATE_DIMS))
    next_states = np.arange(N_PATHS * STATE_DIMS).reshape((N_PATHS, STATE_DIMS))
    actions = np.arange(N_PATHS * ACTION_DIMS).reshape((N_PATHS, ACTION_DIMS))
    rewards = np.arange(N_PATHS)
    dones = np.zeros(N_PATHS)

    replay_buffer.store_transition(states, actions, rewards, next_states, dones)
    replay_buffer.sample_batch(30)
# %%
