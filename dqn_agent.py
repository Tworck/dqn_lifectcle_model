import os
from sys import breakpointhook
import torch
import torch.nn.functional as F

import numpy as np
from numpy import ndarray
from torch.optim import optimizer
from buffer import ReplayBuffer
import math

import logging

# Setup logger
logger = logging.getLogger(__name__)


class DQNAgent:
    def __init__(
        self,
        name,
        environment,
        dqn,
        optimizer,
        epochs,
        mem_factor: int,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500,
    ):

        logger.info("Initializing DQN-Agent...")
        self.name = name

        self.env = environment
        self.dqn = dqn
        self.optimizer = optimizer

        self.state_dims = self.env.observation_space.shape
        self.action_dims = self.env.action_space.shape[0]

        self.n_paths = self.env.n_paths if hasattr(self.env, "n_paths") else 1
        self.mem_factor = mem_factor

        self.epochs = epochs
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayBuffer(
            self.state_dims, self.action_dims, self.mem_factor, self.n_paths
        )

    def save_models(self):
        """
        Calls the save_checkpoint method for all of the TD3 agents'
        networks and saves them.
        """
        logger.info(f"Saving models to hard drive...")
        self.dqn.save_checkpoint()
        logger.info(f"Finished saving models to hard drive...")

    def load_models(self):
        """
        Calls the load_checkpoint method for all of the TD3 agents'
        networks and loads them.
        """
        logger.info(f"Loading models from hard drive...")
        self.dqn.load_checkpoint()
        logger.info(f"Finished loading models from hard drive...")

    def epsilon_by_epoch(self, epoch):
        epsilon = self.epsilon_final + \
            (self.epsilon_start - self.epsilon_final) * \
            math.exp(-1. * epoch / self.epsilon_decay)
        return epsilon

    def learn(self):
        # Get random batch from replay memory
        memory_sample = self.memory.sample_batch(self.batch_size)

        states = memory_sample["states"].to(self.dqn.device)
        actions = memory_sample["actions"].to(self.dqn.device)
        rewards = memory_sample["rewards"].to(self.dqn.device)
        next_states = memory_sample["next_states"].to(
            self.dqn.device)
        dones = memory_sample["dones"].to(self.dqn.device)

        #! most likely there is work to be done at this point
        #! depending if it is a discrete model or continous etc.
        q_values = self.dqn(states)
        next_q_values = self.dqn(next_states)

        # print(f"{q_values.shape=}")
        # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        #! why would we even do the sqeueeze unsqueeze stuff
        # q_values = q_values.gather(
        #     1, actions.type(torch.int64).unsqueeze(1)).squeeze(1)
        #! the q_values tensor is currently shape (batch_size, n_action_discr)
        #! for the calculation of mse loss we need to have a tensor
        #! with shape: batch_size. Since all values along the
        #! n_action_disr dimension are the same we can just take
        #! [:,0] (we could take any other value i.e. [:,4])
        #! THIS IS REALLY NOT A GOOD WAY TO DO IT
        q_values = q_values.gather(1, actions.type(torch.int64))[:, 0]
        next_q_values = next_q_values.max(1)[0]

        #! Does 1-dones work?
        expected_q_values = rewards + \
            self.gamma*next_q_values*(1-dones)
        #! is this correct?
        loss = F.mse_loss(expected_q_values, q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, render: bool = False):
        # The best reward is set to the lowest possible reward at the start
        # to have a benchmark for when to save the model
        best_reward = 0
        epoch_rewards_history = []
        epoch_losses = []

        for epoch in range(self.epochs):

            episode_len = 0
            cumulative_rewards = 0
            episode_reward_history = []

            episode_loss = []

            states = self.env.reset()
            dones = False

            while not dones:
                epsilon = self.epsilon_by_epoch(epoch)

                actions = self.dqn.act(states, epsilon)
                next_states, rewards, dones, info = self.env.step(actions)

                if render:
                    self.env.render()

                self.memory.store_transition(
                    states, actions, rewards, next_states, dones)

                # check if the replay memory is filled enough to sample from
                # if the required batch_size is smaller than the total amount of
                # experiences([states,actions,next_states,rewards,dones]) in the replay
                # buffer, the agent should learn (optimiye) at this point
                if self.memory.mem_cntr > self.batch_size:

                    loss = self.learn()
                    #! does this work?
                    episode_loss.append(loss)

                #! does this work? since we do not use npaths? or do we?
                cumulative_rewards += rewards
                # print(f"{cumulative_rewards=}")
                episode_len += 1

                #! do we want to run n paths?
                # we can run more than one path at a time. In this case
                # we need to reduce the dones array to a single value.
                if type(dones) == ndarray:
                    # .all() only evaluates to True if all elements are True
                    dones = dones.all()

                # average the cumulative rewards (1D array) of n_paths into one
                # value
                # referred as "rewards path average"
                rewards_path_avg = np.mean(cumulative_rewards)

                episode_reward_history.append(rewards_path_avg)

            # Stores the path averaged cumulative rewards of each episode
            epoch_rewards_history.append(rewards_path_avg)

            # average over the path averages of the last 100 episodes
            # referred as "rewards history average"
            rewards_history_avg = np.mean(epoch_rewards_history[-100:])
            epoch_losses.append(episode_loss)

            # Saves the most recent high score in terms of rewards_history_avg
            # In case of new high scores, the model parameters are saved
            if epoch % 1000 == 0:
                self.save_models()

            #! Commented out so that it does not save too much for now
            # if rewards_history_avg > best_reward:
            #     print("Saving model")
            #     best_reward = rewards_history_avg
            #     self.save_models()

            if epoch % 100 == 0:
                print(
                    "Epoch ",
                    epoch,
                    "Cumulative Score %.2f" % rewards_path_avg,
                    "Trailing 100 steps avg %.3f" % rewards_history_avg,
                )

    def play_market(self, agent="Random"):

        last_episode_wealths = []
        last_episode_utilities = []
        last_episode_cum_rewards = []

        for epoch in range(self.epochs):

            wealth = 0
            cumulative_rewards = 0
            episode_rewards = []
            episode_wealth = []

            states = self.env.reset()
            dones = False

            with torch.no_grad():
                while not dones:

                    if agent == "Test":
                        # Take some negative value for epsilon
                        actions = self.dqn.act(states, -1)

                    elif agent == "Random":
                        actions = self.env.action_space.sample()

                    elif agent == "Merton":
                        actions = self.env.merton_ratio()

                    states, rewards, dones, _ = self.env.step(actions)

                    wealth += self.env.new_wealth/self.env.wealth
                    cumulative_rewards += rewards

                    if type(dones) == ndarray:
                        # .all() only evaluates to True if all elements are True
                        dones = dones.all()

                    #! this will not work for n_paths > 1
                    episode_rewards.append(rewards)
                    episode_wealth.append(wealth)

            #! this will not work for n_paths > 1
            last_episode_cum_rewards.append(cumulative_rewards)
            last_episode_wealths.append(wealth)
            last_episode_utilities.append(np.log(wealth))

        return last_episode_utilities, last_episode_cum_rewards, \
            last_episode_wealths, episode_rewards, episode_wealth
