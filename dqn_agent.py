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


class TD3Agent:
    def __init__(
        self,
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
        save_freq: int = 1
    ):

        logger.info("Initializing DQN-Agent...")

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
        self.save_freq = save_freq

        self.memory = ReplayBuffer(
            self.state_dims, self.action_dims, self.mem_factor, self.n_paths
        )

    def epsilon_by_epoch(self, epoch):
        epsilon = self.epsilon_final + \
            (self.epsilon_start - self.epsilon_final) * \
            math.exp(-1. * epoch / self.epsilon_decay)
        return epsilon

    def train(self):

        episode_losses = []
        epoch_rewards = []

        for epoch in range(self.epochs):
            cumulative_rewards = 0

            states = self.env.reset()
            dones = False

            while not dones:
                epsilon = self.epsilon_by_epoch()

                actions = self.dqn(states)
                next_states, rewards, dones, info = self.env.step(actions)

                self.memory.store_transition(
                    states, actions, rewards, next_states, dones)

                # check if the replay memory is filled enough to sample from
                # if the required batch_size is larger than the total amount of
                # experiences([states,actions,next_states,rewards,dones]) in the replay
                # buffer, the function learn() is stopped at this point
                if self.memory.mem_cntr > self.batch_size:

                    # Get random batch from replay memory
                    memory_sample = self.memory.sample_batch(self.batch_size)

                    states = memory_sample["states"].to(self.dqn.device)
                    actions = memory_sample["actions"].to(self.dqn.device)
                    rewards = memory_sample["rewards"].to(self.dqn.device)
                    next_states = memory_sample["next_states"].to(self.dqn.device)
                    dones = memory_sample["dones"].to(self.dqn.device)

                    #! most likely there is work to be done at this point
                    #! depending if it is a discrete model or continous etc.
                    q_values = self.dqn(states)
                    next_q_values = self.dqn(next_states)

                    #! Does 1-dones work?
                    expected_q_values = rewards + self.gamma*next_q_values*(1-dones)
                    #! is this correct?
                    loss = F.mse_loss(expected_q_values, q_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    #! does this work?
                    episode_losses.append(loss)

                #! does this work?
                cumulative_rewards += rewards
                epoch_rewards.append(cumulative_rewards)
