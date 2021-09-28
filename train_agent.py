from network_utils import *
import merton_environment as env
from neural_network_dqn import DQN
from buffer import ReplayBuffer
from dqn_agent import DQNAgent
from plot_agent_performance import *

import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from datetime import datetime

from IPython.display import clear_output
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # This needs to be fixed to run on GPU
    # USE_CUDA = torch.cuda.is_available()
    USE_CUDA = False
    Variable = lambda *args, **kwargs: autograd.Variable(
        *args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

    MODEL_NAME = "DQN Network"
    SAVE_DIR = "./models"
    TRAIN = True
    LOAD = False

    name_prefix = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss") + "_"
    # name_prefix = "2021_01_27_17h05m27s_"
    dir_name = name_prefix + MODEL_NAME
    save_dir = os.path.join(SAVE_DIR, dir_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    batch_size = 32
    gamma = 0.99

    buffer_size = 1000
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    # training_epochs = 1000000
    training_epochs = 1000
    # test_epochs = 100000
    test_epochs = 1000

    rf = 0.02
    mu = 0.1
    sigma = 0.2
    kappa = 0.008
    stock_price = 1.0
    bond_price = 1.0
    wealth_0 = 100.0
    T = 1  # Normed to one period. I.e. 10 years = 1 period
    n_paths = 1
    n_discr = 20
    n_action_discr = 15

    seed = 0

    # Initialize two environments. One for training and one for testing
    market_train = env.MertonEnvironment(
        wealth_0,
        rf,
        mu,
        sigma,
        kappa,
        stock_price=stock_price,
        bond_price=bond_price,
        n_paths=n_paths,
        T=T,
        n_discr=n_discr,
        n_action_discr=n_action_discr,
        seed=seed,
        # render=False,
        render=False,
    )
 
    state_dims = market_train.observation_space.shape[0]
    action_dims = market_train.action_space.shape[0]

    # Instatiate the DQN model
    model = DQN(
        MODEL_NAME,
        state_dims,
        action_dims,
        market_train,
        checkpoint_dir=save_dir)

    if USE_CUDA:
        model = model.cuda()

    # Initialize optimizer and replay buffer
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(
        state_dims, action_dims, buffer_size, n_paths=n_paths)

    agent = DQNAgent(
        MODEL_NAME,
        market_train,
        model,
        optimizer,
        training_epochs,
        buffer_size,
        batch_size=32,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_final=epsilon_final,
        epsilon_decay=epsilon_decay,
    )

    if TRAIN:
        agent.train()
    elif LOAD:
        # The correct model name has to be initialized in DQN for this
        agent.load() 

    last_episode_utilities_test, \
    last_episode_cum_rewards_test, \
    last_episode_wealths_test, \
    episode_rewards_test, \
    episode_wealth_test  = agent.play_market(agent="Test")

    last_episode_utilities_merton, \
    last_episode_cum_rewards_merton, \
    last_episode_wealths_merton, \
    episode_rewards_merton, \
    episode_wealth_merton  = agent.play_market(agent="Merton")

    last_episode_utilities_random, \
    last_episode_cum_rewards_random, \
    last_episode_wealths_random, \
    episode_rewards_random, \
    episode_wealth_random  = agent.play_market(agent="Random")

    make_agent_graphs(
        last_episode_cum_rewards_merton, last_episode_cum_rewards_random, last_episode_cum_rewards_test,
        last_episode_utilities_merton, last_episode_utilities_random,last_episode_utilities_test,
        last_episode_wealths_merton, last_episode_wealths_random, last_episode_wealths_test
    )