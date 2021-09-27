from play_market import play_strategy
from network_utils import *
import merton_environment as env
from neural_network_dqn import DQN
from buffer import ReplayBuffer
from dqn_agent import DQNAgent

import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import math
from datetime import datetime

from IPython.display import clear_output
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # This needs to be fixed to run on GPU
    # USE_CUDA = torch.cuda.is_available()
    USE_CUDA = False
    Variable = lambda *args, **kwargs: autograd.Variable(
        *args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

    MODEL_NAME = "test1"
    SAVE_DIR = "./models"

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
        render=False,
    )
    market_test = env.MertonEnvironment(
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
        render=False,
    )

    state_dims = market_train.observation_space.shape[0]
    action_dims = market_train.action_space.shape[0]

    # Instatiate the DQN model
    model = DQN(
        MODEL_NAME,
        state_dims,
        action_dims,
        market_train)

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

    agent.train()

    # #! -------- Test --------
    # if train == False:
    #     load_model(model, path_to_load)

    # utilities = []
    # rewards = []
    # step_rewards = []
    # wealth_epochs = []
    # rewards_sum = 0

    # #? epochs + 1 and then in for loop range(epochs - 1) ???
    # for epoch in range(test_epochs):

    #     if epoch % 10000 == 0:
    #         print(epoch)

    #     state = market_test.reset()
    #     # I think that no argmax().item() is needed since we sample continous
    #     # actions. item() is required to take the value out of the torch tensor
    #     action = model(Variable(torch.FloatTensor(np.float32(state)))).item()#.argmax().item()
    #     # print(action)

    #     while True:
    #         state, reward, done, _  = market_test.step(action)
    #         action = model(Variable(torch.FloatTensor(np.float32(state)))).item()#.argmax().item()
    #         rewards_sum += reward

    #         step_rewards.append(reward)

    #         if done:
    #             wealth_epochs.append(state[0])
    #             utilities.append(np.log(state[0]))
    #             rewards.append(rewards_sum)
    #             break

    # utilities_test_rand, rewards_test_rand, step_rew_rand, wealth_test_rand = play_strategy(
    #                                                                                         wealth_0,
    #                                                                                         rf,
    #                                                                                         mu,
    #                                                                                         sigma,
    #                                                                                         kappa,
    #                                                                                         "Random",
    #                                                                                         n_paths = n_paths,
    #                                                                                         T = T,
    #                                                                                         n_discr = n_discr,
    #                                                                                         epochs=100000,
    #                                                                                         )

    # utilities_test_rand, rewards_test_rand, step_rew_rand, wealth_test_rand = play_strategy(
    # wealth_0,
    # rf,
    # mu,
    # sigma,
    # kappa,
    # "Merton",
    # n_paths = n_paths,
    # T = T,
    # n_discr = n_discr,
    # epochs=100000,
    # )
