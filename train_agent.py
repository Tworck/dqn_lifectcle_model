from play_market import play_strategy
from network_utils import *
import merton_environment as env
from neural_network_dqn import DQN
from replay_buffer import ReplayBuffer
import torch
import torch.autograd as autograd 
import torch.optim as optim
import numpy as np
import math
import time

from IPython.display import clear_output
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # This needs to be fixed to run on GPU
    # USE_CUDA = torch.cuda.is_available()
    USE_CUDA = False
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

    MODEL_NAME = str(
        f"model-{int(time.time())}")

    train = True
    path_to_load = r"/models/"

    batch_size = 32
    gamma      = 0.99

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
    T = 1 # Normed to one period. I.e. 10 years = 1 period
    n_paths = 1
    n_discr = 20

    seed=0

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
        seed=seed,
        render=False,
    )

    # Instatiate the DQN model
    #? Is this okay to abstract a deterministic model to a continuous just
    #? like this? probably not
    model = DQN(market_train.observation_space.shape[0],
                market_train.action_space.shape[0], 
                market_train)

    if USE_CUDA:
        model = model.cuda()

    # Calculate the epsilon for each epoch with a decay such that the 
    # exploration via greedy is not done until the very end of training
    epsilon_by_epoch = lambda epoch: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. 
                                                                * epoch / epsilon_decay)

    # Initialize optimizer and replay buffer
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(buffer_size)
    
    #! -------- Train --------

    losses = []
    all_rewards = []
    episode_reward = 0

    if train == True:
        state = market_train.reset()
        
        for epoch in range(training_epochs):
            epsilon = epsilon_by_epoch(epoch)

            action = model.act(state, epsilon)

            next_state, reward, done, _ = market_train.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += 1

            if done:
                state = market_train.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            # once the buffer is filled with values the algorithm should 
            # start to learn
            if len(replay_buffer) > batch_size:
                state, action, reward, next_state, done = replay_buffer.sample(batch_size)
                # state, next_state, action, reward, done = replay_buffer.sample(batch_size)

                # nicer if transformation is done in replay buffer imo
                state = tuple([i.tolist()[0] for i in state]) # dont like this solution
                # print(f"{state=}")
                state      = Variable(torch.FloatTensor(np.float32(state)))
                next_state = tuple([i.tolist()[0] for i in next_state]) # dont like this solution
                next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
                action = tuple([i.item() for i in action]) # dont like this solution
                print(f"{action=}")
                action     = Variable(torch.LongTensor(action))
                reward = tuple([i.item() for i in reward]) # dont like this solution
                reward     = Variable(torch.FloatTensor(reward))
                done = tuple([i.item() for i in done]) # dont like this solution
                done       = Variable(torch.FloatTensor(done))

                q_values = model(state)
                print(f"{q_values.shape=}")
                next_q_values = model(next_state)

                #? does this work for continuous values of actions?
                print(f"{action=}")
                print(f"{action.shape=}")
                q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
                next_q_value = next_q_values.max(1)[0]
                expected_q_value = reward + gamma * next_q_value * (1 - done)
                
                loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
                losses.append(loss.data)    

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if epoch % 400 == 0:
            #     clear_output(True)
            #     plt.figure(figsize=(20,5))
            #     plt.subplot(131)
            #     plt.title('frame %s. reward: %s' % (epoch, np.mean(all_rewards[-10:])))
            #     plt.plot(all_rewards)
            #     plt.subplot(132)
            #     plt.title('loss')
            #     print(f"{losses=}")
            #     plt.plot(np.convolve(losses,np.ones((1000,))/1000,mode='valid'))
            #     plt.show()

        save_model(MODEL_NAME, model, optimizer)

    #! -------- Test --------
    if train == False:
        load_model(model, path_to_load)


    utilities = []
    rewards = []
    step_rewards = []
    wealth_epochs = []
    rewards_sum = 0

    #? epochs + 1 and then in for loop range(epochs - 1) ???
    for epoch in range(test_epochs):

        if epoch % 10000 == 0:
            print(epoch)
        
        state = market_test.reset()
        # I think that no argmax().item() is needed since we sample continous
        # actions. item() is required to take the value out of the torch tensor
        action = model(Variable(torch.FloatTensor(np.float32(state)))).item()#.argmax().item()
        # print(action)

        while True:
            state, reward, done, _  = market_test.step(action)
            action = model(Variable(torch.FloatTensor(np.float32(state)))).item()#.argmax().item()
            rewards_sum += reward
            
            step_rewards.append(reward)

            if done:
                wealth_epochs.append(state[0])
                utilities.append(np.log(state[0]))
                rewards.append(rewards_sum)
                break


    utilities_test_rand, rewards_test_rand, step_rew_rand, wealth_test_rand = play_strategy(
                                                                                            wealth_0,
                                                                                            rf,
                                                                                            mu,
                                                                                            sigma,
                                                                                            kappa,
                                                                                            "Random",
                                                                                            n_paths = n_paths,
                                                                                            T = T,
                                                                                            n_discr = n_discr,
                                                                                            epochs=100000, 
                                                                                            )

    utilities_test_rand, rewards_test_rand, step_rew_rand, wealth_test_rand = play_strategy(
                                                                                            wealth_0,
                                                                                            rf,
                                                                                            mu,
                                                                                            sigma,
                                                                                            kappa,
                                                                                            "Merton",
                                                                                            n_paths = n_paths,
                                                                                            T = T,
                                                                                            n_discr = n_discr,
                                                                                            epochs=100000, 
                                                                                            )