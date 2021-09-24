import numpy as np
from merton_environment import MertonEnvironment
import torch

def play_strategy(
        wealth_0: float,
        rf: float,
        mu,
        sigma: float,
        kappa: float,
        policy: str, 
        stock_price: float = 1.0,
        bond_price: float = 1.0,
        n_paths: int = 1,
        T: int = 1,
        n_discr: int = 1,
        seed: int = None,
        q_values=None, 
        epochs=300000, 
        ):
    
    wealth = wealth_0

    utilities_test = []
    rewards_test = []
    step_rewards = []
    rsum = 0
    wealth_epochs = []

    # #this needs tidying - specific for problem above
    # start_state = int(wealth/10)
    # state = start_state 
    
    env = MertonEnvironment(        
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
        render=False,)


    for epochs in range(epochs):
        state = env.reset()

        while True:
            
            if policy=="Agent":
                #? not sure if this is correct for the agent. I think it
                #? is more sensible to just use the trained network here
                # something like this??
                # action = model(Variable(torch.FloatTensor(np.float32(state)))).item()
                action = np.argmax(q_values[state])

            elif policy=="Random":
                # action = int(torch.LongTensor(1).random_(0, number_of_actions))
                action = env.action_space.sample()
                
            elif policy=="Merton":
                # action = best_action
                action = env.merton_ratio()

            state, reward, done, _ = env.step(action)

            d_wealth = env.new_wealth - env.wealth
            # reward, d, new_state, dx, done = env.step(action)

            wealth += d_wealth
            # new_state = int(wealth/10)

            rsum += reward
            
            step_rewards.append(reward)

            if done:
                # should be covered by reset # state = start_state
                utilities_test.append(np.log(wealth))
                rewards_test.append(rsum)
                wealth_epochs.append(wealth)
                rsum = 0
                wealth = wealth_0
                break
                
    return utilities_test, rewards_test, step_rewards, wealth_epochs
