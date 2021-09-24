import torch.nn as nn
import math, random
import torch
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, env):
        super(DQN, self).__init__()
        
        self.env = env
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.shape[0])
        )

        # Initialize device to which the network should be passed to
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Pass the Critic network to said device
        self.to(self.device)
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            # action = random.randrange(env.action_space.n)
            action = self.env.action_space.sample()
        return action