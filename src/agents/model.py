import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.uniform_(self.fc2.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes = [256, 256, 128], seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], 1)
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_uniform_(self.fc1.weight.data) 
        nn.init.xavier_uniform_(self.fc2.weight.data) 
        nn.init.xavier_uniform_(self.fc3.weight.data) 
        nn.init.uniform_(self.fc4.weight.data, -3e-3, 3e-3)
    
    def forward(self, state, action):
        """Q-Value critic"""
        state_input = F.leaky_relu(self.fc1(state))
        x = torch.cat((state_input, action), dim=-1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
