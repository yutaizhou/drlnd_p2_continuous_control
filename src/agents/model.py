import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256,256], seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes = [256, 256], seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_uniform_(self.fc1.weight.data) 
        nn.init.xavier_uniform_(self.fc2.weight.data) 
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
    
    def forward(self, state, action):
        """Q-Value critic"""
        state_input = F.relu(self.fc1(state))
        x = torch.cat((state_input, action), dim=-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return 
