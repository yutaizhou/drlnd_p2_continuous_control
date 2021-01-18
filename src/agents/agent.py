import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import Actor, Critic
from .replay import ReplayBuffer
from .utils import OUNoise
from ..utils.utils import DEVICE



BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay


class DDPG():
    def __init__(self, state_size, action_size, random_seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor w/ target
        self.actor_local = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
    
        # Critic w/ target 
        self.critic_local = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
    
        # Misc
        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(DEVICE)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, +1)

    def reset(self):
        self.noise.reset()
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # update critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + gamma * Q_targets_next * (1-dones)

        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update actor
        actions_pred = self.actor_local(states)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


        # target network upates
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            mixed_param = tau * local_param.data + (1-tau)*target_param.data
            target_param.data.copy_(mixed_param)



