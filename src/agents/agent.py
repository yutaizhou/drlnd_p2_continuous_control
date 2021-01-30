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
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.95            # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0   # L2 weight decay
TRAIN_FREQ = 20 # update net work after this many time steps


class DDPG():
    def __init__(self, state_size, action_size, num_agents=1, random_seed=42):
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
        self.noise = OUNoise((num_agents, action_size), random_seed)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.num_batches = int(num_agents/2)
        self.t = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if (self.t % TRAIN_FREQ == 0) & (len(self.memory) > BATCH_SIZE * self.num_batches):
            experiences = self.memory.sample(self.num_batches)
            self._learn(experiences, GAMMA)
        self.t += 1

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
    
    @staticmethod
    def _batch_experiences(experiences):
        all_states, all_actions, all_rewards, all_next_states, all_dones = experiences
        b_states = torch.split(all_states, BATCH_SIZE)
        b_actions = torch.split(all_actions, BATCH_SIZE)
        b_rewards = torch.split(all_rewards, BATCH_SIZE)
        b_next_states = torch.split(all_next_states, BATCH_SIZE)
        b_dones = torch.split(all_dones, BATCH_SIZE)
        return b_states, b_actions, b_rewards, b_next_states, b_dones 
    
    def _learn(self, experiences, gamma):
        b_states, b_actions, b_rewards, b_next_states, b_dones = self._batch_experiences(experiences)

        for states, actions, rewards, next_states, dones in zip(b_states, b_actions, b_rewards, b_next_states, b_dones):
            # update critic
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next).detach()
            Q_targets = rewards + gamma * Q_targets_next * (1-dones)

            Q_expected = self.critic_local(states, actions)

            critic_loss = F.mse_loss(Q_expected, Q_targets)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_opt.step()

            # update actor
            actions_pred = self.actor_local(states)

            actor_loss = -self.critic_local(states, actions_pred).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # target network upates
            self._soft_update(self.actor_local, self.actor_target, TAU)
            self._soft_update(self.critic_local, self.critic_target, TAU)
    
    @staticmethod
    def _soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            mixed_param = tau * local_param.data + (1-tau)*target_param.data
            target_param.data.copy_(mixed_param)
    
    def save(self, fp: str):
        models = {
                'actor': self.actor_local.state_dict(),
                'critic': self.critic_local.state_dict()
        }
        torch.save(models, fp)

