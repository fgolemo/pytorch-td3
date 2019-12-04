import copy
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class MlpActor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(MlpActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
    x, 0), nn.init.calculate_gain('relu'))


class CnnActor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(CnnActor, self).__init__()

        self.img_embed = nn.Sequential(
            init_cnn(nn.Conv2d(3, 96, 3, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(96, 96, 5, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(96, 32, 5, stride=2)), nn.ReLU(), Flatten(),
            init_cnn(nn.Linear(32 * 8 * 8, 256)), nn.ReLU())

        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = state.view(-1, 84, 84, 3)
        a = a.permute(0,3,1,2)

        a = self.img_embed(a)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class MlpCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(MlpCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class CnnCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(CnnCritic, self).__init__()

        # Q1 architecture
        self.img_embed_q1 = nn.Sequential(
            init_cnn(nn.Conv2d(3, 96, 3, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(96, 96, 5, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(96, 32, 5, stride=2)), nn.ReLU(), Flatten(),
            init_cnn(nn.Linear(32 * 8 * 8, 128)), nn.ReLU())
        self.l1 = nn.Linear(128 + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.img_embed_q2 = nn.Sequential(
            init_cnn(nn.Conv2d(3, 96, 3, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(96, 96, 5, stride=2)), nn.ReLU(),
            init_cnn(nn.Conv2d(96, 32, 5, stride=2)), nn.ReLU(), Flatten(),
            init_cnn(nn.Linear(32 * 8 * 8, 128)), nn.ReLU())
        self.l4 = nn.Linear(128 + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = self.img_embed_q1(state.view(-1, 3, 84, 84))
        sa1 = torch.cat([x, action], 1)

        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        x = self.img_embed_q1(state.view(-1, 3, 84, 84))
        sa2 = torch.cat([x, action], 1)

        q2 = F.relu(self.l4(sa2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        x = self.img_embed_q1(state.view(-1, 3, 84, 84))
        sa = torch.cat([x, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):

    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 policy="Mlp"):

        assert policy in ["Mlp", "Cnn"]

        actor_class = str_to_class(f"{policy}Actor")
        critic_class = str_to_class(f"{policy}Critic")

        self.actor = actor_class(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)

        self.critic = critic_class(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state, device_override=None):
        state = torch.FloatTensor(state.reshape(1, -1))
        if device_override is not None:
            state = state.to(torch.device(device_override))
        else:
            state = state.to(device)

        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)
