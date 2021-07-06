import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

load_filename = 'checkpoint_420.pth'
directory = "./cartpole_pg"
env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)
gamma = 0.99


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.model = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1)
        )
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def reset(self):
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        return self.model(x)


def predict(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action_probs = policy(state)
    distribution = Categorical(action_probs)
    action = distribution.sample()
    policy.episode_actions = torch.cat([
        policy.episode_actions,
        distribution.log_prob(action).reshape(1)
    ])
    return action


def update_policy():
    R = 0
    rewards = []
    for r in policy.episode_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    loss = torch.sum(torch.mul(policy.episode_actions, rewards).mul(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.episode_rewards))
    policy.reset()


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=5e-4, eps=1e-4)
episode = 0
if load_filename is not None:
    policy.model.load_state_dict(torch.load(os.path.join(directory, load_filename)))
    episode = int(load_filename[11:-4])
    print('Resuming training from checkpoint \'{}\'.'.format(load_filename))
while True:
    episode += 1
    state = env.reset()
    while True:
        action = predict(state)
        env.render()
        state, reward, done, _ = env.step(action.item())
        policy.episode_rewards.append(reward)
        if done:
            if episode % 10 == 0:
                print('Episode {} total reward: {:.2f}'.format(episode, np.sum(policy.episode_rewards)))
                filename = os.path.join(directory, 'checkpoint_{}.pth'.format(episode))
                torch.save(policy.model.state_dict(), f=filename)
            update_policy()
            break
