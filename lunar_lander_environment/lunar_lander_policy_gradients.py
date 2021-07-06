import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
torch.random.manual_seed(42)
np.random.seed(42)
env = gym.make('LunarLander-v2')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
env.seed(42)


class LunarLanderSolver:
    def __init__(self, hidden_size, input_size, output_size, learning_rate, eps):
        self.model = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=False),
            nn.Softmax(dim=-1)
        ).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=eps)
        self.reset()

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.episode_actions = torch.tensor([], requires_grad=True).cuda()
        self.episode_rewards = []

    def save_checkpoint(self, directory, episode):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, 'checkpoint_{}.pth'.format(episode))
        torch.save(self.model.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def load_checkpoint(self, directory, filename):
        self.model.load_state_dict(torch.load(os.path.join(directory, filename)))
        print('Resuming training from checkpoint \'{}\'.'.format(filename))
        return int(filename[11:-4])

    def backward(self):
        future_reward = 0
        rewards = []
        for r in self.episode_rewards[::-1]:
            future_reward = r + gamma * future_reward
            rewards.append(future_reward)
        rewards = torch.tensor(rewards[::-1], dtype=torch.float32).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()


batch_size = 10
gamma = 0.99
load_filename = 'checkpoint_4500.pth'
save_directory = "./lunar_lander_pg"
batch_rewards = []
episode = 0

model = LunarLanderSolver(hidden_size=1024, input_size=observation_space, output_size=action_space, learning_rate=1e-4, eps=1e-4)
if load_filename is not None:
    episode = model.load_checkpoint(save_directory, load_filename)
while True:
    observation = env.reset()
    done = False
    while not done:
        env.render()
        frame = np.reshape(observation, [1, observation_space])
        action_probs = model.forward(torch.tensor(observation, dtype=torch.float32).cuda())
        distribution = Categorical(action_probs)
        action = distribution.sample()
        observation, reward, done, _ = env.step(action.item())
        model.episode_actions = torch.cat([model.episode_actions, distribution.log_prob(action).reshape(1)])
        model.episode_rewards.append(reward)
        if done:
            batch_rewards.append(np.sum(model.episode_rewards))
            model.backward()
            episode += 1
            if episode % batch_size == 0:
                print('Batch: {}, average reward: {}'.format(episode // batch_size, np.array(batch_rewards).mean()))
                batch_rewards = []
            if episode % 50 == 0 and save_directory is not None:
                model.save_checkpoint(save_directory, episode)
