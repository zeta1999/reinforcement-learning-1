import copy
import os
import random
from collections import deque
import gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

env = gym.make("merge-v0")
config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (128, 64),
           "stack_size": 4,
           "weights": [0.2989, 0.5870, 0.1140],
           "scaling": 2.0,
       },
       "real_time_rendering": True,
       "simulation_frequency": 30
   }
env.configure(config)
env.seed(42)
env.action_space.seed(42)
torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)


class DDQNSolver(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class DDQNAgent:
    def __init__(self, action_dim, save_directory):
        self.action_dim = action_dim
        self.save_directory = save_directory
        self.net = DDQNSolver(self.action_dim).cuda()
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.9995
        self.exploration_rate_min = 0.05
        self.current_step = 0
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.sync_period = 500
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-4, eps=1e-4)
        self.loss = torch.nn.SmoothL1Loss()
        self.episode_rewards = []
        self.moving_average_episode_rewards = []
        self.current_episode_reward = 0.0

    def log_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0

    def log_period(self, episode, epsilon, step):
        self.moving_average_episode_rewards.append(np.round(np.mean(self.episode_rewards[-checkpoint_period:]), 3))
        print(f"Episode {episode} - Step {step} - Epsilon {epsilon} - Mean Reward {self.moving_average_episode_rewards[-1]}")
        plt.plot(self.moving_average_episode_rewards)
        plt.savefig(os.path.join(self.save_directory, f"episode_rewards_plot_{episode}.png"))
        plt.clf()

    def remember(self, state, next_state, action, reward, done):
        self.memory.append((torch.tensor(state.__array__(), dtype=torch.float32), torch.tensor(next_state.__array__(), dtype=torch.float32),
                            torch.tensor([action]), torch.tensor([reward]), torch.tensor([done])))

    def experience_replay(self, step_reward):
        self.current_episode_reward += step_reward
        if self.current_step % self.sync_period == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())
        if self.batch_size > len(self.memory):
            return
        state, next_state, action, reward, done = self.recall()
        q_estimate = self.net(state.cuda(), model="online")[np.arange(0, self.batch_size), action.cuda()]
        with torch.no_grad():
            best_action = torch.argmax(self.net(next_state.cuda(), model="online"), dim=1)
            next_q = self.net(next_state.cuda(), model="target")[np.arange(0, self.batch_size), best_action]
            q_target = (reward.cuda() + (1 - done.cuda().float()) * self.gamma * next_q).float()
        loss = self.loss(q_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def recall(self):
        state, next_state, action, reward, done = map(torch.stack, zip(*random.sample(self.memory, self.batch_size)))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action_values = self.net(torch.tensor(state.__array__(), dtype=torch.float32).cuda().unsqueeze(0), model="online")
            action = torch.argmax(action_values, dim=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1
        return action

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']

    def save_checkpoint(self):
        filename = os.path.join(self.save_directory, 'checkpoint_{}.pth'.format(episode))
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))


checkpoint_period = 10
save_directory = "merge_ql"
load_checkpoint = "checkpoint_340.pth"
agent = DDQNAgent(action_dim=env.action_space.n, save_directory=save_directory)
if load_checkpoint is not None:
    agent.load_checkpoint(save_directory + "/" + load_checkpoint)
episode = 0
while True:
    state = env.reset()
    while True:
        action = agent.act(state)
        env.render()
        next_state, reward, done, info = env.step(action)
        agent.remember(state, next_state, action, reward, done)
        agent.experience_replay(reward)
        state = next_state
        if done:
            episode += 1
            agent.log_episode()
            if episode % checkpoint_period == 0:
                agent.log_period(episode=episode, epsilon=agent.exploration_rate, step=agent.current_step)
                agent.save_checkpoint()
            break
