import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
torch.random.manual_seed(42)
np.random.seed(42)
env = gym.make('Pong-v0')
env.seed(42)


class PongSolver:
    def __init__(self, hidden_size, input_size, learning_rate, eps):
        self.model = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, 2, bias=False),
            nn.Softmax(dim=-1)
        ).cuda()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, eps=eps)
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
            if r != 0:
                future_reward = 0
            future_reward = r + gamma * future_reward
            rewards.append(future_reward)
        rewards = torch.tensor(rewards[::-1], dtype=torch.float32).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()


def preprocess(frame):
    frame = frame[35:195]
    frame = frame[::2, ::2, 0]
    frame[frame == 144] = 0
    frame[frame == 109] = 0
    frame[frame != 0] = 1
    return frame.astype(float).ravel()


batch_size = 10
gamma = 0.99
load_filename = "checkpoint_3000.pth"
save_directory = "./pong_pg"
batch_rewards = []
episode = 0

model = PongSolver(hidden_size=1024, input_size=6400, learning_rate=0.001, eps=1e-5)
if load_filename is not None:
    episode = model.load_checkpoint(save_directory, load_filename)
while True:
    observation = env.reset()
    prev_frame = np.zeros(6400)
    done = False
    while not done:
        env.render()
        frame = preprocess(observation)
        x = frame - prev_frame
        prev_frame = frame
        action_probs = model.forward(torch.tensor(x, dtype=torch.float32).cuda())
        distribution = Categorical(action_probs)
        action = distribution.sample()
        observation, reward, done, _ = env.step(2 if action.item() == 1 else 3)
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

