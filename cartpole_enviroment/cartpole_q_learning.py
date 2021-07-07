import os
import random
from collections import deque

import gym
import numpy as np
import torch
from torch import nn, optim

GAMMA = 0.99
MEMORY_SIZE = 1000000
BATCH_SIZE = 40
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.995

env = gym.make("CartPole-v1")
random.seed(42)
env.seed(42)
torch.manual_seed(42)
directory = './cartpole_ql'
load_filename = 'checkpoint_107.pth'


class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = torch.nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-4)
        self.loss = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model(torch.from_numpy(state).type(torch.FloatTensor))
        return np.argmax(q_values.detach().numpy()[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.max(self.model(torch.from_numpy(state_next).type(torch.FloatTensor)).detach().numpy()[0]))
            q_values = self.model(torch.from_numpy(state).type(torch.FloatTensor))
            q_values_target = torch.clone(q_values)
            q_values_target[0][action] = q_update
            loss = self.loss(q_values, q_values_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.exploration_rate *= EXPLORATION_DECAY
        if self.exploration_rate < 1e-3:
            self.exploration_rate = 0.0


observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
dqn_solver = DQNSolver(observation_space, action_space)
episode = 0
if load_filename is not None:
    dqn_solver.model.load_state_dict(torch.load(os.path.join(directory, load_filename)))
    episode = int(load_filename[11:-4])
    print('Resuming training from checkpoint \'{}\'.'.format(load_filename))
while True:
    episode += 1
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    episode_reward = 0
    while True:
        env.render()
        action = dqn_solver.act(state)
        state_next, reward, terminal, info = env.step(action)
        episode_reward += reward
        state_next = np.reshape(state_next, [1, observation_space])
        dqn_solver.remember(state, action, reward, state_next, terminal)
        state = state_next
        if terminal:
            print("Run: " + str(episode) + ", exploration: " + str(dqn_solver.exploration_rate)
                  + ", total reward: " + str(episode_reward), ", last reward: " + str(reward))
            filename = os.path.join(directory, 'checkpoint_{}.pth'.format(episode))
            torch.save(dqn_solver.model.state_dict(), f=filename)
            break
        dqn_solver.experience_replay()