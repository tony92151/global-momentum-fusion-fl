import copy
import glob
import os
import json
from datetime import datetime

import numpy as np
import random
import time
import sys
from collections import deque
import torch.nn as nn
import torch
from tqdm import tqdm
from copy import deepcopy
sys.path.append(".")
from dqn.model import DQN

class ReinforceAgent:
    def __init__(self, root, state_size, action_size, load_model=False, load_memory=False):
        self.root = root

        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        # train self.regular_model per self.target_update steps
        self.target_update = 100
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 0.7
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 10

        # if len of memory > self.train_start, star training
        self.train_start = 20
        self.memory = deque(maxlen=5000)
        self.model = self.buildModel()
        self.target_model = copy.deepcopy(self.model)
        # self.updateTargetModel()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.1)
        self.loss_function = nn.MSELoss()

        if load_model:
            self.load_model()

        if load_memory:
            self.load_memory()

    def load_model(self):
        print("Load model from : {}".format(os.path.join(self.root, "models", "models.pt")))
        assert os.path.isfile(os.path.join(self.root, "models", "models.pt"))
        self.model = self.buildModel()
        lmodel = torch.load(os.path.join(self.root, "models", "models.pt"))
        self.model = self.buildModel()
        self.target_model = copy.deepcopy(self.model)
        self.model.load_state_dict(lmodel["model"])
        self.target_model.load_state_dict(lmodel["model"])

    def load_memory(self):
        
        assert os.path.isdir(os.path.join(self.root, "checkpoint"))
        list_of_files = glob.glob(os.path.join(self.root, "checkpoint", "*"))
        latest_file = max(list_of_files, key=os.path.getctime)
        print("Load memory from : {}".format(latest_file))
        self.memory = torch.load(latest_file)

    def buildModel(self):
        return DQN(self.state_size, self.action_size)

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * next_target.max().item()

    def getAction(self, state, random_select=False):
        """
        This function use to determined next action.
        We setting self.epsilon to give an opportunity to try new action

        If self.epsilon = 0.2, mean 20% random action
        """
        if np.random.rand() <= self.epsilon or random_select:
            self.q_value = np.zeros(self.action_size)
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_value = self.model(torch.tensor(state, dtype=torch.float).reshape(1, len(state)))
            self.q_value = q_value
            action = np.argmax(q_value[0]).item()
        return action

    def appendMemory(self, state, action, reward, next_state, done):
        """
        This function use to append Memory.
        """
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) % 10 == 0:
            now = datetime.now()
            checkpoint_name = "{}_{}_{}_{}_{}.pt".format(now.year, now.month, now.day, now.hour, now.minute)
            os.makedirs(os.path.join(self.root, "checkpoint"), exist_ok=True)
            torch.save(self.memory, os.path.join(self.root, "checkpoint", checkpoint_name))

    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())
        os.makedirs(os.path.join(self.root, "models"), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
        }, os.path.join(self.root, "models", "models.pt"))

    def trainModel(self):
        if len(self.memory) < self.train_start:
            return

        target = True if len(self.memory) > self.target_update else False

        mini_batch_x = []
        mini_batch_y = []

        mini_batch_sample = random.sample(self.memory, self.batch_size)
        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch_sample[i]

            x = torch.tensor(state, dtype=torch.float).reshape(1, len(state))
            nx = torch.tensor(next_state, dtype=torch.float).reshape(1, len(next_state))
            with torch.no_grad():
                q_value = self.model(x)
                sample_y = deepcopy(q_value)
                if target:
                    next_target = self.target_model(nx)
                else:
                    next_target = self.model(nx)

                next_q_value = self.getQvalue(reward, next_target, done)
                sample_y[0][action] = torch.tensor(next_q_value, dtype=torch.float)
            mini_batch_x.append(x)
            mini_batch_y.append(sample_y[0])

        mini_batch_x = torch.stack(mini_batch_x)
        mini_batch_y = torch.stack(mini_batch_y)
        # update
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(mini_batch_x)
        loss = self.loss_function(output.view(10,-1), mini_batch_y)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    print("Create agent: \n")
    ra = ReinforceAgent("./agent_test", 2, 21)
    print("getAction test by model")
    ra.epsilon = -1
    print(ra.getAction((1, 0), random_select=False))
    print("getAction test by random")
    print(ra.getAction((1, 0), random_select=True))

    print("\n")
    print("getQvalue test by done=1")
    print(ra.getQvalue(1, ra.model(torch.tensor((1, 1), dtype=torch.float).reshape(1, len((1, 1)))), 1))
    print("getQvalue test by done=0")
    print(ra.getQvalue(0.5, ra.model(torch.tensor((1, 1), dtype=torch.float).reshape(1, len((1, 1)))), 0))

    print("\n")
    print("updateTargetModel test")
    print(ra.updateTargetModel())

    print("\n")
    print("appendMemory test")
    for _ in tqdm(range(ra.target_update + 10)):
        time.sleep(0.2)
        ra.appendMemory(state=(1, 1), action=3, reward=0.1, next_state=(1, 1), done=0)

    print("\n")
    print("reload test")
    ra = ReinforceAgent("./agent_test", 2, 21, True, True)

    print("\n")
    print("trainModel test")
    for _ in tqdm(range(10)):
        ra.trainModel()
    ra.updateTargetModel()
    # ra = ReinforceAgent("./agent_test", 2, 21, True, True)


