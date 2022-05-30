import copy
import os
import json
import numpy as np
import random
import time
import sys
from collections import deque

from qdn.model import DQN


class ReinforceAgent:
    def __init__(self, state_size, action_size):

        self.load_model = False
        self.save_model = True

        self.load_memory = False
        self.save_memory = False

        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        # train self.regular_model per self.target_update steps
        self.target_update = 100
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 10

        # if len of memory > self.train_start, star training
        self.train_start = 50
        self.memory = deque(maxlen=1000)
        self.main_model = self.buildModel()
        self.target_model = copy.deepcopy(self.main_model)
        self.updateTargetModel()

        if self.load_model:
            pass

        if self.load_memory:
            pass

    def restore(self, memory, model):
        pass

    def buildModel(self):
        return DQN(self.state_size, self.action_size)

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        # self.target_model.set_weights(self.model.get_weights())
        pass

    def getAction(self, state):
        """
        This function use to determined next action.
        We setting self.epsilon to give an opportunity to try new action

        If self.epsilon = 0.2, mean 20% random action
        """
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.main_model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done, cache=None):
        """
        This function use to append Memory.
        """
        if cache:
            # do something
            pass
        self.memory.append((state, action, reward, next_state, done, cache))

    def trainModel(self, target=False):
        """
        This function use to append Memory.

        If target = True, mean we get next_target by self.target_model.
        """
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        pass

        # for i in range(self.batch_size):
        #     states = mini_batch[i][0]
        #     actions = mini_batch[i][1]
        #     rewards = mini_batch[i][2]
        #     next_states = mini_batch[i][3]
        #     dones = mini_batch[i][4]
        #
        #     q_value = self.model.predict(states.reshape(1, len(states)))
        #     self.q_value = q_value
        #
        #     if target:
        #         next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))
        #
        #     else:
        #         next_target = self.model.predict(next_states.reshape(1, len(next_states)))
        #
        #     next_q_value = self.getQvalue(rewards, next_target, dones)
        #
        #     X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
        #     Y_sample = q_value.copy()
        #
        #     Y_sample[0][actions] = next_q_value
        #     Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)
        #
        #     if dones:
        #         X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
        #         Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)
        #
        # self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0, callbacks=[tbCallBack])
