import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, sampler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
import pandas as pd
import random
import numpy as np
import sys, os, json
import pickle
from PIL import Image

import argparse
import copy, sys
import glob
import json
import os, copy
import time

import matplotlib.pyplot as plt
import numpy as np
from utils.configer import Configer


class evaluater:
    def __init__(self, config=None, dataloader=None, device=torch.device("cpu"), writer=None):
        self.config = config
        self.dataloader = copy.deepcopy(dataloader)
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()

        self.writer = writer

        ####
        self.sampled_data = None
        ####

        self.round = None
        self.last_acc = None
        self.last_loss = None
        self.verbose = False

    def print_(self, val):
        if self.verbose:
            print(val)

    def sample_data_from_dataloader(self):
        self.sampled_data = []
        for data, target in self.dataloader:
            self.sampled_data.append((data, target))

    def eval_run(self, model, round_=None):
        if round_ is None:
            round_ = self.round
        losses = []
        ans = np.array([])
        res = np.array([])
        correct = 0
        model.eval().to(self.device)
        self.print_("eval >> eval start, {}".format(time.time()))
        with torch.no_grad():
            for data, target in self.sampled_data:
                # data = data.view(data.size(0),-1)
                data = data

                data = data.to(self.device)
                target = target.to(self.device)

                output = model(data)

                loss = self.loss_function(output, target)
                losses.append(loss.item())

                _, preds_tensor = output.max(1)
                correct += preds_tensor.eq(target).sum().item()

        losses = sum(losses) / len(losses)
        acc = correct / len(self.dataloader.dataset)
        self.print_("eval >> eval done, {}".format(time.time()))
        if self.writer is not None:
            self.writer.add_scalar("test loss", losses, global_step=round_, walltime=None)
            self.writer.add_scalar("test acc", acc, global_step=round_, walltime=None)
        return acc, losses


class lstm_evaluater(evaluater):
    def __init__(self, **kwargs):
        super(lstm_evaluater, self).__init__(**kwargs)

    def eval_run(self, model, round_=None):
        if round_ is None:
            round_ = self.round
        losses = []
        ans = np.array([])
        res = np.array([])
        correct = 0
        model.eval().to(self.device)
        self.print_("eval >> eval start, {}".format(time.time()))
        lstm_state = model.zero_state(batch_size=self.config.trainer.get_local_bs(), device=self.device)
        with torch.no_grad():
            for data, target in self.sampled_data:
                # data = data.view(data.size(0),-1)
                data = data

                data = data.to(self.device)
                target = target.to(self.device)

                output, lstm_state = model(data, lstm_state)

                loss = self.loss_function(output, target)
                losses.append(loss.item())

                _, preds_tensor = output.max(1)
                correct += preds_tensor.eq(target).sum().item()

        losses = sum(losses) / len(losses)
        acc = correct / len(self.dataloader.dataset)
        self.print_("eval >> eval done, {}".format(time.time()))
        if self.writer is not None:
            self.writer.add_scalar("test loss", losses, global_step=round_, walltime=None)
            self.writer.add_scalar("test acc", acc, global_step=round_, walltime=None)
        return acc, losses



