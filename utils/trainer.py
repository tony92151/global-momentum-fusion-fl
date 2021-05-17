import glob
from shutil import copyfile

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
import argparse
import base64
import io, os, json
import time
import copy
import random
from globalfusion.optimizer import GFDGCSGD
from utils.configer import Configer


class trainer:
    def __init__(self, config=None, dataloader=None, device=torch.device("cpu"), cid=-1, writer=None,
                 warmup=None):
        self.config = config
        self.cid = cid
        # self.dataloader = copy.deepcopy(dataloader)
        self.dataloader = dataloader
        self.device = device
        self.last_gradient = None
        self.last_model = None
        self.training_loss = 0
        self.warmup = warmup
        self.optimizer = GFDGCSGD()
        if self.config.trainer.get_lossfun() == "crossentropy":
            self.loss_function = nn.CrossEntropyLoss()

        self.writer = writer
        self.verbose = False

    def print_(self, val):
        if self.verbose:
            print(val)

    def set_mdoel(self, mod):
        self.last_model = copy.deepcopy(mod)

        self.optimizer = GFDGCSGD(params=copy.deepcopy(mod).parameters(),
                                  lr=0.01,
                                  momentum=0.9,
                                  cid=self.cid,
                                  weight_decay=1e-4,
                                  nesterov=True,
                                  dgc_momentum=self.config.dgc.get_momentum(),
                                  compress_ratio=self.config.dgc.get_compress_ratio(),
                                  fusing_ratio=self.config.gf.get_fusing_ratio(),
                                  checkpoint=False,
                                  device=self.device,
                                  pool=None)

    def train_run(self, round_, base_model=None):
        if base_model is None:
            model = copy.deepcopy(self.last_model)
        else:
            model = copy.deepcopy(base_model)

        lr = self.warmup.get_lr_from_step(round_)
        self.optimizer.set_learning_rate(lr=lr)

        model.train().to(self.device)
        self.optimizer.reload_model_parameter(model.parameters())

        # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        # optimizer = SGDD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

        eploss = []
        self.print_("trainer >> cid: {} >> train start, {}".format(self.cid, time.time()))
        # print(">> train", time.time())
        for i in range(1):
            # print("CID: {}, ep :{}".format(round_, i))
            losses = []
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                # data = data.view(data.size(0),-1)

                output = model(data.float())

                loss = self.loss_function(output, target)
                # print(loss.item())
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                # optimizer.gradient_collect()

            losses = sum(losses) / len(losses)
            eploss.append(losses)
            # optimizer.compress(compress=False)
            # local_g.append(optimizer.decompress(optimizer.get_compressed_gradient()))
        self.optimizer.set_accumulate_gradient(model=model, record_batchnorm=True)
        self.print_("trainer >> cid: {} >> compress, {}".format(self.cid, time.time()))
        if not self.config.gf.get_global_fusion() or \
                (round_ < self.config.trainer.get_base_step() and self.config.gf.get_global_fusion_after_warmup()):
            self.optimizer.compress(compress=True, momentum_correction=True)
        else:
            self.optimizer.compress(global_momentum=self.last_gradient["gradient"], compress=True, momentum_correction=True)
        # optimizer.compress(compress=True, momentum_correction=True)

        eploss = sum(eploss) / len(eploss)

        if self.writer is not None:
            self.writer.add_scalar("loss of {}".format(self.cid), eploss, global_step=round_, walltime=None)

        # update bn
        self.last_gradient = copy.deepcopy(self.optimizer.get_compressed_gradient())
        self.training_loss = eploss
        self.print_("trainer >> cid: {} >> done, {}".format(self.cid, time.time()))
        return copy.deepcopy(model)  # , eploss, copy.deepcopy(optimizer.memory.compressed_mem)

    def opt_step_base_model(self, round_, base_model=None, base_gradient=None):
        if base_model is None:
            model = copy.deepcopy(self.last_model)
        else:
            model = copy.deepcopy(base_model)

        lr = self.warmup.get_lr_from_step(round_)

        model.to(self.device).train()
        optimizer = GFDGCSGD(params=model.parameters(), lr=lr, device=self.device)
        # self.last_gradient = copy.deepcopy(base_gradient)
        # optimizer.set_gradient(base_gradient)
        base_gradient["gradient"] = [t.to(self.device) for t in base_gradient["gradient"]]
        optimizer.one_step(base_gradient["gradient"])

        if 'bn' in self.last_gradient.keys():
            idx = 0
            for layer in model.cpu().modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.running_mean = torch.tensor(self.last_gradient["bn"][idx]).clone().detach()
                    layer.running_var = torch.tensor(self.last_gradient["bn"][idx + 1]).clone().detach()
                    layer.num_batches_tracked = torch.tensor(self.last_gradient["bn"][idx + 2]).clone().detach()
                    idx += 3
        self.last_model = copy.deepcopy(model)
        return copy.deepcopy(model)
