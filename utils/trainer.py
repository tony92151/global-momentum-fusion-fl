import glob
from shutil import copyfile
from collections import OrderedDict
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
import numpy as np
from globalfusion.gfcompressor import GFCCompressor
from globalfusion.optimizer import GFDGCSGD
from utils.configer import Configer
from utils.opti import SERVEROPT
from utils.aggregator import set_gradient
from torch_optimizer import Yogi


class trainer:
    def __init__(self, config=None, dataloader=None, dataloader_iid=None, device=torch.device("cpu"), cid=-1,
                 writer=None,
                 warmup=None):
        self.config: Configer = config
        self.cid = cid
        # self.dataloader = copy.deepcopy(dataloader)
        self.dataloader = dataloader
        self.dataloader_iid = dataloader_iid
        self.device = device
        self.round = None
        self.last_gradient = None
        self.last_de_gradient = None
        self.global_momentum = None
        self.last_model = None
        self.last_state = None
        self.training_loss = 0
        self.warmup = warmup
        self.optimizer: GFDGCSGD = None

        self.weight_divergence = None
        if self.config.trainer.get_lossfun() == "crossentropy":
            self.loss_function = nn.CrossEntropyLoss()

        self.writer = writer
        self.verbose = False

    def print_(self, val):
        if self.verbose:
            print(val)

    def set_mdoel(self, mod):
        self.last_model = copy.deepcopy(mod)
        self.weight_divergence = OrderedDict()
        names = [i[0] for i in self.last_model.named_parameters()]
        for i in names:
            self.weight_divergence[i] = 0.0

    def train_run(self, round_, base_model=None):
        if base_model is None:
            model = copy.deepcopy(self.last_model)
        else:
            model = copy.deepcopy(base_model)

        lr = self.warmup.get_lr_from_step(round_)
        model.train().to(self.device)
        chunk = self.config.trainer.get_max_iteration() / len(self.config.dgc.get_compress_ratio())
        chunk_ = self.config.trainer.get_max_iteration() / len(self.config.gf.get_fusing_ratio())
        cr = self.config.dgc.get_compress_ratio()[min(len(self.config.dgc.get_compress_ratio()), int(round_ / chunk))]
        fr = self.config.gf.get_fusing_ratio()[min(len(self.config.gf.get_fusing_ratio()), int(round_ / chunk_))]
        if self.cid == 0 and self.writer is not None:
            self.writer.add_scalar("Compress ratio", cr, global_step=round_, walltime=None)
            self.writer.add_scalar("Fusion ratio", fr, global_step=round_, walltime=None)
        optimizer = GFDGCSGD(params=model.parameters(),
                             lr=lr,
                             momentum=0.9,
                             cid=self.cid,
                             weight_decay=1e-4,
                             nesterov=True,
                             dgc_momentum=self.config.dgc.get_momentum(),
                             compress_ratio=cr,
                             fusing_ratio=fr,
                             checkpoint=False,
                             device=self.device,
                             pool=None)

        if self.last_state is not None:
            optimizer.set_state(self.last_state)
        # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        # optimizer = SGDD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

        eploss = []
        self.print_("trainer >> cid: {} >> train start, {}".format(self.cid, time.time()))
        for i in range(self.config.trainer.get_local_ep()):
            losses = []
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.loss_function(output, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            losses = sum(losses) / len(losses)
            eploss.append(losses)

        optimizer.set_accumulate_gradient(model=model, record_batchnorm=True)
        self.print_("trainer >> cid: {} >> compress, {}".format(self.cid, time.time()))
        ############################################################
        if self.config.dgc.get_dgc():
            if not self.config.gf.get_global_fusion() or \
                    (round_ < self.config.trainer.get_base_step() and self.config.gf.get_global_fusion_after_warmup()):
                optimizer.compress(compress=True, momentum_correction=True)
            else:
                optimizer.compress(global_momentum=self.global_momentum, compress=True,
                                   momentum_correction=True)
        else:
            optimizer.compress(compress=False, momentum_correction=False)
        ############################################################
        eploss = sum(eploss) / len(eploss)
        if self.writer is not None:
            self.writer.add_scalar("loss of {}".format(self.cid), eploss, global_step=round_, walltime=None)
        # update bn
        self.last_gradient = copy.deepcopy(optimizer.get_compressed_gradient())
        self.training_loss = eploss
        self.print_("trainer >> cid: {} >> done, {}".format(self.cid, time.time()))
        self.last_state = optimizer.get_state()
        del optimizer
        del model
        return

    def mask(self, val):
        for t in range(len(val)):
            _, ctx = self.last_gradient["gradient"][t]
            shape, mask, _ = ctx
            mask = torch.tensor(mask).view(shape)
            val[t].mul_(mask.float())

    def wdv_test(self, round_, gradients=None, agg_gradient=None, compare_with=None, mask=False,
                 weight_distribution=False, layer_info=False):
        if self.writer is None:
            raise ValueError("Tensorboard writer not define.")
        types = ["iid", "momentum", "agg"]
        if compare_with is None or compare_with not in types:
            raise ValueError("compare_with should be {}".format(types))
        d_niid = gradients[self.cid]["gradient"]

        if compare_with == "iid":
            d_compare = self.train_run_iid(round_)
            if mask:
                self.mask(d_compare)
        elif compare_with == "momentum":
            d_compare = self.global_momentum if self.global_momentum is not None else agg_gradient["gradient"]
            if mask:
                self.mask(d_compare)
        elif compare_with == "agg":
            d_compare = agg_gradient["gradient"]

        dvs = []
        for i in range(len(d_compare)):
            dv = torch.norm(
                torch.add(d_compare[i].to(self.device), d_niid[i].to(self.device), alpha=-1.0)) / torch.norm(
                d_niid[i].to(self.device))
            dvs.append(dv)
            self.weight_divergence[list(self.weight_divergence.keys())[i]] = dv
            if layer_info:
                self.writer.add_scalar("{} wdv layer {}".format(self.cid, list(self.weight_divergence.keys())[i])
                                       , dv, global_step=round_, walltime=None)
        dvs = sum(dvs) / len(dvs)
        if layer_info:
            self.writer.add_scalar("{} wdv avg".format(self.cid), dvs, global_step=round_, walltime=None)

        if weight_distribution:
            measurement_max = 2
            measurement_min = -2
            chunk_size = 0.0001
            chunks = [round(measurement_min + (i * chunk_size), 5)
                      for i in range(int((measurement_max - measurement_min) / chunk_size))]
            params = np.array([])
            for i in self.last_model.parameters():
                param = i.flatten()
                param = param[param >= measurement_min]
                param = param[param >= measurement_max]
                params = np.append(params, param.detach().numpy())

            values = np.array(params).astype(float).reshape(-1)
            counts, limits = np.histogram(values, bins=chunks)
            sum_sq = values.dot(values)
            self.writer.add_histogram_raw(
                tag='{} weight_distribution_histogram'.format(self.cid),
                min=values.min(),
                max=values.max(),
                num=len(values),
                sum=values.sum(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
                global_step=0)
            # self.last_model
            # self.last_de_gradient

    def train_run_iid(self, round_, base_model=None):
        if base_model is None:
            model = copy.deepcopy(self.last_model)
        else:
            model = copy.deepcopy(base_model)

        lr = self.warmup.get_lr_from_step(round_)
        model.train().to(self.device)
        chunk = self.config.trainer.get_max_iteration() / len(self.config.dgc.get_compress_ratio())
        chunk_ = self.config.trainer.get_max_iteration() / len(self.config.gf.get_fusing_ratio())
        cr = self.config.dgc.get_compress_ratio()[min(len(self.config.dgc.get_compress_ratio()), int(round_ / chunk))]
        fr = self.config.gf.get_fusing_ratio()[min(len(self.config.gf.get_fusing_ratio()), int(round_ / chunk_))]
        optimizer = GFDGCSGD(params=model.parameters(),
                             lr=lr,
                             momentum=0.9,
                             cid=self.cid,
                             weight_decay=1e-4,
                             nesterov=True,
                             dgc_momentum=self.config.dgc.get_momentum(),
                             compress_ratio=1.0,
                             fusing_ratio=fr,
                             checkpoint=False,
                             device=self.device,
                             pool=None)
        if self.last_state is not None:
            optimizer.set_state(self.last_state)
        eploss = []
        self.print_("trainer_iid >> cid: {} >> train start, {}".format(self.cid, time.time()))
        for i in range(self.config.trainer.get_local_ep()):
            losses = []
            for data, target in self.dataloader_iid:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data.float())
                loss = self.loss_function(output, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            losses = sum(losses) / len(losses)
            eploss.append(losses)

        optimizer.set_accumulate_gradient(model=model, record_batchnorm=True)
        self.print_("trainer_iid >> cid: {} >> compress, {}".format(self.cid, time.time()))
        ############################################################
        if self.config.dgc.get_dgc():
            if not self.config.gf.get_global_fusion() or \
                    (round_ < self.config.trainer.get_base_step() and self.config.gf.get_global_fusion_after_warmup()):
                optimizer.compress(compress=True, momentum_correction=True)
            else:
                optimizer.compress(global_momentum=self.global_momentum, compress=True,
                                   momentum_correction=True)
        else:
            optimizer.compress(compress=False, momentum_correction=False)
        ############################################################
        eploss = sum(eploss) / len(eploss)
        # if self.writer is not None:
        #     self.writer.add_scalar("loss of {}".format(self.cid), eploss, global_step=round_, walltime=None)
        iid_last_gradient = copy.deepcopy(optimizer.get_compressed_gradient())
        d_iid = optimizer.decompress(iid_last_gradient["gradient"])

        return d_iid

    def opt_step_base_model(self, base_gradient=None, round_=None, base_model=None):
        self.print_("trainer >> cid: {} >> opt_step, {}".format(self.cid, time.time()))
        if base_model is None:
            model = copy.deepcopy(self.last_model)
        else:
            model = copy.deepcopy(base_model)

        if round_ is None:
            round_ = self.round
        lr = self.warmup.get_lr_from_step(round_)

        model.to(self.device).train()
        opt = self.config.agg.get_optimizer()
        optimizer = SERVEROPT[opt](params=model.parameters(), lr=lr)
        # optimizer = GFDGCSGD(params=model.parameters(), lr=lr, device=self.device)
        base_gradient["gradient"] = [t.to(self.device) for t in base_gradient["gradient"]]
        # optimizer.one_step(base_gradient["gradient"])
        set_gradient(opt=optimizer, cg=base_gradient["gradient"])
        optimizer.step()
        if 'bn' in self.last_gradient.keys():
            idx = 0
            for layer in model.cpu().modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.running_mean = torch.tensor(self.last_gradient["bn"][idx]).clone().detach()
                    layer.running_var = torch.tensor(self.last_gradient["bn"][idx + 1]).clone().detach()
                    layer.num_batches_tracked = torch.tensor(self.last_gradient["bn"][idx + 2]).clone().detach()
                    idx += 3
        self.last_model = copy.deepcopy(model)
        self.last_de_gradient = copy.deepcopy(base_gradient)
        if round_ >= self.config.trainer.get_base_step() - 1:
            self.update_global_momentum()
        return

    def update_global_momentum(self):
        if self.global_momentum is None and self.last_de_gradient is not None:
            self.global_momentum = copy.deepcopy(self.last_de_gradient["gradient"])
        else:
            for i in range(len(self.global_momentum)):
                self.global_momentum[i].mul_(0.8).add_(self.last_de_gradient["gradient"][i])
