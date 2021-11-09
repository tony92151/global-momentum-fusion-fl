import argparse
import copy
import json
import os
import random
import time
from concurrent.futures import as_completed
from bounded_pool_executor import BoundedThreadPoolExecutor as ThreadPoolExecutor

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sparse_optimizer.warmup import warmup
from utils.aggregator import aggregater, decompress, compress, parameter_count
from utils.configer import Configer
from utils.dataloaders import cifar_dataloaders, femnist_dataloaders, DATALOADER
from utils.eval import evaluater, lstm_evaluater
from utils.models import MODELS
from utils.trainer import trainer, lstm_trainer


class client_manager:
    def __init__(self, config: Configer,
                 gpus=0,
                 warmup_scheduler=None,
                 writer=None,
                 executor=None):
        self.trainers = []
        self.config = config
        self.executor = executor

        print("\nInit dataloader...")
        dataloaders, emb = DATALOADER(config=config, emd_measurement=True)

        # write earth_moving_distance
        writer.add_scalar("earth_moving_distance", emb, global_step=0, walltime=None)

        if "lstm" in config.trainer.get_model():
            evaluater_ = lstm_evaluater
        else:
            evaluater_ = evaluater
        self.evaluater = evaluater_(config=config,
                                    dataloader=dataloaders["test"],
                                    device=torch.device("cuda:{}".format(gpus[0])),
                                    writer=None)

        # Init trainers
        print("\nInit trainers...")
        print("Nodes: {}".format(config.general.get_nodes()))
        print(">>>{}<<<".format(config.trainer.get_model()))
        if "lstm" in config.trainer.get_model():
            trainer_ = lstm_trainer
            print("latm")
        else:
            trainer_ = trainer
        for i in tqdm(range(config.general.get_nodes())):
            self.trainers.append(
                trainer_(config=config,
                         device=torch.device("cuda:{}".format(gpus[i % len(gpus)])),
                         dataloader=dataloaders["train_s"][i],
                         cid=i,
                         writer=writer,
                         warmup=warmup_scheduler)
            )

        self.sampled_trainer = []

    def set_init_mdoel(self):
        print("\nInit model...")
        net = MODELS(self.config)()
        for tr in self.trainers:
            tr.set_mdoel(net)

        return net

    def set_sampled_trainer(self, cids):
        self.sampled_trainer = cids

    def sample_data(self):
        for tr in self.trainers:
            # if tr.cid in self.sampled_trainer:
            #     tr.sample_data_from_dataloader()
            tr.sample_data_from_dataloader()
        self.evaluater.sample_data_from_dataloader()

    def training(self, epoch):
        gs = []
        if self.executor is not None:
            futures = []
            for tr in self.trainers:
                if tr.cid in self.sampled_trainer:
                    futures.append(self.executor.submit(tr.train_run, epoch))
                else:
                    futures.append(self.executor.submit(tr.eval_run, epoch))

            for _ in as_completed(futures):
                pass
            del futures

            for tr in self.trainers:
                if tr.cid in self.sampled_trainer:
                    gs.append(tr.last_gradient)
        else:
            for i, tr in enumerate(self.trainers):
                if tr.cid in self.sampled_trainer:
                    _ = tr.train_run(round_=epoch)
                else:
                    _ = tr.eval_run(round_=epoch)
            for tr in self.trainers:
                if tr.cid in self.sampled_trainer:
                    gs.append(tr.last_gradient)

        return gs

    def opt_one_step(self, epoch, aggregated_gradient):
        for tr in self.trainers:
            _ = tr.opt_step_base_model(round_=epoch, base_gradient=aggregated_gradient)
