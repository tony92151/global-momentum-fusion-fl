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
from typing import List

from sparse_optimizer.warmup import warmup
from utils.aggregator import aggregater, momentum_aggregater,decompress, compress, parameter_count
from utils.configer import Configer
from utils.dataloaders import cifar_dataloaders, femnist_dataloaders, DATALOADER
from utils.eval import evaluater, lstm_evaluater
from utils.models import MODELS
from utils.trainer import trainer, lstm_trainer


class server:
    def __init__(self, config: Configer, device=torch.device("cpu")):
        self.config = config
        if self.config.trainer.get_optimizer() == "SGCSGD":
            self.aggregater = momentum_aggregater(server_momentun=config.sgc.get_global_momentum(),
                                                  device=device)
        else:
            self.aggregater = aggregater(device=device)

    def aggregate(self, gs: List[dict], aggrete_bn=False):
        aggregated_gradient = self.aggregater.aggregate(gs, aggrete_bn=aggrete_bn)
        return aggregated_gradient
