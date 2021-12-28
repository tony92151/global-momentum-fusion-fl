import os
from typing import List

import numpy as np
from utils.configer import Configer


class warmup_scheduler:
    def __init__(self, start_lr=0.0001, max_lr=0.001, min_lr=0.0001, base_step=10, end_step=50):
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.base_step = base_step
        self.end_step = end_step

    def get_lr_from_step(self, step):
        if step < self.base_step:
            retult = self.start_lr + step * (self.max_lr - self.start_lr) / self.base_step
        elif step == self.base_step:
            retult = self.max_lr
        elif step > self.base_step and step < self.end_step:
            c = (np.cos(np.pi * (step - self.base_step) / (self.end_step - self.base_step)) + 1) / 2
            retult = self.max_lr - (self.max_lr - self.min_lr) * (1 - c)
        elif step >= self.end_step:
            retult = self.min_lr
        return round(retult, 8)


class compress_rate_scheduler:
    def __init__(self, max_iteration: int, compress_rate_list: List):
        self.max_iteration = max_iteration
        self.compress_rate_list = compress_rate_list
        self.chunk = self.max_iteration / len(self.compress_rate_list)

    def get_compress_rate_from_step(self, step):
        cr = self.compress_rate_list[min(len(self.compress_rate_list), int(step / self.chunk))]
        return cr


class fusion_ratio_scheduler:
    def __init__(self, max_iteration: int, fusing_ratio_list: List):
        self.max_iteration = max_iteration
        self.fusing_ratio_list = fusing_ratio_list
        self.chunk = self.max_iteration / len(self.fusing_ratio_list)

    def get_fusion_ratio_from_step(self, step):
        fr = self.fusing_ratio_list[min(len(self.fusing_ratio_list), int(step / self.chunk))]
        return fr
