# This script is an implement of this paper.
# https://arxiv.org/abs/1806.00582

import torch
import copy
from utils.aggregator import set_gradient
from utils.opti import SERVEROPTS, FEDOPTS
import numpy as np
from torch.utils.data import DataLoader
import tqdm

# earth_moving_distance = sum_from_i=1_to_C(|| P^k(y=i) - P(y=i) ||)


def earth_moving_distance(dataloaders: DataLoader = None, number_of_calss=10):
    print("Calculate earth_moving_distance of this dataloader...")
    if dataloaders is None or not isinstance(dataloaders, list):
        raise ValueError("Error dataloaders.")

    # data_statistics = [
    #  [11,56,23,...10],   -> Dataloader1 have 11 images in class 0 ...
    #  [31,51,13,...1],    -> Dataloader2 have 11 images in class 0 ...
    #  ...
    #  ]
    data_statistics = [[0 for _ in range(number_of_calss)] for _ in dataloaders]
    for i in tqdm(range(len(dataloaders))):
        for x, y in dataloaders[i]:
            for target in y:
                data_statistics[i][int(target)] += 1

    # data_statistics_sum = [1341,5226,1234,...3210],   -> Total have 1341 images in class 0 ...
    data_statistics_sum = np.array(data_statistics).sum(axis=0).tolist()
    total_data = sum(data_statistics_sum)

    emds_all_dataloader = []
    for i in range(number_of_calss):
        embs = []
        for ds in data_statistics:
            emd = (ds[i] / sum(ds)) - (data_statistics_sum[i] / total_data)
            embs.append(emd)
        norm_result = torch.norm(torch.tensor(embs, dtype=torch.float).unsqueeze(0), dim=1).tolist()[0]
        emds_all_dataloader.append(norm_result)

    return sum(emds_all_dataloader)




