import argparse
import glob
import json
import os
import sys
import time
import random

import torch

sys.path.append("../")
from utils.weight_divergence.emd import earth_moving_distance
from utils.dataloaders import femnist_dataloaders, MNISTDataset

"""
structure in .pt dataset

train_data.pt = {
    'users': ['f0087_24', 'f0029_05','f0063_38', ...],
    'num_samples': [292, 223, 372, ...], 
    'user_data': {'f0087_24': {'x':[image_1, image_2, image_3, ...], 'y': [47,25,55, ...]},
                  'f0029_05': ...,
                  'f0063_38': ...,
                  ...
                 },
}

Total clients: 3560

"""


def set_seed(value):
    random.seed(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=200, help="Separate to n datasets")
    parser.add_argument('--data', type=str, default="./femnist", help="Path to .pt dataset")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('-f')
    args = parser.parse_args()

    if args.num_clients > 3560:
        raise ValueError("Request clients({}) larger than dataset provide({}).".format(args.num_clients, 3560))

    print("\nThis might take 10 min.")
    # path
    root = os.path.abspath(args.data)
    print("\nLoad dataset's name {}".format(os.path.join(root, "train_data_name_target_only.pt")))
    train_data_name_target = torch.load(os.path.join(root, "train_data_name_target_only.pt"))

    # sample
    set_seed(args.seed)
    print("\nSample {} users in dataset.".format(args.num_clients))
    sampled_clients_name = random.sample(train_data_name_target['users'], args.num_clients)

    print(sampled_clients_name)
    # cdataloders = femnist_dataloaders(root=root, clients=sampled_clients_name)

    trainloaders = []
    for name in sampled_clients_name:
        trainloaders.append(torch.utils.data.DataLoader(
            MNISTDataset(feature=torch.tensor(train_data_name_target["user_data"][name]["x"]),
                         target=torch.tensor(train_data_name_target["user_data"][name]["y"])),
            batch_size=10
        ))

    emd = earth_moving_distance(dataloaders=trainloaders, number_of_class=62)
    print("\nEarth moving distance: ", emd)

    cdataloders = femnist_dataloaders(root=root, clients=sampled_clients_name)
    emd = earth_moving_distance(dataloaders=cdataloders["train_s"], number_of_class=62)
    print("\nEarth moving distance: ", emd)


    pass
