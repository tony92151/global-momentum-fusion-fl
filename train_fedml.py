import argparse
import copy
import json
import os, sys
import time
from concurrent.futures import as_completed
from bounded_pool_executor import BoundedThreadPoolExecutor as ThreadPoolExecutor

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from globalfusion.warmup import warmup
from utils.aggregator import aggregater, decompress, get_serialize_size
from utils.configer import Configer
from utils.dataloaders import cifar_dataloaders, femnist_dataloaders, DATALOADER
from utils.eval import evaluater
from utils.models import MODELS
from utils.trainer import trainer

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import wandb
from FedML.fedml_api.model.cv.resnet import resnet56
from FedML.fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from FedML.fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS

torch.manual_seed(0)


def init_writer(tbpath):
    # tbpath = "/root/notebooks/tensorflow/logs/test"
    os.makedirs(tbpath, exist_ok=True)
    writer = SummaryWriter(tbpath)
    print("$ tensorboard --logdir={} --port 8123 --host 0.0.0.0 \n".format(os.path.dirname(tbpath)))
    print("\nName: {}".format(tbpath.split("/")[-1]))
    return writer


def add_args(parser):
    """
    parser: argparse.ArgumentParser
    return a
    parser
    added
    with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')


    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    return parser


def load_data(dataloaders):
    # dataloaders["test"]: single loader
    # dataloaders["train"]: single loader
    # dataloaders["train_s"]: loaders
    # dataloaders["test_s"]: loaders
    # dataloaders["train_s_iid"]: loaders with iid data

    train_data_num = len(dataloaders["train"].dataset)
    test_data_num = len(dataloaders["test"].dataset)
    train_data_global = dataloaders["train"]
    test_data_global = dataloaders["test"]
    train_data_local_num_dict = dict()
    for i, loader in enumerate(dataloaders["train_s"]):
        train_data_local_num_dict[i] = len(loader.dataset)

    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for i, loader in enumerate(dataloaders["train_s"]):
        train_data_local_dict[i] = loader
    for i, loader in enumerate(dataloaders["train_s"]):
        test_data_local_dict[i] = dataloaders["test"]
    class_num = 10
    # train_data_num:               number of total train data
    # test_data_num:                number of total test data
    # train_data_global:            train data -> [[batch0_0, batch0_1,...], [batch1_0, batch1_1,...],...]
    # test_data_global:             test data -> [[batch0_0, batch0_1,...], [batch1_0, batch1_1,...],...]
    # train_data_local_num_dict:    total number of total train data in each clients -> {0: 10, 1:20,...}
    # train_data_local_dict:        train data -> {0: [[batch0_0, batch0_1,...],...], 1:[[batch0_0, batch0_1,...],...],...}
    # test_data_local_dict:         test data -> {0: [[batch0_0, batch0_1,...],...], 1:[[batch0_0, batch0_1,...],...],...}
    # class_num:                    ->10
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(model_name, output_dim):
    pass


def custom_model_trainer(model):
    return MyModelTrainerCLS(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config", type=str, default=None)
    parser.add_argument('--output', help="output", type=str, default=None)
    # parser.add_argument('--gpu', help="GPU usage. ex: 0,1,2", type=str, default="0")

    add_args(parser)
    args = parser.parse_args()

    if args.config is None or args.output is None:
        print("Please set --config & --output.")
        exit()

    con_path = os.path.abspath(args.config)
    out_path = os.path.abspath(args.output)
    os.makedirs(out_path, exist_ok=True)

    # gpus = [int(i) for i in args.gpu.split(",")]
    # if len(gpus) > torch.cuda.device_count() or max(gpus) > torch.cuda.device_count():
    #     raise ValueError("GPU unavailable.")
    # else:
    #     print("\nGPU usage: {}".format(gpus))

    print("Read config at : {}".format(con_path))
    config = Configer(con_path)

    tb_path = os.path.join(config.general.get_tbpath(),
                           str(int(time.time()))) if config.general.get_tbpath() == "./tblogs" \
        else config.general.get_tbpath()

    writer = init_writer(tbpath=os.path.abspath(tb_path))

    print("\nInit dataloader...")
    dataloaders = DATALOADER(config)
    dataset = load_data(dataloaders)

    print("\nInit model...")
    # model = MODELS(config)()
    model = resnet56(class_num=10)

    # Init trainers
    print("\nInit trainers...")
    model_trainer = custom_model_trainer(model)

    wandb.init(
        project="fedml",
        name="FedAVG-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr) + "single",
        config=args
    )

    print("\nStart training...")
    fedavgAPI = FedAvgAPI(dataset, torch.device("cuda:{}".format(args.gpu)), args, model_trainer)
    fedavgAPI.train()
