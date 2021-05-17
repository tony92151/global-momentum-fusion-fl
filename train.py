import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import os, copy, json, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from utils.configer import Configer
import time, copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from globalfusion.warmup import warmup
from torch.utils.tensorboard import SummaryWriter
from utils.dataloaders import cifar_dataloaders, femnist_dataloaders
from utils.trainer import trainer
from utils.aggregator import aggrete, decompress, get_serialize_size
from utils.eval import evaluater
from utils.models import *

torch.manual_seed(0)


def init_writer(tbpath):
    # tbpath = "/root/notebooks/tensorflow/logs/test"
    os.makedirs(tbpath, exist_ok=True)
    writer = SummaryWriter(tbpath)
    print("$ tensorboard --logdir={} --port 8123 --host 0.0.0.0 \n".format(os.path.dirname(tbpath)))
    return writer


def run(job, arg):
    return job(arg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config", type=str, default=None)
    parser.add_argument('--output', help="output", type=str, default=None)
    parser.add_argument('--pool', help="Multiprocess Worker Pools", type=str, default="-1")
    parser.add_argument('--gpu', help="GPU usagre. ex: 0,1,2", type=str, default="0")
    args = parser.parse_args()

    if args.config is None or args.output is None:
        print("Please set --config & --output.")
        exit()

    con_path = os.path.abspath(args.config)
    out_path = os.path.abspath(args.output)
    os.makedirs(out_path, exist_ok=True)

    gpus = [int(i) for i in args.gpu.split(",")]
    if len(gpus) > torch.cuda.device_count() or max(gpus) > torch.cuda.device_count():
        raise ("GPU unavailable.")
    else:
        print("\nGPU uasge: {}".format(gpus))

    print("Read cinfig at : {}".format(con_path))
    config = Configer(con_path)

    num_pool = args.pool
    if num_pool == "auto":
        num_pool = int(config.general.get_nodes() / 2) + 5
        print("\nPool: {}".format(num_pool))
    else:
        num_pool = int(num_pool)
        print("\nPool: {}".format(num_pool))

    tb_path = os.path.join(config.general.get_tbpath(),
                           str(int(time.time()))) if config.general.get_tbpath() == "./tblogs" \
        else config.general.get_tbpath()

    writer = init_writer(tbpath=os.path.abspath(tb_path))

    w = warmup(start_lr=config.trainer.get_start_lr(),
               max_lr=config.trainer.get_max_lr(),
               min_lr=config.trainer.get_min_lr(),
               base_step=config.trainer.get_base_step(),
               end_step=config.trainer.get_end_step())

    print("\nInit dataloader...")
    if "cifar10" in config.trainer.get_dataset_path():
        dataloaders = cifar_dataloaders(root=config.trainer.get_dataset_path(),
                                        index_path=os.path.join(config.trainer.get_dataset_path(),
                                                                config.trainer.get_dataset_type(), "index.json"),
                                        batch_size=config.trainer.get_local_bs())
    elif "femnist" in config.trainer.get_dataset_path():
        dataloaders = femnist_dataloaders(root=config.trainer.get_dataset_path(),
                                          batch_size=config.trainer.get_local_bs(),
                                          clients=config.general.get_nodes())

    print("Total train images: {}".format(len(dataloaders["train"].dataset)))
    print("Total test images: {}".format(len(dataloaders["test"].dataset)))

    # Init trainers
    print("\nInit trainers...")
    print("Nodes: {}".format(config.general.get_nodes()))
    trainers = []
    for i in tqdm(range(config.general.get_nodes())):
        trainers.append(trainer(config=config,
                                device=torch.device("cuda:{}".format(gpus[i % len(gpus)])),
                                dataloader=dataloaders["train_s"][i],
                                cid=i,
                                writer=writer,
                                warmup=w))

    # net = Net()
    model_table = {
        # for cifar10
        "resnet18_cifar": ResNet18_cifar,
        "resnet50_cifar": ResNet50_cifar,
        "resnet101_cifar": ResNet101_cifar,
        "small_cifar": Net_cifar,
        "resnet110_cifar": ResNet110_cifar_gdc,
        # for femnist
        "small_femnist": Net_femnist,
        "resnet9_femnist": ResNet9_femnist,
        "resnet18_femnist": ResNet18_femnist,
        "resnet50_femnist": ResNet50_femnist,
        "resnet101_femnist": ResNet101_femnist,
    }
    print("\nInit model...")
    net = model_table[config.trainer.get_model()]()

    for tr in trainers:
        tr.set_mdoel(net)

    # Init trainers evaluater
    ev = evaluater(config=config, dataloader=dataloaders["test"], device=torch.device("cuda:0"), writer=None)

    # inint traffic simulator
    traffic = 0
    traffic += get_serialize_size(net) * 4  # 4 clients download
    # train
    print("\nStart training...")
    for epoch in tqdm(range(config.trainer.get_max_iteration())):
        gs = []
        if not num_pool == -1:
            futures = []
            executor = ThreadPoolExecutor(max_workers=num_pool)
            for tr in trainers:
                future = executor.submit(tr.train_run, epoch)
                futures.append(future)
            executor.shutdown(True)
            for tr in trainers:
                gs.append(tr.last_gradient)

        else:
            for i, tr in zip(range(len(trainers)), trainers):
                # print("trainer: {}".format(i), time.time())
                _ = tr.train_run(round_=epoch)
                # print("trainer done: {}".format(i), time.time())
                gs.append(tr.last_gradient)
                # writer.add_scalar("loss of {}".format(i), tr.training_loss, global_step=epoch, walltime=None)
        # print("decompress", time.time())
        for i in range(len(gs)):
            gs[i]["gradient"] = decompress(gs[i]["gradient"], device=torch.device("cuda:0"))

        rg = aggrete(gs, device=torch.device("cuda:0"), aggrete_bn=False)
        # print("one step", time.time())
        for tr in trainers:
            tm = tr.opt_step_base_model(round_=epoch, base_gradient=rg)
        # print("eval_run", time.time())
        # eval
        test_acc = []
        test_loss = []
        for tr in trainers:
            a, l = ev.eval_run(model=copy.deepcopy(tr.last_model), round_=epoch)
            test_acc.append(a)
            test_loss.append(l)

        test_acc = sum(test_acc) / len(test_acc)
        test_loss = sum(test_loss) / len(test_loss)
        writer.add_scalar("test loss", test_loss, global_step=epoch, walltime=None)
        writer.add_scalar("test acc", test_acc, global_step=epoch, walltime=None)

        writer.add_scalar("traffic(MB)", traffic, global_step=epoch, walltime=None)

    # save result
    result_path = os.path.join(out_path, "result.json")
    if not os.path.isfile(result_path):
        f = open(result_path, 'w')
        json.dump({}, f)
        f.close()

    file_ = open(result_path, 'r')
    context = json.load(file_)
    file_.close()

    name = tb_path.split("/")[-1]
    if name not in context.keys():
        context[name] = [{"test_acc": test_acc, "test loss": test_loss}]
    else:
        context[name].append({"test_acc": test_acc, "test loss": test_loss})

    f = open(result_path, 'w')
    json.dump(context, f, indent=4)
    f.close()

    time.sleep(30)
    print("Done")
