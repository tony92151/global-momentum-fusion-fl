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

from globalfusion.warmup import warmup
from torch.utils.tensorboard import SummaryWriter
from utils.dataloaders import cifar_dataloaders
from utils.trainer import trainer
from utils.aggregator import aggrete, decompress, get_serialize_size
from utils.eval import evaluater
from utils.models import ResNet101_cifar, ResNet50_cifar, ResNet18_cifar, Net


def init_writer(tbpath):
    # tbpath = "/root/notebooks/tensorflow/logs/test"
    os.makedirs(tbpath, exist_ok=True)
    writer = SummaryWriter(tbpath)
    print("$ tensorboard --logdir={} --port 8123 --host 0.0.0.0 \n".format(os.path.dirname(tbpath)))
    return writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config", type=str, default=None)
    parser.add_argument('--output', help="output", type=str, default=None)
    args = parser.parse_args()

    if args.config is None or args.output is None:
        print("Please set --config & --output.")
        exit()

    con_path = os.path.abspath(args.config)
    out_path = os.path.abspath(args.output)
    os.makedirs(out_path, exist_ok=True)

    print("Read cinfig at : {}".format(con_path))
    config = Configer(con_path)

    tb_path = os.path.join(config.general.get_tbpath(), str(int(time.time()))) if config.general.get_tbpath() == "./tblogs" \
        else config.general.get_tbpath()

    writer = init_writer(tbpath=os.path.abspath(tb_path))

    w = warmup(start_lr=config.trainer.get_start_lr(),
               max_lr=config.trainer.get_max_lr(),
               min_lr=config.trainer.get_min_lr(),
               base_step=config.trainer.get_base_step(),
               end_step=config.trainer.get_end_step())

    if "cifar10" in config.trainer.get_dataset_path():
        dataloaders = cifar_dataloaders(root=config.trainer.get_dataset_path(),
                                        index_path=os.path.join(config.trainer.get_dataset_path(),
                                                                config.trainer.get_dataset_type(), "index.json"))

    # Init trainers
    trainers = []
    for i in range(config.general.get_nodes()):
        trainers.append(trainer(config=config,
                                device=torch.device("cuda:0"),
                                dataloader=dataloaders["train"][i],
                                cid=i,
                                writer=writer,
                                warmup=w))

    # net = Net()
    model_table={
        "resnet18": ResNet18_cifar,
        "resnet50": ResNet50_cifar,
        "resnet101": ResNet101_cifar,
        "small": Net,
    }
    net = model_table[config.trainer.get_model()]()

    for tr in trainers:
        tr.set_mdoel(net)

    # Init trainers evaluater
    ev = evaluater(config=config, dataloader=dataloaders["test"], device=torch.device("cuda:0"), writer=None)

    # inint traffic simulator
    traffic = 0
    traffic += get_serialize_size(net) * 4  # 4 clients download
    # train
    for epoch in tqdm(range(config.trainer.get_max_iteration())):
        gs = []
        for i, tr in zip(range(len(trainers)), trainers):
            _ = tr.train_run(round_=epoch)
            gs.append(tr.last_gradient)
            # writer.add_scalar("loss of {}".format(i), tr.training_loss, global_step=epoch, walltime=None)

        for i in range(len(gs)):
            gs[i]["gradient"] = decompress(gs[i]["gradient"], device=torch.device("cuda:0"))

        rg = aggrete(gs, device=torch.device("cuda:0"), aggrete_bn=False)

        for tr in trainers:
            tm = tr.opt_step_base_model(round_=epoch, base_gradient=rg)

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
    if os.path.isfile(result_path):
        f = open(result_path, 'w')
        json.dump({}, f)
        f.close()

    file_ = open(result_path, 'r')
    context = json.load(file_)
    file_.close()

    name = config.general.get_tbpath().split("/")[-1]
    if name not in context.keys():
        context[name] = [{"test_acc": test_acc, "test loss": test_loss}]
    else:
        context[name].append({"test_acc": test_acc, "test loss": test_loss})

    f = open(result_path, 'w')
    json.dump(context, f)
    f.close()

    time.sleep(30)
    print("Done")
