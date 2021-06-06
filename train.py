import argparse
import copy
import json
import os
import time
from concurrent.futures import as_completed
from bounded_pool_executor import BoundedThreadPoolExecutor as ThreadPoolExecutor

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from globalfusion.warmup import warmup
from utils.aggregator import aggregater, decompress, get_serialize_size
from utils.configer import Configer
from utils.dataloaders import cifar_dataloaders, femnist_dataloaders
from utils.eval import evaluater
from utils.models import *
from utils.trainer import trainer

# Thread out of memory issue.
# https://github.com/dchevell/flask-executor/issues/6

torch.manual_seed(0)


def init_writer(tbpath):
    # tbpath = "/root/notebooks/tensorflow/logs/test"
    os.makedirs(tbpath, exist_ok=True)
    writer = SummaryWriter(tbpath)
    print("$ tensorboard --logdir={} --port 8123 --host 0.0.0.0 \n".format(os.path.dirname(tbpath)))
    print("\nName: {}".format(tbpath.split("/")[-1]))
    return writer


def run(job, arg):
    return job(arg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="path to config", type=str, default=None)
    parser.add_argument('--output', help="output", type=str, default=None)
    parser.add_argument('--pool', help="Multiprocess Worker Pools", type=str, default="-1")
    parser.add_argument('--gpu', help="GPU usage. ex: 0,1,2", type=str, default="0")
    parser.add_argument('--baseline', help="baseline single trainer training,", type=bool, default=False)
    args = parser.parse_args()

    if args.config is None or args.output is None:
        print("Please set --config & --output.")
        exit()

    con_path = os.path.abspath(args.config)
    out_path = os.path.abspath(args.output)
    os.makedirs(out_path, exist_ok=True)

    gpus = [int(i) for i in args.gpu.split(",")]
    if len(gpus) > torch.cuda.device_count() or max(gpus) > torch.cuda.device_count():
        raise ValueError("GPU unavailable.")
    else:
        print("\nGPU usage: {}".format(gpus))

    baseline = args.baseline

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
                                        index_path=os.path.join(config.trainer.get_dataset_path(), "index.json"),
                                        batch_size=config.trainer.get_local_bs())
    elif "femnist" in config.trainer.get_dataset_path():
        dataloaders = femnist_dataloaders(root=config.trainer.get_dataset_path(),
                                          batch_size=config.trainer.get_local_bs(),
                                          clients=config.general.get_nodes())

    print("Total train images: {}".format(len(dataloaders["train"].dataset)))
    print("Total test images: {}".format(len(dataloaders["test"].dataset)))

    # Init trainers
    if not baseline:
        print("\nInit trainers...")
        print("Nodes: {}".format(config.general.get_nodes()))
        trainers = []
        if config.trainer.get_dataset_type() == "niid":
            train_d = dataloaders["train_s"]
            print("\nUse non-iid dataloader...")
        else:
            train_d = dataloaders["train_s_iid"]
            print("\nUse iid dataloader...")
        for i in tqdm(range(config.general.get_nodes())):
            trainers.append(trainer(config=config,
                                    device=torch.device("cuda:{}".format(gpus[i % len(gpus)])),
                                    dataloader=train_d[i],
                                    dataloader_iid=dataloaders["train_s_iid"][i],
                                    cid=i,
                                    writer=writer,
                                    warmup=w))
    else:
        print("\nInit baseline trainer...")
        print("Nodes: {}".format(config.general.get_nodes()))
        trainers = []
        for i in tqdm(range(1)):
            trainers.append(trainer(config=config,
                                    device=torch.device("cuda:{}".format(gpus[i % len(gpus)])),
                                    dataloader=dataloaders["train"],
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
    ev = evaluater(config=config, dataloader=dataloaders["test"], device=torch.device("cuda:{}".format(gpus[0])),
                   writer=None)

    # inint traffic simulator
    traffic = 0
    traffic += get_serialize_size(net) * 4  # 4 clients download
    # train
    print("\nStart training...")
    if not num_pool == -1:
        executor = ThreadPoolExecutor(max_workers=num_pool)
    for epoch in tqdm(range(config.trainer.get_max_iteration())):
        gs = []

        ####################################################################################################
        ####################################################################################################
        if not num_pool == -1:
            # training
            futures = []
            for tr in trainers:
                futures.append(executor.submit(tr.train_run, epoch))
            for future in as_completed(futures):
                pass
            del futures

            for tr in trainers:
                gs.append(tr.last_gradient)

            traffic += get_serialize_size(gs) * 2  # 4 clients upload and aggregator download download

            # decompress
            result = executor.map(decompress, [gs[i]["gradient"] for i in range(len(gs))])
            result = [i for i in result]
            for i in range(len(gs)):
                gs[i]["gradient"] = result[i]

            # aggregate
            rg = aggregater(gs, device=torch.device("cuda:{}".format(gpus[0])), aggrete_bn=False)

            # one-step update
            futures_ = []
            for tr in trainers:
                tr.round = epoch
                futures_.append(executor.submit(tr.opt_step_base_model, rg))
            for future in as_completed(futures_):
                pass
            del futures_

            # test
            test_acc = []
            test_loss = []
            ev.round = epoch
            # executor = ThreadPoolExecutor(max_workers=num_pool)
            evl_models = [copy.deepcopy(tr.last_model) for tr in trainers]
            result = executor.map(ev.eval_run, evl_models)
            for acc, loss in result:
                test_acc.append(acc)
                test_loss.append(loss)
            del evl_models
        ####################################################################################################
        else:
            for i, tr in zip(range(len(trainers)), trainers):
                _ = tr.train_run(round_=epoch)
                gs.append(tr.last_gradient)

            traffic += get_serialize_size(gs) * 2  # 4 clients upload and aggregator download download

            for i in range(len(gs)):
                gs[i]["gradient"] = decompress(gs[i]["gradient"], device=torch.device("cuda:0"))

            rg = aggregater(gs, device=torch.device("cuda:0"), aggrete_bn=False)

            for tr in trainers:
                tm = tr.opt_step_base_model(round_=epoch, base_gradient=rg)

            test_acc = []
            test_loss = []
            for tr in trainers:
                acc, loss = ev.eval_run(model=copy.deepcopy(tr.last_model), round_=epoch)
                test_acc.append(acc)
                test_loss.append(loss)
        ####################################################################################################
        ####################################################################################################
        for tr in trainers:
            tr.wdv_test(round_=epoch, gradients=gs, agg_gradient=rg, compare_with_iid_data=False)

        test_acc = sum(test_acc) / len(test_acc)
        test_loss = sum(test_loss) / len(test_loss)
        writer.add_scalar("test loss", test_loss, global_step=epoch, walltime=None)
        writer.add_scalar("test acc", test_acc, global_step=epoch, walltime=None)
        writer.add_scalar("traffic(MB)", traffic, global_step=epoch, walltime=None)

        traffic += get_serialize_size(rg) * 4  # 4 clients download

        l = 0.0
        ls = []
        for k in list(trainers[0].weight_divergence.keys()):
            l = [tr.weight_divergence[k] for tr in trainers]
            l = sum(l) / len(l)
            ls.append(l)
            writer.add_scalar("wdv client_avg layer {}".format(k), l, global_step=epoch, walltime=None)

        writer.add_scalar("wdv client_avg", sum(ls) / len(ls), global_step=epoch, walltime=None)

    if not num_pool == -1:
        executor.shutdown(True)
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
