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

from globalfusion.warmup import warmup
from utils.aggregator import aggregater, decompress, compress, parameter_count
from utils.configer import Configer
from utils.dataloaders import cifar_dataloaders, femnist_dataloaders, DATALOADER
from utils.eval import evaluater
from utils.models import MODELS
from utils.trainer import trainer


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


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
    dataloaders = DATALOADER(config)
    if config.trainer.get_dataset_type() == "iid":
        dataloaders["train_s"] = dataloaders["train_s_iid"]
        print("\nUse iid dataloader...")
    else:
        print("\nUse non-iid dataloader...")

    # Init trainersd
    print("\nInit trainers...")
    print("Nodes: {}".format(config.general.get_nodes()))
    trainers = []
    for i in tqdm(range(config.general.get_nodes())):
        trainers.append(trainer(config=config,
                                device=torch.device("cuda:{}".format(gpus[i % len(gpus)])),
                                dataloader=dataloaders["train_s"][i],
                                dataloader_ii=dataloaders["train_s_iid"][i],
                                cid=i,
                                writer=writer,
                                warmup=w))

    print("\nInit model...")
    set_seed(123)
    net = MODELS(config)()

    for tr in trainers:
        tr.set_mdoel(net)

    # Init trainers evaluater
    ev = evaluater(config=config, dataloader=dataloaders["test"], device=torch.device("cuda:{}".format(gpus[0])),
                   writer=None)

    # init traffic simulator (count number of parameters of transmitted gradient)
    traffic = 0
    traffic += (parameter_count(net) * config.general.get_nodes())  # clients download
    # train
    print("\nStart training...")
    if not num_pool == -1:
        executor = ThreadPoolExecutor(max_workers=num_pool)
    for epoch in tqdm(range(config.trainer.get_max_iteration())):
        gs = []
        # sample dataset
        set_seed(epoch)
        for tr in trainers:
            tr.sample_data_from_dataloader()
        ev.sample_data_from_dataloader()

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

            traffic += parameter_count(gs) * 2  # clients upload and aggregator download

            # decompress
            # result = executor.map(decompress, [gs[i]["gradient"] for i in range(len(gs))])
            # result = [i for i in result]
            # for i in range(len(gs)):
            #     gs[i]["gradient"] = result[i]
            for i in range(len(gs)):
                gs[i]["gradient"] = decompress(gs[i]["gradient"])

            # aggregate
            rg = aggregater(gs, aggrete_bn=False)

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
            set_seed(epoch + 1000)
            acc, loss = ev.eval_run(trainers[0].last_model)
            test_acc.append(acc)
            test_loss.append(loss)
            # evl_models = [copy.deepcopy(tr.last_model) for tr in trainers]
            # result = executor.map(ev.eval_run, evl_models)
            # for acc, loss in result:
            #     test_acc.append(acc)
            #     test_loss.append(loss)
            # del evl_models
        ####################################################################################################
        else:
            for i, tr in zip(range(len(trainers)), trainers):
                _ = tr.train_run(round_=epoch)
                gs.append(tr.last_gradient)

            traffic += parameter_count(gs) * 2  # clients upload and aggregator download

            for i in range(len(gs)):
                gs[i]["gradient"] = decompress(gs[i]["gradient"])

            rg = aggregater(gs, device=torch.device("cuda:0"), aggrete_bn=False)

            for tr in trainers:
                tm = tr.opt_step_base_model(round_=epoch, base_gradient=rg)

            test_acc = []
            test_loss = []
            ev.round = epoch
            acc, loss = ev.eval_run(model=trainers[0].last_model)
            test_acc.append(acc)
            test_loss.append(loss)
        ####################################################################################################
        ####################################################################################################
        for tr in trainers:
            tr.wdv_test(round_=epoch, gradients=gs, agg_gradient=rg,
                        compare_with="agg", mask=False, weight_distribution=False, layer_info=False)
            # ["momentum", "agg"]

        test_acc = sum(test_acc) / len(test_acc)
        test_loss = sum(test_loss) / len(test_loss)
        writer.add_scalar("test loss", test_loss, global_step=epoch, walltime=None)
        writer.add_scalar("test acc", test_acc, global_step=epoch, walltime=None)
        writer.add_scalar("traffic(number_of_parameters)", traffic, global_step=epoch, walltime=None)

        # clients download
        traffic += (parameter_count([{"gradient": compress(rg["gradient"])}]) * config.general.get_nodes())

        l = 0.0
        ls = []
        for k in list(trainers[0].weight_divergence.keys()):
            l = [tr.weight_divergence[k].cpu() for tr in trainers]
            l = sum(l) / len(l)
            ls.append(l)
            if False:  # print layer-wise wdv
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

    time.sleep(20)
    print("Done")
