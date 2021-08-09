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


def add_info(epoch, config, warmup):
    lr = warmup.get_lr_from_step(epoch)
    chunk = config.trainer.get_max_iteration() / len(config.dgc.get_compress_ratio())
    chunk_ = config.trainer.get_max_iteration() / len(config.gf.get_fusing_ratio())
    cr = config.dgc.get_compress_ratio()[min(len(config.dgc.get_compress_ratio()), int(epoch / chunk))]
    fr = config.gf.get_fusing_ratio()[min(len(config.gf.get_fusing_ratio()), int(epoch / chunk_))]
    writer.add_scalar("Compress ratio", cr, global_step=epoch, walltime=None)
    if config.gf.get_global_fusion():
        writer.add_scalar("Fusion ratio", fr, global_step=epoch, walltime=None)
    writer.add_scalar("Learning rate", lr, global_step=epoch, walltime=None)


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
                                dataloader_iid=dataloaders["train_s_iid"][i],
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

    client_smapled_count = [0 for i in range(config.general.get_nodes())]

    # weight divergence record
    weight_divergence_each_round = []
    # train
    print("\nStart training...")
    if not num_pool == -1:
        executor = ThreadPoolExecutor(max_workers=num_pool)
    for epoch in tqdm(range(config.trainer.get_max_iteration())):
        gs = []
        # sample trainer
        set_seed(epoch + 1)
        sample_trainer_cid = random.sample(range(config.general.get_nodes()),
                                           round(config.general.get_nodes() * config.trainer.get_frac()))
        sample_trainer_cid = sorted(sample_trainer_cid)
        # sample dataset
        set_seed(epoch)
        for tr in trainers:
            if tr.cid in sample_trainer_cid:
                tr.sample_data_from_dataloader()
        ev.sample_data_from_dataloader()

        add_info(epoch, config, w)
        ####################################################################################################
        ####################################################################################################
        if not num_pool == -1:
            # training
            futures = []
            for tr in trainers:
                if tr.cid in sample_trainer_cid:
                    futures.append(executor.submit(tr.train_run, epoch))
                else:
                    futures.append(executor.submit(tr.eval_run, epoch))

            for future in as_completed(futures):
                pass
            del futures

            for tr in trainers:
                if tr.cid in sample_trainer_cid:
                    gs.append(tr.last_gradient)

            traffic += parameter_count(gs)  # clients transmit to aggregator(server)

            for i in range(len(gs)):
                gs[i]["gradient"] = decompress(gs[i]["gradient"])

            # aggregate
            aggregated_gradient = aggregater(gs, aggrete_bn=False)

            # calculate the weight_divergence (one-step update will overwrite the base_model, so do it first)
            weight_divergence_list = []
            for tr in trainers:
                if tr.cid in sample_trainer_cid:
                    result = tr.weight_divergence_test(round_=epoch,
                                                       aggregated_gradient=aggregated_gradient)
                    weight_divergence_list.append(result)
            weight_divergence_avg = sum(weight_divergence_list) / len(weight_divergence_list)

            # one-step update
            futures_ = []
            for tr in trainers:
                tr.round = epoch
                futures_.append(executor.submit(tr.opt_step_base_model, aggregated_gradient))
            for future in as_completed(futures_):
                pass
            del futures_

        ####################################################################################################
        else:
            for i, tr in zip(range(len(trainers)), trainers):
                _ = tr.train_run(round_=epoch)
                gs.append(tr.last_gradient)

            traffic += parameter_count(gs)  # clients upload and aggregator download

            for i in range(len(gs)):
                gs[i]["gradient"] = decompress(gs[i]["gradient"])

            aggregated_gradient = aggregater(gs, device=torch.device("cuda:0"), aggrete_bn=False)

            # calculate the weight_divergence (one-step update will overwrite the base_model, so do it first)
            weight_divergence_list = []
            for tr in trainers:
                if tr.cid in sample_trainer_cid:
                    result = tr.weight_divergence_test(epoch, aggregated_gradient=aggregated_gradient)
                    weight_divergence_list.append(result)
            weight_divergence_avg = sum(weight_divergence_list) / len(weight_divergence_list)

            for tr in trainers:
                tm = tr.opt_step_base_model(round_=epoch, base_gradient=aggregated_gradient)

        ####################################################################################################
        ####################################################################################################
        # for tr in trainers:
        #     if tr.cid in sample_trainer_cid:
        #         tr.weight_divergence_test(epoch,
        #                                   aggregated_gradient=aggregated_gradient,
        #                                   trainer_gradient=None,
        #                                   base_model=None)

        # clients download
        traffic += (parameter_count(
            [{"gradient": compress(aggregated_gradient["gradient"])}]) * config.general.get_nodes())

        test_acc, test_loss = ev.eval_run(model=trainers[0].last_model)
        writer.add_scalar("test loss", test_loss, global_step=epoch, walltime=None)
        writer.add_scalar("test acc", test_acc, global_step=epoch, walltime=None)
        writer.add_scalar("traffic(number_of_parameters)", traffic, global_step=epoch, walltime=None)

        writer.add_scalar("weight_divergence_avg", weight_divergence_avg, global_step=epoch, walltime=None)
        weight_divergence_each_round.append(weight_divergence_avg)

        for cid in sample_trainer_cid:
            client_smapled_count[cid] += 1

    weight_divergence_all_rounds_avg = sum(weight_divergence_each_round)/len(weight_divergence_each_round)
    writer.add_scalar("weight_divergence_all_rounds_avg", weight_divergence_all_rounds_avg, global_step=0, walltime=None)

    print(client_smapled_count)
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
        context[name] = [{"test_acc": test_acc, "test loss": test_loss, "client_smapled_count": client_smapled_count}]
    else:
        context[name].append(
            {"test_acc": test_acc, "test loss": test_loss, "client_smapled_count": client_smapled_count})

    f = open(result_path, 'w')
    json.dump(context, f, indent=4)
    f.close()

    time.sleep(20)
    print("Done")
