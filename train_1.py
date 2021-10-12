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
from utils.eval import evaluater, lstm_evaluater
from utils.models import MODELS
from utils.trainer import trainer, lstm_trainer

from client_manager import client_manager
from server import server


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def init_writer(tbpath, name_prefix=""):
    # tbpath = "/root/notebooks/tensorflow/logs/test"
    tbpath = os.path.join(os.path.dirname(tbpath), name_prefix + tbpath.split("/")[-1])
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
    parser.add_argument('--name_prefix', help="name_prefix", type=str, default="")
    parser.add_argument('--seed', help="seed", type=int, default=123)

    parser.add_argument('--tensorboard_path', type=str, default=None)
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

    tb_path = args.tensorboard_path if args.tensorboard_path is not None else config.general.get_tbpath()

    writer = init_writer(tbpath=os.path.abspath(tb_path), name_prefix=args.name_prefix)

    w = warmup(start_lr=config.trainer.get_start_lr(),
               max_lr=config.trainer.get_max_lr(),
               min_lr=config.trainer.get_min_lr(),
               base_step=config.trainer.get_base_step(),
               end_step=config.trainer.get_end_step())

    if not num_pool == -1:
        executor = ThreadPoolExecutor(max_workers=num_pool)
    else:
        executor = None

    client_manager = client_manager(config=config,
                                    gpus=gpus,
                                    warmup_scheduler=w,
                                    writer=writer,
                                    executor=executor)
    server = server(config=config, device=torch.device("cuda:{}".format(gpus[0])))
    set_seed(args.seed+1)
    net = client_manager.set_init_mdoel()

    # init traffic simulator (count number of parameters of transmitted gradient)
    traffic = 0
    traffic += (parameter_count(net) * config.general.get_nodes())  # clients download

    client_smapled_count = [0 for i in range(config.general.get_nodes())]

    # train
    print("\nStart training...")
    for epoch in tqdm(range(config.trainer.get_max_iteration())):
        # sample trainer
        set_seed(args.seed+epoch)
        sample_trainer_cid = random.sample(range(config.general.get_nodes()),
                                           round(config.general.get_nodes() * config.trainer.get_frac()))
        sample_trainer_cid = sorted(sample_trainer_cid)

        # sample dataset
        set_seed(args.seed+epoch+1)
        client_manager.set_sampled_trainer(sample_trainer_cid)
        client_manager.sample_data()

        add_info(epoch, config, w)
        ####################################################################################################
        gs = client_manager.training(epoch=epoch)

        # clients transmit to aggregator(server)
        traffic += parameter_count(gs)

        for i in range(len(gs)):
            gs[i]["gradient"] = decompress(gs[i]["gradient"])

        # aggregate
        aggregated_gradient = server.aggregate(gs, aggrete_bn=False)

        # one set update
        client_manager.opt_one_step(epoch, aggregated_gradient)

        # clients download
        traffic += (parameter_count(
            [{"gradient": compress(aggregated_gradient["gradient"])}]) * config.general.get_nodes())
        ####################################################################################################

        test_acc, test_loss = client_manager.evaluater.eval_run(model=client_manager.trainers[0].last_model)
        writer.add_scalar("test loss", test_loss, global_step=epoch, walltime=None)
        writer.add_scalar("test acc", test_acc, global_step=epoch, walltime=None)
        writer.add_scalar("traffic(number_of_parameters)", traffic, global_step=epoch, walltime=None)

        for cid in sample_trainer_cid:
            client_smapled_count[cid] += 1

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
