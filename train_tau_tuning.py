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
from sparse_optimizer.warmup import warmup_scheduler
from utils.configer import Configer
from client_manager import client_manager

from utils.parameter_counter import parameter_count
from sparse_compressor.scheduler import fusion_ratio_scheduler


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


#
# def add_info(epoch, config, warmup):
#     lr = warmup.get_lr_from_step(epoch)
#     chunk = config.trainer.get_max_iteration() / len(config.dgc.get_compress_ratio())
#     chunk_ = config.trainer.get_max_iteration() / len(config.gf.get_fusing_ratio())
#     cr = config.dgc.get_compress_ratio()[min(len(config.dgc.get_compress_ratio()), int(epoch / chunk))]
#     fr = config.gf.get_fusing_ratio()[min(len(config.gf.get_fusing_ratio()), int(epoch / chunk_))]
#     writer.add_scalar("Compress ratio", cr, global_step=epoch, walltime=None)
#     if config.gf.get_global_fusion():
#         writer.add_scalar("Fusion ratio", fr, global_step=epoch, walltime=None)
#     writer.add_scalar("Learning rate", lr, global_step=epoch, walltime=None)

def init_regular_training(config, logdir, executor, seed):
    writer = init_writer(tbpath=os.path.abspath(logdir), name_prefix=args.name_prefix)

    w_scheduler = warmup_scheduler(start_lr=config.trainer.get_start_lr(),
                                   max_lr=config.trainer.get_max_lr(),
                                   min_lr=config.trainer.get_min_lr(),
                                   base_step=config.trainer.get_base_step(),
                                   end_step=config.trainer.get_end_step())

    cm = client_manager(config=config,
                        warmup_scheduler=w_scheduler,
                        writer=writer,
                        executor=executor,
                        available_gpu=gpus)

    set_seed(seed + 1)
    net = cm.set_init_mdoel()

    # init traffic simulator (count number of parameters of transmitted gradient)
    traffic = 0
    traffic += (parameter_count(net) * config.general.get_nodes())  # clients download

    logname = logdir.split("/")[-1]
    record = {logname: {"train": {}, "test": {}, "final": {}}}

    return cm, traffic, record


def regular_training(cm, communication_round, traffic, record):
    cm.set_communication_round(communication_round=communication_round)
    # seed
    set_seed(args.seed + communication_round)
    # sample trainer
    sampled_client_id = cm.sample_client()
    # sample dataset
    cm.sample_data()
    ####################################################################################################
    trained_gradients = cm.train()
    # clients transmit to server
    traffic += sum([parameter_count(g) for g in trained_gradients])
    # aggregate
    aggregated_gradient = cm.aggregate(trained_gradients=trained_gradients)
    # server transmit to clients
    traffic += parameter_count(aggregated_gradient) * config.general.get_nodes()
    # one step update
    cm.one_step_update(aggregated_gradient=aggregated_gradient)
    one_step_done_time = time.time()
    ####################################################################################################

    test_acc, test_loss = cm.global_test()
    print("Test acc: {}, loss: {}".format(test_acc, test_loss))
    global_test_done_time = time.time()

    # for cid in sampled_client_id:
    #     client_sampled_count[cid] += 1

    # record[logname]["train"][communication_round] = client_manager.train_result
    # record[logname]["test"][communication_round] = client_manager.test_result

    return traffic


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
        exit(1)

    dqn_round = 10

    con_path = os.path.abspath(args.config)
    out_path = os.path.abspath(args.output)
    os.makedirs(out_path, exist_ok=True)

    gpus = [int(i) for i in args.gpu.split(",")]
    if len(gpus) > torch.cuda.device_count() or max(gpus) > torch.cuda.device_count():
        raise ValueError("GPU unavailable.")
    else:
        print("\nGPU usage: {}".format(gpus))

    print("Read config at : {}".format(con_path))
    config = Configer(con_path)

    num_pool = args.pool
    if num_pool == "auto":
        num_pool = int(config.general.get_nodes() / 2)
        print("\nPool: {}".format(num_pool))
    else:
        num_pool = int(num_pool)
        print("\nPool: {}".format(num_pool))

    if not num_pool == -1:
        executor = ThreadPoolExecutor(max_workers=num_pool)
    else:
        executor = None

    logdir = args.tensorboard_path if args.tensorboard_path is not None else config.general.get_logdir()

    # train
    print("\nStart training...")

    ########
    # DQN
    ########
    # agent = ReinforceAgent(root="./agent_cache")

    for di in range(dqn_round):
        print("=" * 10, " DQN round ({}/{}) : Regular training ".format(di, dqn_round), "=" * 10)
        #### Regular training ####

        # step1: init new client_manager, communication_round, traffic and record
        logdir = os.path.join(args.tensorboard_path, "DQN_round_{}_of_{}_regular".format(di, dqn_round))
        cm, traffic, record = init_regular_training(config,
                                                    logdir,
                                                    executor,
                                                    int(time.time() * 1000) % 1000)
        # cache = {
        #     "client_manager": client_manager.clients,
        #     "communication_round": 1,
        #     "traffic": traffic,
        #     "record": record,
        # }
        # os.makedirs("./agent_cache", exist_ok=True)
        # torch.save(cache, os.path.join("./agent_cache",  "DQN_round_{}_of_{}_regular".format(di, dqn_round)))
        for communication_round in tqdm(range(0, config.trainer.get_max_iteration())):
            # overwrite fusion_ratio_scheduler
            qvalue = 1  # agent.getQvalue()
            for c in cm.clients:
                c.fusion_ratio_scheduler = fusion_ratio_scheduler(1, [qvalue])

            #traffic = regular_training(cm, communication_round, traffic, record)

            if communication_round % 10 == 0 and communication_round != 0:
                # agent append memory
                pass

        # print("=" * 10, " DQN round ({}/{}) : Cache training ".format(di, dqn_round), "=" * 10)
        # #### Cache training ####
        # step1 : random select a cache in cache_queue
        # cache = {
        #     "cm": None,
        #     "communication_round": None,
        #     "traffic": None,
        #     "record": None,
        # }

        # # step2 : Resume
        # cm = cache["cm"]
        # last_cm = cache["communication_round"]
        # last_traffic = cache["traffic"]
        # last_record = cache["record"]

    if not num_pool == -1:
        executor.shutdown(True)

    time.sleep(5)
    print("Done")
