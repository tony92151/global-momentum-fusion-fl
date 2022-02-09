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
        exit(1)

    time_analyze = True

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

    logdir = args.tensorboard_path if args.tensorboard_path is not None else config.general.get_logdir()

    writer = init_writer(tbpath=os.path.abspath(logdir), name_prefix=args.name_prefix)

    w_scheduler = warmup_scheduler(start_lr=config.trainer.get_start_lr(),
                                   max_lr=config.trainer.get_max_lr(),
                                   min_lr=config.trainer.get_min_lr(),
                                   base_step=config.trainer.get_base_step(),
                                   end_step=config.trainer.get_end_step())

    if not num_pool == -1:
        executor = ThreadPoolExecutor(max_workers=num_pool)
    else:
        executor = None

    client_manager = client_manager(config=config,
                                    warmup_scheduler=w_scheduler,
                                    writer=writer,
                                    executor=executor, 
                                    available_gpu=gpus)

    set_seed(args.seed + 1)
    net = client_manager.set_init_mdoel()

    # init traffic simulator (count number of parameters of transmitted gradient)
    traffic = 0
    traffic += (parameter_count(net) * config.general.get_nodes())  # clients download
    full_size = parameter_count(net)

    client_smapled_count = [0 for i in range(config.general.get_nodes())]

    logname = logdir.split("/")[-1]
    record = {logname:{"train":{}, "test":{}, "final":{}}}
    # train
    print("\nStart training...")
    for communication_round in tqdm(range(config.trainer.get_max_iteration())):
        start_time = time.time()
        client_manager.set_communication_round(communication_round=communication_round)

        # seed
        set_seed(args.seed + communication_round)

        # sample trainer
        sampled_client_id = client_manager.sample_client()
        
        # sample dataset
        client_manager.sample_data()
        sample_done_time = time.time()

        ####################################################################################################
        trained_gradients = client_manager.train()
        train_done_time = time.time()

        # clients transmit to server
        traffic += sum([parameter_count(g) for g in trained_gradients])

        # aggregate
        aggregated_gradient = client_manager.aggregate(trained_gradients=trained_gradients)
        aggregate_done_time = time.time()

        # server transmit to clients
        traffic += parameter_count(aggregated_gradient) * config.general.get_nodes()

        # one step update
        client_manager.one_step_update(aggregated_gradient=aggregated_gradient)
        one_step_done_time = time.time()
        ####################################################################################################

        test_acc, test_loss = client_manager.global_test()
        print("Test acc: {}, loss: {}".format(test_acc, test_loss))
        global_test_done_time = time.time()
        # writer.add_scalar("test loss", test_loss, global_step=communication_round, walltime=None)
        # writer.add_scalar("test acc", test_acc, global_step=communication_round, walltime=None)
        writer.add_scalar("traffic(number_of_parameters)", traffic, global_step=communication_round, walltime=None)

        # Time consume analyze:

        if time_analyze:
            print("> Total take: {:.2f} in round {}".format(global_test_done_time-start_time, communication_round))
            print("> Sample data take: {:.2f} ({:.2f}%)".format(sample_done_time-start_time, 100*((sample_done_time-start_time)/(global_test_done_time-start_time))))
            print("> Train take: {:.2f} ({:.2f}%)".format(train_done_time-sample_done_time, 100*((train_done_time-sample_done_time)/(global_test_done_time-start_time))))
            print("> Aggregate take: {:.2f} ({:.2f}%)".format(aggregate_done_time-train_done_time, 100*((aggregate_done_time-train_done_time)/(global_test_done_time-start_time))))
            print("> One step take: {:.2f} ({:.2f}%)".format(one_step_done_time-aggregate_done_time, 100*((one_step_done_time-aggregate_done_time)/(global_test_done_time-start_time))))
            print("> Global test take: {:.2f} ({:.2f}%)".format(global_test_done_time-one_step_done_time, 100*((global_test_done_time-one_step_done_time)/(global_test_done_time-start_time))))

        for cid in sampled_client_id:
            client_smapled_count[cid] += 1

        record[logname]["train"][communication_round] = client_manager.train_result
        record[logname]["test"][communication_round] = client_manager.test_result

    print(client_smapled_count)

    if not num_pool == -1:
        executor.shutdown(True)

    # save result
    result_path = os.path.join(out_path, "result_{}.json".format(int(time.time())))
    
    record[logname]["final"] = {"test_acc": test_acc, "test loss": test_loss, "client_smapled_count": client_smapled_count}

    f = open(result_path, 'w')
    json.dump(record, f, indent=4)
    f.close()

    time.sleep(5)
    print("Done")
