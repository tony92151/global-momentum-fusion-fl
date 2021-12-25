import math
import random
from concurrent.futures import as_completed

import torch
from tqdm import tqdm
from utils.configer import Configer
from utils.dataloaders import DATALOADER
from utils.models import MODELS
from client.clients import get_client
from server.servers import get_server
from trainer.trainers import get_trainer
from sparse_compressor.topk_compressor import topkCompressor


class client_manager:
    def __init__(self, config: Configer,
                 warmup_scheduler=None,
                 writer=None,
                 executor=None,
                 available_gpu=None):

        self.config = config
        self.warmup_scheduler = warmup_scheduler
        self.writer = writer
        self.executor = executor
        self.available_gpu = available_gpu

        self.clients = []
        self.server = get_server(con=self.config)

        print("\nInit dataloader...")
        self.dataloaders, emb = DATALOADER(config=config, emd_measurement=True)

        # write earth_moving_distance
        if writer is not None:
            writer.add_scalar("earth_moving_distance", emb, global_step=0, walltime=None)

        self.sampled_client_id = []
        self.init_clients()

    def init_clients(self):
        print("\nInit trainers...")
        print("Nodes: {}".format(self.config.general.get_nodes()))

        client = get_client(self.config)
        trainer = get_trainer(self.config)
        compressor = topkCompressor()

        for i in tqdm(range(self.config.general.get_nodes())):
            if (self.available_gpu is not None) and (not self.available_gpu  == []):
                device = torch.device("cuda:{}".format(self.available_gpu[i%len(self.available_gpu)]))
            else:
                device = torch.device("cpu")

            self.clients.append(
                client(config=self.config,
                       cid=i,
                       compressor=compressor,
                       trainer=trainer, 
                       warmup_scheduler=self.warmup_scheduler,
                       data={
                           "train_dataloader": self.dataloaders['train_s'][i],
                           "test_dataloader": self.dataloaders['test_s'][i] if self.dataloaders[
                                                                                   'test_s'] is not None else
                           self.dataloaders['test'],
                           "test_global_dataloader": self.dataloaders['test']
                       },
                       writer=self.writer,
                       device=device
                       )
            )

    def set_init_mdoel(self):
        print("\nInit model...")
        net = MODELS(self.config)()
        for client in self.clients:
            client.set_model(net)
        return net

    def sample_client(self):
        number_of_client = math.ceil(self.config.general.get_nodes() * self.config.general.get_frac())
        sample_result = random.sample(range(self.config.general.get_nodes()), number_of_client)
        self.sampled_client_id = sorted(sample_result)
        return self.sampled_client_id

    def sample_data(self):
        for client in self.clients:
            if client.cid in self.sampled_client_id:
                client.sample_train_data()

    def set_communication_round(self, communication_round):
        # update client information
        for client in self.clients:
            client.set_communication_round(communication_round)

    def train(self):
        trained_gradients = []
        if self.executor is not None:
            futures = []
            for client in self.clients:
                if client.cid in self.sampled_client_id:
                    futures.append(self.executor.submit(client.train))
                else:
                    futures.append(self.executor.submit(client.test))

            for _ in as_completed(futures):
                pass
            del futures

            for client in self.clients:
                if client.cid in self.sampled_client_id:
                    trained_gradients.append(client.get_gradient())
        else:
            for client in self.clients:
                if client.cid in self.sampled_client_id:
                    client.train()
                else:
                    client.test()

            for client in self.clients:
                if client.cid in self.sampled_client_id:
                    trained_gradients.append(client.get_gradient())

        return trained_gradients

    def global_test(self):
        test_acc, test_loss = self.clients[0].global_test()
        return test_acc, test_loss

    def one_step_update(self, aggregated_gradient):
        for client in self.clients:
            client.one_step_update(aggregated_gradient=aggregated_gradient)

    def aggregate(self, trained_gradients):
        aggregated_gradient = self.server.aggregate(trained_gradients=trained_gradients)
        return aggregated_gradient
