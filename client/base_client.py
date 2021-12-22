import os
import json
import torch
from copy import deepcopy as dcopy


class BASE_CLIENT:
    def __init__(self, config=None, cid=None, compressor=None, trainer=None,
                 data=None, warmup_scheduler=None, writer=None, device=torch.device("cpu")):
        self.config = config
        self.cid = cid
        self.compressor = dcopy(compressor)
        self.trainer = dcopy(trainer)
        self.data = data
        # data = {
        #           "train_dataloader":<dataloader>,
        #           "test_dataloader":<dataloader>,
        #           "test_global_dataloader": <dataloader>,
        # }
        self.sampled_data = None

        # communication round
        self.communication_round = -1

        self.warmup_scheduler = warmup_scheduler
        self.writer = writer
        self.device = device

        if self.cid is None:
            raise ValueError("client id not define.")

        ###########################
        self.last_gradient = None
        self.mount_of_data = sum(1 for _ in self.data['train_dataloader'])
        # since we use subsampler in dataloader, mount of data not equal to len(data.train_dataloader.dataset)
        ###########################

    def set_communication_round(self, communication_round):
        self.communication_round = communication_round
        self.trainer.communication_round = communication_round

    def sample_train_data(self):
        self.sampled_data = []
        for data, target in self.dataloader:
            self.sampled_data.append((data, target))

    def train(self):
        self.trainer.train(dataloader=self.data["train_dataloader"])
        #
        self.last_gradient = None
        raise NotImplementedError()

    def test(self):
        self.trainer.train(dataloader=self.data["test_dataloader"])
        raise NotImplementedError()

    def global_test(self):
        test_acc, test_loss = self.trainer.test(dataloader=self.data["test_global_dataloader"])
        if self.writer is not None:
            self.writer.add_scalar("global_test_acc", test_acc, global_step=self.communication_round, walltime=None)
            self.writer.add_scalar("global_test_loss", test_loss, global_step=self.communication_round, walltime=None)
        return test_acc, test_loss

    def set_model(self, model):
        self.trainer.set_mdoel(model)

    def get_gradient(self):
        return self.last_gradient

    def one_step_update(self, aggregated_gradient=None):
        raise NotImplementedError()
