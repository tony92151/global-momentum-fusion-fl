from abc import ABC

import torch

from sparse_compressor.topk_compressor import topkCompressor
from client.base_client import BASE_CLIENT

from copy import deepcopy as dcopy
import time
from utils.configer import Configer
from sparse_compressor.scheduler import warmup_scheduler, compress_rate_scheduler, fusion_ratio_scheduler


class sgc_client(BASE_CLIENT):
    def __init__(self, config: Configer, cid=None, compressor=None, trainer=None,
                 data=None, warmup_scheduler=None, writer=None, device=torch.device("cpu")):
        super(sgc_client, self).__init__(config=config, cid=cid, compressor=compressor,
                                         trainer=trainer, data=data, warmup_scheduler=warmup_scheduler,
                                         writer=writer, device=device)
        self.memory = sgc_memory(sgc_momentum=self.config.sgc.get_local_momentum(),
                                 device=self.device)

        self.warmup_scheduler = warmup_scheduler
        self.compress_rate_scheduler = compress_rate_scheduler(max_iteration=config.trainer.get_max_iteration(),
                                                               compress_rate_list=config.sgc.get_compress_rate())


    def train(self):
        self.loginfo()
        # local update
        train_model = self.sgc_local_update()

        # train
        lr = self.warmup_scheduler.get_lr_from_step(self.communication_round)
        train_acc, train_loss = self.trainer.train_run(model=train_model, data=self.sampled_data, lr=lr)
        self.sampled_data = None

        if self.writer is not None:
            self.writer.add_scalar("train_acc of {}".format(self.cid), train_acc, global_step=self.communication_round,
                                   walltime=None)
            self.writer.add_scalar("train_loss of {}".format(self.cid), train_loss,
                                   global_step=self.communication_round, walltime=None)

        # compensate
        compensate_gradient = self.memory.compensate(self.trainer.last_gradient)

        # compressed
        self.compressor.set_compress_rate(
            self.compress_rate_scheduler.get_compress_rate_from_step(self.communication_round))
        compressed_compensate_gradient = self.compressor.compress(gradient_dict=compensate_gradient, compress=True)

        # update
        self.memory.update(compressed_compensate_gradient)

        compressed_compensate_gradient["step_count"] = self.step_count
        self.last_gradient = compressed_compensate_gradient

    def test(self):
        self.loginfo()
        test_acc, test_loss = self.trainer.test_run(data=self.data['test_dataloader'])
        if self.writer is not None:
            self.writer.add_scalar("test_acc of {}".format(self.cid), test_acc, global_step=self.communication_round,
                                   walltime=None)
            self.writer.add_scalar("test_loss of {}".format(self.cid), test_loss, global_step=self.communication_round,
                                   walltime=None)

    def one_step_update(self, aggregated_gradient=None):
        if aggregated_gradient["compressed"]:
            aggregated_gradient = self.compressor.decompress(aggregated_gradient)

        lr = self.warmup_scheduler.get_lr_from_step(self.communication_round)
        self.trainer.one_step_update(aggregated_gradient=aggregated_gradient, lr=lr)

    def loginfo(self):
        if self.cid == 0 and self.writer is not None:
            cr = self.compress_rate_scheduler.get_compress_rate_from_step(self.communication_round)
            lr = self.warmup_scheduler.get_lr_from_step(self.communication_round)
            self.writer.add_scalar("Compress rate", cr, global_step=self.communication_round, walltime=None)
            self.writer.add_scalar("Learning rate", lr, global_step=self.communication_round, walltime=None)

    def sgc_local_update(self):
        if self.memory.momentums is None:
            return None
        print("trainer >> cid: {} >> sgc_local_update, {}".format(self.cid, time.time()))
        model = dcopy(self.trainer.model)
        model.to(self.device).train()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=1)
        self.trainer.set_gradient(optimizer=optimizer, uncompressed_aggregate_gradient=self.memory.momentums)
        optimizer.step()
        return model
        # self.trainer.model = dcopy(model)
        # return None



class sgc_memory:
    def __init__(self, sgc_momentum=None, device=torch.device("cpu")):
        self.sgc_momentum = sgc_momentum
        self.momentums = None
        self.device = device

    def compensate(self, gradient):
        if gradient["compressed"]:
            raise ValueError("SGC compensate expect input un-compressed gradient.")

        copy_gradient = dcopy(gradient)
        for k in copy_gradient['gradient'].keys():
            copy_gradient['gradient'][k].to(self.device)

        if self.momentums is None:
            self.momentums = dcopy(copy_gradient)
            for k in self.momentums['gradient'].keys():
                self.momentums['gradient'][k].to(self.device)
        else:
            for k in copy_gradient['gradient'].keys():
                self.momentums["gradient"][k].add_(copy_gradient['gradient'][k].to(self.device))
                # self.momentums["gradient"][k].add_(copy_gradient['gradient'][k].to(self.device).mul_(self.sgc_momentum))

        return self.momentums

    def update(self, compressed_gradient=None):
        if not compressed_gradient["compressed"]:
            raise ValueError("SGC update expect input compressed gradient.")

        for k in compressed_gradient['gradient'].keys():
            new_mem, ctx = compressed_gradient['gradient'][k]
            shape, mask, numel = ctx

            indices, = torch.where(torch.BoolTensor(mask).to(self.device))
            self.momentums["gradient"][k] = \
                dcopy(self.momentums["gradient"][k]).view(-1).index_fill_(0, indices, 0).view(shape).detach()
