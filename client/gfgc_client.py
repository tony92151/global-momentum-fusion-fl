from abc import ABC

import torch

from sparse_compressor.topk_compressor import topkCompressor
from client.base_client import BASE_CLIENT

from copy import deepcopy as dcopy

from utils.configer import Configer
from sparse_compressor.scheduler import warmup_scheduler, compress_rate_scheduler, fusion_ratio_scheduler


class gfgc_client(BASE_CLIENT):
    def __init__(self, config: Configer, cid=None, compressor=None, trainer=None,
                 data=None, warmup_scheduler=None, writer=None, device=torch.device("cpu")):
        super(gfgc_client, self).__init__(config=config, cid=cid, compressor=compressor,
                                          trainer=trainer, data=data, warmup_scheduler=warmup_scheduler,
                                          writer=writer, device=device)
        self.memory = gfgc_memory(gfgc_momentum=self.config.dgc.get_momentum(),
                                  device=self.device)

        self.warmup_scheduler = warmup_scheduler
        self.compress_rate_scheduler = compress_rate_scheduler(max_iteration=config.trainer.get_max_iteration(),
                                                               compress_rate_list=config.dgc.get_compress_rate())
        # global fusion
        self.global_gradient = None
        self.global_momentum = self.config.gfdgc.get_global_momentum()

    def train(self):
        self.loginfo()

        # train
        lr = self.warmup_scheduler.get_lr_from_step(self.communication_round)
        train_acc, train_loss = self.trainer.train_run(data=self.sampled_data, lr=lr)
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
        fusion_ratio = self.fusion_ratio_scheduler.get_fusion_ratio_from_step(self.communication_round)
        # if global_gradient is None, it skip fusion technique and return  compressor.compress() result
        compressed_compensate_gradient = self.compressor.gf_compress(gradient_dict=compensate_gradient,
                                                                     global_gradient_dict=self.global_gradient,
                                                                     fusion_ratio=fusion_ratio,
                                                                     compress=True)

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
            self.writer.add_scalar("Compress ratio", cr, global_step=self.communication_round, walltime=None)
            self.writer.add_scalar("Learning rate", lr, global_step=self.communication_round, walltime=None)


class gfgc_memory:
    def __init__(self, gfgc_momentum=None, device=torch.device("cpu")):
        self.gfgc_momentum = gfgc_momentum
        self.momentums = None
        self.device = device

    def compensate(self, gradient):
        if gradient["compressed"]:
            raise ValueError("GFGC compensate expect input un-compressed gradient.")

        copy_gradient = dcopy(gradient)
        # for k in copy_gradient['gradient'].keys():
        #     copy_gradient['gradient'][k].to(self.device)

        if self.momentums is None:
            self.momentums = dcopy(copy_gradient)
            for k in self.momentums['gradient'].keys():
                self.momentums['gradient'][k].to(self.device)
        else:
            for k in copy_gradient['gradient'].keys():
                self.momentums["gradient"][k].mul_(self.gfgc_momentum).add_(copy_gradient['gradient'][k].to(self.device))

        return self.momentums

    def update(self, compressed_gradient=None):
        if not compressed_gradient["compressed"]:
            raise ValueError("GFGC update expect input compressed gradient.")

        for k in compressed_gradient['gradient'].keys():
            new_mem, ctx = compressed_gradient['gradient'][k]
            shape, mask, numel = ctx

            indices, = torch.where(torch.BoolTensor(mask).to(self.device))
            self.momentums["gradient"][k] = \
                dcopy(self.momentums["gradient"][k]).view(-1).index_fill_(0, indices, 0).view(shape).detach()
