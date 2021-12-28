import torch
import torch.nn as nn
import time
from copy import deepcopy as dcopy
from utils.configer import Configer
from utils.opti import SERVEROPTS, FEDOPTS
from utils.aggregator import set_gradient
from utils.weight_divergence.weight_divergence import weight_divergence_mod

from sparse_compressor.record_SGD import RSGD


class BASE_TRAINER:
    def __init__(self, config=None, cid=None, warmup_scheduler=None, device=torch.device("cpu")):
        self.config: Configer = config
        self.cid = cid
        self.communication_round = None
        self.global_momentum = None
        self.model = None
        self.warmup_scheduler = warmup_scheduler
        self.device = device

        self.last_gradient = None

        if self.config.trainer.get_lossfun() == "crossentropy":
            self.loss_function = nn.CrossEntropyLoss()

        self.verbose = True

    def print_(self, val):
        if self.verbose:
            print(val)

    def set_model(self, model):
        self.model = dcopy(model)

    def train_run(self, model=None, data=None, lr=None):
        raise NotImplementedError

    def test_run(self, data=None):
        raise NotImplementedError

    def test_global_run(self, data=None):
        raise NotImplementedError

    def test(self, data=None):
        raise NotImplementedError

    def one_step_update(self, aggregated_gradient=None, lr=None):
        raise NotImplementedError

    def set_gradient(self, optimizer, uncompressed_aggregate_gradient):
        if uncompressed_aggregate_gradient["compressed"]:
            raise ValueError("In trainer.set_gradient(), got compressed gradient.")
        uag = [uncompressed_aggregate_gradient["gradient"][k] for k in
               uncompressed_aggregate_gradient["gradient"].keys()]

        for group in optimizer.param_groups:
            for p in range(len(group['params'])):
                group['params'][p].grad = dcopy(uag[p]).to(group['params'][p].device)

    # def opt_step_base_model(self, base_gradient=None, round_=None, base_model=None):
    #     self.print_("trainer >> cid: {} >> opt_step, {}".format(self.cid, time.time()))
    #     if base_model is None:
    #         model = copy.deepcopy(self.last_model)
    #     else:
    #         model = copy.deepcopy(base_model)
    #
    #     if round_ is None:
    #         round_ = self.round
    #     lr = self.warmup.get_lr_from_step(round_)
    #
    #     model.to(self.device).train()
    #     optimizer = SERVEROPTS(config=self.config, params=model.parameters(), lr=lr)
    #     if self.last_onestep_state is not None:
    #         optimizer.load_state_dict(self.last_onestep_state)
    #     # optimizer = GFDGCSGD(params=model.parameters(), lr=lr, device=self.device)
    #     base_gradient["gradient"] = [t.to(self.device) for t in base_gradient["gradient"]]
    #     # optimizer.one_step(base_gradient["gradient"])
    #     set_gradient(opt=optimizer, cg=base_gradient["gradient"])
    #     optimizer.step()
    #     # if 'bn' in self.last_gradient.keys():
    #     #     idx = 0
    #     #     for layer in model.cpu().modules():
    #     #         if isinstance(layer, torch.nn.BatchNorm2d):
    #     #             layer.running_mean = torch.tensor(self.last_gradient["bn"][idx]).clone().detach()
    #     #             layer.running_var = torch.tensor(self.last_gradient["bn"][idx + 1]).clone().detach()
    #     #             layer.num_batches_tracked = torch.tensor(self.last_gradient["bn"][idx + 2]).clone().detach()
    #     #             idx += 3
    #     self.last_model = copy.deepcopy(model)
    #     self.last_de_gradient = copy.deepcopy(base_gradient)
    #     # self.last_onestep_state = optimizer.state_dict()
    #     if round_ >= self.config.trainer.get_base_step() - 1:
    #         self.update_global_momentum()
    #     return

    # def update_global_momentum(self):
    #     if self.global_momentum is None and self.last_de_gradient is not None:
    #         self.global_momentum = copy.deepcopy(self.last_de_gradient["gradient"])
    #     else:
    #         for i in range(len(self.global_momentum)):
    #             self.global_momentum[i].mul_(self.config.gf.get_fusion_momentum()).add_(
    #                 self.last_de_gradient["gradient"][i])


