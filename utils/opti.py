from globalfusion.optimizer import GFDGCSGD as GFDGCSGD_
from torch.optim import Adagrad
from torch.optim import SGD
from torch.optim import Adam
import torch
from copy import deepcopy as dcopy
from torch_optimizer import Yogi

SERVEROPT = {"SGD": SGD,
             "ADAGRAD": Adagrad,
             "ADAM": Adam,
             "YOGI": Yogi}

#
# class SGD(SGD_):
#     def set_gradient(self, cg):
#         agged_grad = cg
#         for group in self.param_groups:
#             for p in range(len(group['params'])):
#                 group['params'][p].grad = dcopy(agged_grad[p]).to(group['params'][p].device)
#
#
# class Adam(Adam_):
#     def set_gradient(self, cg):
#         agged_grad = cg
#         for group in self.param_groups:
#             for p in range(len(group['params'])):
#                 group['params'][p].grad = dcopy(agged_grad[p]).to(group['params'][p].device)
#
#
# class Adagrad(Adagrad_):
#     def set_gradient(self, cg):
#         agged_grad = cg
#         for group in self.param_groups:
#             for p in range(len(group['params'])):
#                 group['params'][p].grad = dcopy(agged_grad[p]).to(group['params'][p].device)
#
#
# class Adagrad(Adagrad_):
#     def set_gradient(self, cg):
#         agged_grad = cg
#         for group in self.param_groups:
#             for p in range(len(group['params'])):
#                 group['params'][p].grad = dcopy(agged_grad[p]).to(group['params'][p].device)
