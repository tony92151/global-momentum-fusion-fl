import torch
import copy, os, time
from sparse_optimizer.topk_compressor import topkCompressor
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from sparse_optimizer.base_optimizer import BASE_SGD, Memory

class SGCSGD(BASE_SGD):
    def __init__(self, params, cid=-1, lr=None, dgc_momentum=0.9, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, compress_ratio=0.5, fusing_ratio=0.5, checkpoint=False,
                 device=torch.device("cpu"), pool=None):

        super(SGCSGD, self).__init__(params=params, 
                                     lr=lr, 
                                     momentum=momentum, 
                                     dampening=dampening,
                                     weight_decay=weight_decay, 
                                     nesterov=nesterov, 
                                     device=device)

        if lr is None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.memory = SPCMemory(momentum=dgc_momentum, device=device)
        self.compressor = topkCompressor(compress_ratio=compress_ratio, device=device)
        self.device = device
        ######
        self.verbose = False
        ######

    def print_(self, val):
        if self.verbose:
            print(val)

    def compress(self, compress=True):
        # r = self.compressor.compress(self.memory.get_mem(), mome=mome, compress=compress, fusing=fusing)
        # self.memory.set_compressed_mem(r)
        self.print_("optimizer >> compensate, {}".format(time.time()))
        compensated_gradient = self.memory.compensate(self.memory.mem["gradient"])
        self.print_("optimizer >> compress, {}".format(time.time()))
        compress_result = self.compressor.compress(mem=compensated_gradient, compress=compress)
        self.print_("optimizer >> compress done, {}".format(time.time()))
        new_compress_result = self.memory.update(compress_result)

        # bn shouldn't be compressed
        if 'bn' in self.memory.mem.keys():
            for p in range(len(self.memory.mem["bn"])):
                self.memory.mem["bn"][p] = self.memory.mem["bn"][p].tolist()

        self.memory.mem["gradient"] = new_compress_result
        # self.memory.mem["step_count"] = self.step_count
        self.memory.set_compressed_mem(self.memory.mem)

    
class SPCMemory(Memory):
    def __init__(self, momentum=0.9, device=torch.device("cpu")):

        super(SPCMemory, self).__init__(momentum=momentum,
                                        device=device)
    
    def compensate(self, gradient):
        gradient = [i.to(self.device) for i in gradient]
        
        if self.momentums is None:
            self.momentums = copy.deepcopy(gradient)
        else:
            mmt = self.momentums

            m_e = []
            for m, g in zip(mmt, gradient):
                m_ = copy.deepcopy(m).to(self.device)
                m_.add_(g.to(self.device).mul_(self.momentum))
                m_e.append(m_)

            self.momentums = m_e
        return self.momentums

    def update(self, compressed_gradient):
        momentums_tmp = copy.deepcopy(self.momentums)
        momentums_tmp = [torch.tensor(i) for i in momentums_tmp]
        # Momentun Approximation
        new_compressed_gradient = []
        for j, m in zip(compressed_gradient, momentums_tmp):
            # unpack
            selected_mem, ctx = j
            shape, mask, numel = ctx
            # add
            selected_mem = torch.tensor(selected_mem).add(m.flatten()[torch.tensor(mask)>0].cpu()).tolist()
            # pack
            j = selected_mem, ctx
            new_compressed_gradient.append(j)

        # zero the gradients that are selected to transmit
        m_e = []
        for j, m in zip(compressed_gradient, momentums_tmp):
            selected_mem, ctx = j
            shape, mask, numel = ctx
            indices, = torch.where(torch.BoolTensor(mask).to(self.device))
            m_ = copy.deepcopy(m).view(-1).index_fill_(0, indices, 0)
            m_e.append(m_.view(shape).detach())

        self.momentums = copy.deepcopy(m_e)

        return new_compressed_gradient

        
