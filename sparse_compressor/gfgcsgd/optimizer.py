import torch
import time
import copy
from sparse_compressor.gfcompressor import GFCCompressor
from sparse_compressor.base_optimizer import BASE_SGD, Memory


class GFGCSGD(BASE_SGD):
    def __init__(self, params, lr=None, dgc_momentum=0.9, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, compress_ratio=0.5, fusing_ratio=0.5,
                 device=torch.device("cpu")):

        super(GFGCSGD, self).__init__(params=params,
                                      lr=lr,
                                      momentum=momentum,
                                      dampening=dampening,
                                      weight_decay=weight_decay,
                                      nesterov=nesterov,
                                      device=device)

        self.device = device
        self.memory = GFGCMemory(momentum=dgc_momentum, device=device)
        self.compressor = GFCCompressor(compress_ratio=compress_ratio, fusing_ratio=fusing_ratio, device=device)

        self.global_momentum = None
        self.last_de_gradient = None
        ######
        self.verbose = False
        ######

    def print_(self, val):
        if self.verbose:
            print(val)

    def compress(self, global_momentum=None, compress: bool = True):
        self.print_("optimizer >> compensate, {}".format(time.time()))
        compensated_gradient = self.memory.compensate(self.memory.mem["gradient"])
        self.print_("optimizer >> compress, {}".format(time.time()))
        compress_result = self.compressor.compress(mem=compensated_gradient, gmome=global_momentum,
                                                   compress=compress)
        self.print_("optimizer >> compress done, {}".format(time.time()))
        self.memory.update(compress_result)

        # bn shouldn't be compressed
        if 'bn' in self.memory.mem.keys():
            for p in range(len(self.memory.mem["bn"])):
                self.memory.mem["bn"][p] = self.memory.mem["bn"][p].tolist()

        self.memory.mem["gradient"] = compress_result
        # self.memory.mem["step_count"] = self.step_count
        self.memory.set_compressed_mem(self.memory.mem)


class GFGCMemory(Memory):
    def __init__(self, momentum=0.9, device=torch.device("cpu")):
        super(GFGCMemory, self).__init__(momentum=momentum,
                                         device=device)

    def compensate(self, gradient):
        avg_gradient = [i.to(self.device) for i in gradient]

        if self.momentums is None:
            self.momentums = copy.deepcopy(avg_gradient)
        else:
            mmt = self.momentums

            m_e = []

            for m, g in zip(mmt, avg_gradient):
                m_ = copy.deepcopy(m).to(self.device)
                m_.mul_(self.momentum).add_(g.to(self.device))
                m_e.append(m_)
            self.momentums = m_e

        return self.momentums

    def update(self, com_gradient):
        m_n = copy.deepcopy(self.momentums)
        m_n = [i.to(self.device) for i in m_n]
        # com_gradients = copy.deepcopy(self.velocities)
        m_e = []
        for j, m in zip(com_gradient, m_n):
            new_mem, ctx = j
            shape, mask, numel = ctx
            indices, = torch.where(torch.BoolTensor(mask).to(self.device))
            m_ = copy.deepcopy(m).view(-1).index_fill_(0, indices, 0)
            m_e.append(m_.view(shape).detach())

        self.momentums = copy.deepcopy(m_e)