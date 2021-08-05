import torch
import copy, os, time
from sparse_optimizer.topk_compressor import topkCompressor
from sparse_optimizer.globalfusion.gfcompressor import GFCCompressor
from sparse_optimizer.base_optimizer import BASE_SGD, Memory

class GFDGCSGD(BASE_SGD):
    def __init__(self, params, lr=None, dgc_momentum=0.9, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, compress_ratio=0.5, fusing_ratio=0.5,
                 device=torch.device("cpu")):

        super(GFDGCSGD, self).__init__(params=params, 
                                     lr=lr, 
                                     momentum=momentum, 
                                     dampening=dampening,
                                     weight_decay=weight_decay, 
                                     nesterov=nesterov, 
                                     device=device)
        
        self.device = device
        self.memory = Memory(momentum=dgc_momentum, device=device)
        self.compressor = GFCCompressor(compress_ratio=compress_ratio, fusing_ratio=fusing_ratio, device=device)

        self.global_momentum = None
        self.last_de_gradient = None
        ######
        self.verbose = False
        ######

    def print_(self, val):
        if self.verbose:
            print(val)

    def compress(self, global_momentum=None, compress:bool=True, momentum_correction:bool=False):
        if momentum_correction:
            self.print_("optimizer >> compensate, {}".format(time.time()))
            compensated_gradient = self.memory.compensate(self.memory.mem["gradient"])
            self.print_("optimizer >> compress, {}".format(time.time()))
            compress_result = self.compressor.compress(mem=compensated_gradient, gmome=global_momentum, compress=compress)
            self.print_("optimizer >> compress done, {}".format(time.time()))
            self.memory.update(compress_result)
        else:
            self.print_("optimizer >> compress, {}".format(time.time()))
            compress_result = self.compressor.compress(mem=self.memory.mem["gradient"], gmome=global_momentum, compress=compress)
            self.print_("optimizer >> compress done, {}".format(time.time()))

        # bn shouldn't be compressed
        if 'bn' in self.memory.mem.keys():
            for p in range(len(self.memory.mem["bn"])):
                self.memory.mem["bn"][p] = self.memory.mem["bn"][p].tolist()

        self.memory.mem["gradient"] = compress_result
        # self.memory.mem["step_count"] = self.step_count
        self.memory.set_compressed_mem(self.memory.mem)