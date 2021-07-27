import torch
import copy, os, time
from sparse_optimizer.topk_compressor import topkCompressor
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR

# copy from torch/optim/sgd.py
class BASE_SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, device=torch.device("cpu")):
        if lr is None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.memory = Memory(momentum=0.9, device=device)
        # self.compressor = topkCompressor(compress_ratio=compress_ratio, device=device)
        self.device = device
        self.step_count = 0
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(BASE_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BASE_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def get_state(self):
        stat = {"momentums": self.memory.momentums, "velocities": self.memory.velocities}
        return stat

    def set_state(self, sta):
        self.memory.momentums = sta["momentums"]
        self.memory.velocities = sta["velocities"]

    def record_batchnorm(self, model=None):
        if model is None:
            raise ("model shouldn't be none")
        else:
            model = copy.deepcopy(model).train().to(self.device)
            param = list(model.parameters())
            for p in param:
                p.requires_grad = False
        # record BATCHNORM2D param
        bn = []
        for layer in model.cpu().modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                # bn.append([layer.running_mean.tolist(), layer.running_var.tolist(), layer.num_batches_tracked.tolist()])
                bn.append(layer.running_mean)
                bn.append(layer.running_var)
                bn.append(layer.num_batches_tracked)
        self.memory.mem["bn"] = bn
        for p in self.memory.mem["bn"]:
            p.detach()

    def compress(self):
        raise NotImplementedError("Do some thing in compress.")

    def decompress(self, compressed_gradient:list=None):
        if compressed_gradient is None:
            raise ValueError(compressed_gradient)
        return topkCompressor().decompress(mem=compressed_gradient)

    def get_compressed_gradient(self):
        return self.memory.compressed_mem

    def set_gradient(self, gradient:list):
        aggregated_gradradient = copy.deepcopy(gradient)
        for group in self.param_groups:
            for p in range(len(group['params'])):
                group['params'][p].grad = copy.deepcopy(aggregated_gradradient[p]).to(group['params'][p].device)

    def set_learning_rate(self, lr=0.001):
        for group in self.param_groups:
            group['lr'] = lr

    @torch.no_grad()
    def step(self, closure=None):

        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        emp = len(self.memory.mem["gradient"]) == 0
        idx = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    print("None")
                    continue

                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if emp:
                    self.memory.mem["gradient"].append(copy.deepcopy(d_p))
                else:
                    self.memory.mem["gradient"][idx].add_(copy.deepcopy(d_p))
                    idx += 1
                p.add_(d_p, alpha=-group['lr'])

        # self.step_count += 1
        self.memory.mem["step_count"]+=1
        # self.memory.clean()
        return loss

    @torch.no_grad()
    def one_step(self, grad):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                d_p = grad[idx]
                # print("{}/{}/{}".format(p, d_p,group['lr']))
                p.add_(d_p, alpha=-group['lr'])
                idx += 1


class Memory:
    def __init__(self, momentum=0.9, device=torch.device("cpu")):
        self.mem = {"gradient": [], 'step_count': 1}
        self.avg_mem = []
        self.compressed_mem = None
        self.decompressed_mem = None
        self.can_add = True
        self.momentum = momentum

        self.momentums = None
        self.velocities = None
        self.device = device

    def add_mem(self, mem=None, avg=False):
        if mem is None:
            gradient_list = copy.deepcopy(self.mem)
        else:
            gradient_list = copy.deepcopy(mem)
        if len(gradient_list) == 1:
            return gradient_list[0]

        avg_gradient = []
        for i in range(len(gradient_list[0])):
            result = torch.stack([j[i].to(self.device) for j in gradient_list]).sum(dim=0)
            if avg:
                avg_gradient.append(result / len(gradient_list))
            else:
                avg_gradient.append(result)
        return avg_gradient

    def compensate(self, gradient):
        avg_gradient = [i.to(self.device) for i in gradient]

        if self.momentums is None and self.velocities is None:
            self.momentums = copy.deepcopy(avg_gradient)
            self.velocities = copy.deepcopy(avg_gradient)
            vec = self.velocities
        else:
            mmt = self.momentums
            vec = self.velocities
            m_e = []
            v_e = []
            for m, v, g in zip(mmt, vec, avg_gradient):
                m_ = copy.deepcopy(m).to(self.device)
                v_ = copy.deepcopy(v).to(self.device)
                m_.mul_(self.momentum).add_(g.to(self.device))
                v_.add_(m_)

                m_e.append(m_)
                v_e.append(v_)

            self.momentums = m_e
            self.velocities = v_e

        return self.velocities

    def update(self, com_gradient):
        m_n = copy.deepcopy(self.momentums)
        m_n = [i.to(self.device) for i in m_n]
        v_n = copy.deepcopy(self.velocities)
        v_n = [i.to(self.device) for i in v_n]

        # com_gradients = copy.deepcopy(self.velocities)

        m_e = []
        v_e = []

        for j, m, v in zip(com_gradient, m_n, v_n):
            new_mem, ctx = j
            shape, mask, numel = ctx
            indices, = torch.where(torch.BoolTensor(mask).to(self.device))
            m_ = copy.deepcopy(m).view(-1).index_fill_(0, indices, 0)
            v_ = copy.deepcopy(v).view(-1).index_fill_(0, indices, 0)
            m_e.append(m_.view(shape).detach())
            v_e.append(v_.view(shape).detach())

        self.momentums = copy.deepcopy(m_e)
        self.velocities = copy.deepcopy(v_e)

    def set_compressed_mem(self, d):
        # self.can_add = False
        self.compressed_mem = d
        pass

    def set_decompressed_mem(self, d):
        self.decompressed_mem = d
        pass

    def add(self, d):
        idx = 0
        emp = not self.mem
        for group in d:
            for p in group['params']:
                if emp:
                    self.mem.append(copy.deepcopy(p.grad))
                else:
                    self.mem[idx].add(p.grad)
                    idx += 1

    def get_mem(self):
        self.can_add = False
        return self.mem

    def get_compressed_mem(self):
        return self.compressed_mem

    def clean(self):
        self.mem = {"gradient": []}
        self.compressed_mem = None
        self.decompressed_mem = None
        self.can_add = True


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()
