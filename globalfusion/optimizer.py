import torch
import copy, os, time
from globalfusion.gfcompressor import GFCCompressor
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
"""
Original usage:

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
"""

"""
FGCSGD usage:

optimizer = FGCSGD(model.parameters(), lr=0.1, compress_ratio=0.5)

optimizer.memory.clean()

for input,target in dataloader:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    
    
optimizer.set_accumulate_gradient(model=model, record_batchnorm=True)
optimizer.compress(global_momentum=self.last_gradient["gradient"], compress=True, momentum_correction=True)
cg = optimizer.memory.compressed_mem
<send gradient>

if <receive aggregated gradient>:
    dg = optimizer.decompress(new_gradient)
    optimizer.set_gradient(dg)
    optimizer.step()
"""


# copy from torch/optim/sgd.py
class GFDGCSGD(torch.optim.Optimizer):
    def __init__(self, params, cid=-1, lr=None, dgc_momentum=0.9, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, compress_ratio=0.5, fusing_ratio=0.5, checkpoint=False,
                 device=torch.device("cpu"), pool=None):
        if lr is None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.memory = FGCMemory(momentum=dgc_momentum, device=device)
        self.compressor = GFCCompressor(compress_ratio=compress_ratio, fusing_ratio=fusing_ratio, device=device,
                                        pool=pool)
        self.checkpoint = checkpoint
        self.device = device
        self.cid = cid
        self.savepath = os.getenv("memory_checkpoint")

        self.verbose = False

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GFDGCSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GFDGCSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def print_(self, val):
        if self.verbose:
            print(val)

    def get_state(self):
        stat = self.state_dict()
        stat["momentums"] = self.memory.momentums
        stat["velocities"] = self.memory.velocities
        return stat

    def set_state(self, sta):
        self.load_state_dict(sta)
        self.memory.momentums = sta["momentums"]
        self.memory.velocities = sta["velocities"]

    def memory_checkpoint_save(self):
        m_ = self.compressor.compress(mem=self.memory.momentums, compress=False)
        v_ = self.compressor.compress(mem=self.memory.velocities, compress=False)
        checkpoint = {"momentums": m_,
                      "velocities": v_}
        if self.savepath is not None:
            os.makedirs(self.savepath, exist_ok=True)
            savepath = os.path.join(self.savepath, "memory_checkpoint_{}".format(self.cid))
        torch.save(checkpoint, savepath)

    def memory_checkpoint_restore(self):
        if self.savepath is not None:
            savepath = os.path.join(self.savepath, "memory_checkpoint_{}".format(self.cid))
        else:
            savepath = "/tmp/memory_checkpoint_{}".format(self.cid)

        if not os.path.exists(savepath):
            return
        try:
            checkpoint = torch.load(savepath)
            m_ = [i.to(self.device) for i in self.compressor.decompress(checkpoint['momentums'])]
            v_ = [i.to(self.device) for i in self.compressor.decompress(checkpoint['velocities'])]
            self.memory.momentums = m_
            self.memory.velocities = v_
        except:
            self.memory.momentums = None
            self.memory.velocities = None

    def memory_checkpoint_remove(self):
        pass
        # torch.save(self.memory, "/tmp/memory_checkpoint")

    def set_accumulate_gradient(self, record_batchnorm=False, model=None):

        if model is None:
            raise ("model sould't be none")
        else:
            mb = copy.deepcopy(model).train().to(self.device)
            param = list(mb.parameters())
            for p in param:
                p.requires_grad = False

        # param = mf.parameters()
        #         idx = 0
        #         for group in self.param_groups:
        #             for p in group['params']:
        #                 param[idx].add_(p, alpha=-1.0)
        #                 param[idx].mul_(1/group['lr'])
        #                 idx+=1
        #         self.memory.mem["gradient"] = copy.deepcopy(param)
        #         for p in self.memory.mem["gradient"]:
        #             p.detach()
        #         print(self.memory.mem["gradient"])
        # record BATCHNORM2D param
        if record_batchnorm:
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

    def compress(self, global_momentum=None, compress=True, momentum_correction=False):
        # r = self.compressor.compress(self.memory.get_mem(), mome=mome, compress=compress, fusing=fusing)
        # self.memory.set_compressed_mem(r)

        if momentum_correction:
            if self.checkpoint:
                self.memory_checkpoint_restore()
            gm = self.memory.mem["gradient"]
            self.print_("optimizer >> compensate, {}".format(time.time()))
            m = self.memory.compensate(gm)
            self.print_("optimizer >> compress, {}".format(time.time()))
            r = self.compressor.compress(m, gmome=global_momentum, compress=compress)
            self.print_("optimizer >> compress done, {}".format(time.time()))
            self.memory.update(r)

            if self.checkpoint:
                self.memory_checkpoint_save()
        else:
            self.print_("optimizer >> compress, {}".format(time.time()))
            r = self.compressor.compress(self.memory.mem["gradient"], gmome=global_momentum, compress=compress)
            self.print_("optimizer >> compress done, {}".format(time.time()))

        # bn should't be compressed
        if 'bn' in self.memory.mem.keys():
            for p in range(len(self.memory.mem["bn"])):
                self.memory.mem["bn"][p] = self.memory.mem["bn"][p].tolist()

        self.memory.mem["gradient"] = r
        self.memory.set_compressed_mem(self.memory.mem)

    def decompress(self, d):
        d = self.compressor.decompress(d)
        self.memory.set_decompressed_mem(d)
        return d

    def get_compressed_gradient(self):
        return self.memory.compressed_mem

    def set_gradient(self, cg):
        agged_grad = copy.deepcopy(cg)
        for group in self.param_groups:
            for p in range(len(group['params'])):
                group['params'][p].grad = copy.deepcopy(agged_grad[p]).to(group['params'][p].device)

    def reload_model_parameter(self, params):
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

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


class FGCMemory:
    def __init__(self, momentum=0.9, device=torch.device("cpu")):
        self.mem = {"gradient": []}
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
