import torch
import copy
import math

# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from torch.multiprocessing import Pool
torch.multiprocessing.set_start_method('spawn', force=True)


def normalize(value):
    value = copy.deepcopy(value)
    value /= torch.norm(value)
    return value


class GFCCompressor:
    def __init__(self, compress_ratio=0.5, fusing_ratio=0.8, device=torch.device("cpu"), pool=None):
        # super().__init__(average=True, tensors_size_are_same=False)
        self.compress_ratio = compress_ratio
        self.fusing_ratio = fusing_ratio
        self.param_groups_c = None
        self.device = device
        self.pool = pool

    def clean(self):
        self.param_groups_c = None

    def compress_by_layer(self, param):
        pass

    def compress(self, mem, gmome=None, compress=True):
        agg_gradient = copy.deepcopy(mem)

        # if gmome is not None:
        #     n_grad = normalize(agg_gradient, device=self.device)
        #     n_mome = normalize(gmome, device=self.device)
        # else:
        #     n_grad = None
        #     n_mome = None

        compressed_grad = []
        ############################################################
        # https://docs.python.org/zh-tw/3/library/concurrent.futures.html#processpoolexecutor-example
        # if self.pool is not None:
        #     # gmome=None, n_grad=None, n_mome=None, fusing_ratio=1.0
        #     prims = [{"tensor": copy.deepcopy(agg_gradient[t]),
        #               "idx": t,
        #               "gmome": gmome,
        #               "n_grad": n_grad,
        #               "n_mome": n_mome,
        #               "compress_ratio": self.compress_ratio,
        #               "compress": compress,
        #               "fusing_ratio": self.fusing_ratio,
        #               "device": self.device}
        #              for t in range(len(agg_gradient))]
        #     #with ProcessPoolExecutor(max_workers=self.pool) as executor:
        #     #    procs = executor.map(layer_compress, prims)
        #     with Pool(processes=self.pool) as p:  # Paralleizing over 2 GPUs
        #        results = p.map(layer_compress, prims)
        #     return [i for i in results]
        ############################################################

        for t in range(len(agg_gradient)):
            tensor = agg_gradient[t].to(self.device)

            shape = list(tensor.size())
            tensor = tensor.flatten()
            numel = tensor.numel()

            if gmome is not None:
                tensor_a = self.fusing_ratio * normalize(gmome[t].flatten()).to(self.device) \
                           + (1.0 - self.fusing_ratio) * normalize(tensor).to(self.device)
            else:
                tensor_a = tensor

            # tensor_a = tensor.abs()
            tensor_a = tensor_a.abs()
            tensor_b = tensor_a[tensor_a > 0]

            if not len(tensor_a) == 0:
                tmin = torch.min(tensor_b)
                tmax = torch.max(tensor_b)
                pass
            else:
                compress = False

            if self.compress_ratio == 1:
                compress = False

            if compress:
                # for i in range(10):
                #     thr = (tmax + tmin) / 2
                #     mask = tensor_b.abs().to(self.device) >= thr.to(self.device)
                #     selected = mask.sum()
                #
                #     if selected > (tensor_b.numel() * min(self.compress_ratio + 0.05, 1)):
                #         tmin = thr
                #         continue
                #     if selected < (tensor_b.numel() * max(self.compress_ratio - 0.05, 0.01)):
                #         tmax = thr
                #         continue
                #     break
                # cr = max(0.0, min(1.0, self.compress_ratio))
                # thr = torch.min(torch.topk(tensor_b.abs(), max(1, int(tensor_b.numel() * cr)),
                #                            largest=True, sorted=False)[0])
                # jit function
                cr = max(0.0, min(1.0, self.compress_ratio))
                # thr = find_threshold(tensor_b, torch.tensor(cr))
                thr = find_threshold_by_sort(tensor_b, cr)

                mask = tensor_a.to(self.device) >= thr.to(self.device)
            else:
                mask = tensor.abs().to(self.device) > 0

            indices, = torch.where(mask)
            values = tensor[indices]

            tensor_compressed = values.cpu().tolist()  # , indices
            ctx = shape, mask.cpu().tolist(), numel
            # tensor boolean is to big

            compressed_grad.append((tensor_compressed, ctx))
        return compressed_grad

    def decompress(self, mem):
        agg_gradient = copy.deepcopy(mem)
        decompressed_mem = []
        for j in agg_gradient:
            new_mem, ctx = j
            shape, mask, numel = ctx

            values = torch.tensor(new_mem).to(self.device)
            indices = torch.tensor([i for i in range(len(mask)) if mask[i]]).type(torch.long).to(self.device)
            mask = torch.tensor(mask)

            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
            tensor_decompressed.scatter_(0, indices, values)
            decompressed_mem.append(tensor_decompressed.view(shape))
        return decompressed_mem


@torch.jit.script
def find_threshold(tensor, cr):
    thr = torch.min(torch.topk(tensor.abs(), max(1, int(tensor.numel() * cr)), largest=True, sorted=False)[0])
    return thr


def find_threshold_by_sort(tensor, cr):
    numel = tensor.numel()
    idx = max(0, min(numel, round(numel * float(cr))))
    values, indices = torch.sort(tensor)
    values = torch.fliplr(values.unsqueeze(0)).squeeze(0)
    return values[idx]


def layer_compress(dic):
    tensor = dic["tensor"]
    t = dic["idx"]
    gmome = dic["gmome"]
    n_grad = dic["n_grad"]
    n_mome = dic["n_mome"]
    compress_ratio = dic["compress_ratio"]
    compress = dic["compress"]
    fusing_ratio = dic["fusing_ratio"]
    device = dic["device"]

    tensor = tensor.to(device)

    shape = list(tensor.size())
    tensor = tensor.flatten()
    numel = tensor.numel()

    if gmome is not None:
        tensor_a = fusing_ratio * n_mome[t] + (1 - fusing_ratio) * n_grad[t]
    else:
        tensor_a = tensor

    tensor_a = tensor_a.abs()
    tensor_a = tensor_a[tensor_a > 0]

    if not len(tensor_a) == 0:
        tmin = torch.min(tensor_a)
        tmax = torch.max(tensor_a)
    else:
        compress = False

    if compress or (compress_ratio == 1):
        if not len(tensor_a) == 0:
            for i in range(10):
                thr = (tmax + tmin) / 2
                mask = tensor.abs().to(device) >= thr.to(device)
                selected = mask.sum()

                if selected > (tensor_a.numel() * min(compress_ratio + 0.05, 1)):
                    tmin = thr
                    continue
                if selected < (tensor_a.numel() * max(compress_ratio - 0.05, 0.01)):
                    tmax = thr
                    continue
                break
        else:
            thr = torch.tensor(1)  # becauce all element are 0, set thr=1 make mask mask out everything
            mask = tensor.abs().to(device) >= thr.to(device)
            selected = mask.sum()
    else:
        mask = tensor.abs().to(device) > 0
        # selected = mask.sum()

    indices, = torch.where(mask)
    values = tensor[indices]

    tensor_compressed = values.cpu().tolist()  # , indices
    ctx = shape, mask.cpu().tolist(), numel
    return tensor_compressed, ctx
