import torch
import copy
import math

class topkCompressor:
    def __init__(self, compress_ratio:float=0.5, device=torch.device("cpu")):
        self.compress_ratio = compress_ratio
        self.device = device

    def compress(self, mem:list, compress:bool=True):
        tensor_memory = copy.deepcopy(mem)
        compressed_grad = []

        for t in range(len(tensor_memory)):
            tensor = tensor_memory[t].to(self.device)

            shape = list(tensor.size())
            tensor = tensor.flatten()
            numel = tensor.numel()
            tensor_calculate = tensor

            tensor_calculate = tensor_calculate.abs()
            tensor_calculate_filtered = tensor_calculate[tensor_calculate > 0]

            if len(tensor_calculate) == 0 or self.compress_ratio == 1.0:
                compress = False

            if compress:
                cr = max(0.0, min(1.0, self.compress_ratio))
                thr = find_threshold_by_sort(tensor_calculate_filtered, cr)
                # thr = find_threshold_by_approach(tensor_calculate_filtered, cr)

                mask = tensor_calculate.to(self.device) >= thr
            else:
                mask = tensor.abs().to(self.device) > 0

            indices, = torch.where(mask)
            values = tensor[indices]

            tensor_compressed = values.cpu().tolist()
            ctx = shape, mask.cpu().tolist(), numel
            compressed_grad.append((tensor_compressed, ctx))
        return compressed_grad

    def decompress(self, mem:list):
        tensor_memory = copy.deepcopy(mem)
        decompressed_mem = []
        for j in tensor_memory:
            new_mem, ctx = j
            shape, mask, numel = ctx

            values = torch.tensor(new_mem).to(self.device)
            indices = torch.tensor([i for i in range(len(mask)) if mask[i]]).type(torch.long).to(self.device)
            mask = torch.tensor(mask)

            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
            tensor_decompressed.scatter_(0, indices, values)
            decompressed_mem.append(tensor_decompressed.view(shape))
        return decompressed_mem


def find_threshold_buildin_function(tensor, cr):
    thr = torch.min(torch.topk(tensor.abs(), max(1, int(tensor.numel() * cr)), largest=True, sorted=False)[0])
    return thr


def find_threshold_by_sort(tensor, cr):
    numel = tensor.numel()
    idx = max(0, min(numel, round(numel * float(cr))))
    values, _ = torch.sort(tensor)
    values = torch.fliplr(values.unsqueeze(0)).squeeze(0)
    return values[idx]

def find_threshold_by_approach(tensor, compress_ratio=1.0, max_iter=10, device=torch.device("cpu")):
    tmin = torch.min(tensor)
    tmax = torch.max(tensor)
    threshold = 0.0
    for _ in range(max_iter):
        threshold = (tmax + tmin) / 2.0
        mask = tensor.abs().to(device) >= threshold
        selected = mask.sum()
        # +- 5% is ok
        if selected > (tensor.numel() * min(compress_ratio + 0.05, 1)):
            tmin = threshold
            continue
        if selected < (tensor.numel() * max(compress_ratio - 0.05, 0.01)):
            tmax = threshold
            continue
        break
    return threshold