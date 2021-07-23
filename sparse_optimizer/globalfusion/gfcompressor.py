import torch
import copy
import math

from sparse_optimizer.topk_compressor import topkCompressor
from sparse_optimizer.topk_compressor import find_threshold_buildin_function
from sparse_optimizer.topk_compressor import find_threshold_by_sort
from sparse_optimizer.topk_compressor import find_threshold_by_approach
# topkCompressor                -> slower
# find_threshold_by_sort        -> medium
# find_threshold_by_approach    -> faster but not accurate


def normalize(value):
    value = copy.deepcopy(value)
    value /= torch.norm(value)
    return value

class GFCCompressor(topkCompressor):
    def __init__(self, compress_ratio=0.5, fusing_ratio=0.8, device=torch.device("cpu")):
        super().__init__(average=True, tensors_size_are_same=False)
        self.compress_ratio = compress_ratio
        self.fusing_ratio = fusing_ratio
        self.device = device
        super(GFCCompressor).__init__(compress_ratio=compress_ratio, fusing_ratio=fusing_ratio, device=device)

    def compress(self, mem:list, gmome:list=None, compress:bool=True):
        tensor_memory = copy.deepcopy(mem)
        compressed_grad = []

        for t in range(len(tensor_memory)):
            tensor = tensor_memory[t].to(self.device)

            shape = list(tensor.size())
            tensor = tensor.flatten()
            numel = tensor.numel()

            if gmome is not None:
                tensor_calculate = self.fusing_ratio * normalize(gmome[t].flatten()).to(self.device) \
                           + (1.0 - self.fusing_ratio) * normalize(tensor).to(self.device)
            else:
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
