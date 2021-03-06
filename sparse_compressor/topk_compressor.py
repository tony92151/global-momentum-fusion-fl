import torch
from copy import deepcopy as dcopy
import math
import collections


class topkCompressor:
    def __init__(self, compress_rate: float = 0.5, device=torch.device("cpu")):
        self.compress_rate = compress_rate
        self.device = device

    def set_compress_rate(self, compress_rate):
        self.compress_rate = compress_rate

    def compress(self, gradient_dict: dict, compress: bool = True):
        if gradient_dict['compressed']:
            return gradient_dict

        # gradient_tmp = dcopy(gradient_dict['gradient'])

        new_gradient = {}
        for k in gradient_dict.keys():
            if k == "gradient":
                new_gradient["gradient"] = collections.OrderedDict({k: None for k in gradient_dict['gradient'].keys()})
            else:
                new_gradient[k] = gradient_dict[k]

        for k in gradient_dict['gradient'].keys():
            tensor = gradient_dict['gradient'][k].to(self.device)

            shape = list(tensor.size())
            tensor = tensor.flatten()
            numel = tensor.numel()
            tensor_calculate = tensor

            tensor_calculate = tensor_calculate.abs()
            tensor_calculate_filtered = tensor_calculate[tensor_calculate > 0]

            if len(tensor_calculate) == 0 or self.compress_rate == 1.0:
                compress = False

            if compress:
                cr = max(0.0, min(1.0, self.compress_rate))
                thr = find_threshold_by_sort(tensor_calculate_filtered, cr)
                # thr = find_threshold_by_approach(tensor_calculate_filtered, cr)

                mask = tensor_calculate.to(self.device) >= thr
            else:
                mask = tensor.abs().to(self.device) > 0

            indices, = torch.where(mask)
            values = tensor[indices]

            tensor_compressed = values.cpu().tolist()
            ctx = shape, mask.cpu().tolist(), numel
            new_gradient['gradient'][k] = (tensor_compressed, ctx)
        new_gradient['compressed'] = True
        return new_gradient

    def gf_compress(self, gradient_dict: dict, global_gradient_dict, fusion_ratio=0.5, compress: bool = True):
        if gradient_dict['compressed']:
            return gradient_dict

        if global_gradient_dict is None:
            return self.compress(gradient_dict=gradient_dict,
                                 compress=True)

        new_gradient = {}
        for k in gradient_dict.keys():
            if k == "gradient":
                new_gradient["gradient"] = collections.OrderedDict({k: None for k in gradient_dict['gradient'].keys()})
            else:
                new_gradient[k] = gradient_dict[k]

        for k in gradient_dict['gradient'].keys():
            tensor = gradient_dict['gradient'][k].to(self.device)
            gf_tensor = global_gradient_dict['gradient'][k].to(self.device)

            shape = list(tensor.size())
            tensor = tensor.flatten()
            numel = tensor.numel()
            tensor_calculate = tensor

            gf_tensor = gf_tensor.flatten()
            gf_tensor_calculate = gf_tensor

            mix_tensor_calculate = torch.add(normalize(tensor_calculate.abs()),
                                             normalize(gf_tensor_calculate.abs()).mul(fusion_ratio/(1-fusion_ratio)))

            mix_tensor_calculate_filtered = mix_tensor_calculate[mix_tensor_calculate > 0]

            if len(tensor_calculate) == 0 or self.compress_rate == 1.0:
                compress = False

            if compress:
                cr = max(0.0, min(1.0, self.compress_rate))
                thr = find_threshold_by_sort(mix_tensor_calculate_filtered, cr)
                # thr = find_threshold_by_approach(tensor_calculate_filtered, cr)

                mask = mix_tensor_calculate.to(self.device) >= thr
            else:
                mask = tensor.abs().to(self.device) > 0

            indices, = torch.where(mask)
            values = tensor[indices]

            tensor_compressed = values.cpu().tolist()
            ctx = shape, mask.cpu().tolist(), numel
            new_gradient['gradient'][k] = (tensor_compressed, ctx)
        new_gradient['compressed'] = True
        return new_gradient

    def decompress(self, gradient_dict: dict):
        if not gradient_dict['compressed']:
            return gradient_dict

        new_gradient = {}
        for k in gradient_dict.keys():
            if k == "gradient":
                new_gradient["gradient"] = collections.OrderedDict({k: None for k in gradient_dict['gradient'].keys()})
            else:
                new_gradient[k] = gradient_dict[k]

        for k in gradient_dict['gradient'].keys():
            # print(time.time())
            j = gradient_dict['gradient'][k]
            new_mem, ctx = j
            shape, mask, numel = ctx

            values = torch.tensor(new_mem).to(self.device)
            indices = torch.tensor([i for i in range(len(mask)) if mask[i]]).type(torch.long).to(self.device)
            mask = torch.tensor(mask)

            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
            tensor_decompressed.scatter_(0, indices, values)
            new_gradient['gradient'][k] = tensor_decompressed.view(shape)
        new_gradient['compressed'] = False
        return new_gradient


def find_threshold_buildin_function(tensor, compress_rate=1.0):
    thr = torch.min(
        torch.topk(tensor.abs(), max(1, int(tensor.numel() * compress_rate)), largest=True, sorted=False)[0])
    return thr


def find_threshold_by_sort(tensor, cr):
    numel = tensor.numel()
    idx = max(0, min(numel, round(numel * float(cr))))
    values, _ = torch.sort(tensor)
    values = torch.fliplr(values.unsqueeze(0)).squeeze(0)
    return values[idx]


def find_threshold_by_approach(tensor, compress_rate=1.0, max_iter=10, device=torch.device("cpu")):
    tmin = torch.min(tensor)
    tmax = torch.max(tensor)
    threshold = 0.0
    for _ in range(max_iter):
        threshold = (tmax + tmin) / 2.0
        mask = tensor.abs().to(device) >= threshold
        selected = mask.sum()
        # +- 5% is ok
        if selected > (tensor.numel() * min(compress_rate + 0.05, 1)):
            tmin = threshold
            continue
        if selected < (tensor.numel() * max(compress_rate - 0.05, 0.01)):
            tmax = threshold
            continue
        break
    return threshold


def normalize(value):
    value /= torch.norm(value)
    return value
