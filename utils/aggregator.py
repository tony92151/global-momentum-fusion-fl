import base64
import pickle
import sys
from copy import deepcopy as dcopy
import torch
from globalfusion.gfcompressor import GFCCompressor


def add_mean_var(means=None, vrs=None, tracks=None):
    if (type(means) is not list) or (type(means) is not list) or (type(means) is not list):
        raise ValueError("means should be list.")
    tracks_sum = sum(tracks)

    means = [torch.tensor(i) for i in means]
    vrs = [torch.tensor(i) for i in vrs]

    means_ = [means[i] * tracks[i] for i in range(len(means))]
    mean_result = means_[0]
    for m in range(1, len(means_)):
        mean_result.add_(means[m])
    mean_result = mean_result.mul(1 / tracks_sum)

    vs = []
    for v in range(len(vrs)):
        vs.append(vrs[v].add(means[v].mul(means[v])) * tracks[v])

    vr_result = vs[0]
    for v in range(1, len(vs)):
        vr_result.add_(vs[v])
    vr_result = vr_result.mul(1 / tracks_sum)

    return mean_result.tolist(), vr_result.tolist(), tracks_sum


def aggregater(gradient_list, device=torch.device("cpu"), aggrete_bn=False):
    agg_gradient = []
    all_steps = sum([j["step_count"] for j in gradient_list])
    for i in range(len(gradient_list[0]["gradient"])):
        result = torch.sum(torch.stack([j["gradient"][i].mul_(j["step_count"]).to(device) for j in gradient_list]),
                           dim=0)
        agg_gradient.append(result.mul_(1.0 / all_steps))

    if 'bn' in gradient_list[0].keys() and aggrete_bn:
        bn_result = []
        for i in range(0, len(gradient_list[0]["bn"]), 3):
            m = []
            v = []
            t = []
            for j in gradient_list:
                m.append(j["bn"][i])
                v.append(j["bn"][i + 1])
                t.append(j["bn"][i + 2])
            m_, v_, t_ = add_mean_var(m, v, t)
            bn_result.append(m_)
            bn_result.append(v_)
            bn_result.append(t_)
        return {"gradient": agg_gradient, "bn": bn_result}

    return {"gradient": agg_gradient}


def decompress(gradient, device=torch.device("cpu")):
    compresser = GFCCompressor(device=device)
    return compresser.decompress(gradient)


def compress(gradient, device=torch.device("cpu")):
    compresser = GFCCompressor(device=device)
    return compresser.compress(gradient, gmome=None, compress=False)


def get_serialize_size(obj):
    b = base64.b64encode(pickle.dumps(obj)).decode('utf-8')
    return round(sys.getsizeof(b) * 10e-6, 3)


def parameter_count(value):
    if isinstance(value, torch.nn.Module):  # value is a model
        return sum([l[1].numel() for l in value.state_dict().items()])
    elif isinstance(value, list) and isinstance(value[0], dict):  # value is a compressed gradient list
        # value = [gradient, gradient, ...]
        # gradient = {"gradient":[(tensor, (shape, mask, numel)), (tensor, (shape, mask, numel)), ...],
        #             "step_count": int}
        value_ = [v["gradient"] for v in value]
        # sum the mask to get number of parameters in each layer and sum these numbers
        return sum([sum([len(l[0]) for l in g]) for g in value_])


def set_gradient(opt, cg):
    agged_grad = cg
    for group in opt.param_groups:
        for p in range(len(group['params'])):
            group['params'][p].grad = dcopy(agged_grad[p]).to(group['params'][p].device)
