import base64
import pickle
import sys
from copy import deepcopy as dcopy
import torch


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


def weight_aggregater(gradient_list, device=torch.device("cpu")):
    for g in gradient_list:
        if g['compressed']:
            raise ValueError("gradient should be decompress before aggregate.")

    new_gradient_list = dcopy(gradient_list[0])
    for k in new_gradient_list['gradient'].keys():
        new_gradient_list['gradient'][k] = None

    all_steps = sum([g["step_count"] for g in gradient_list])

    for k in gradient_list[0]["gradient"].keys():
        result = torch.sum(
            torch.stack([g["gradient"][k].mul_(g["step_count"]).to(device) for g in gradient_list]),
            dim=0)
        new_gradient_list["gradient"][k] = result.mul_(1.0 / all_steps)

    return [new_gradient_list]


def add_aggregater(gradient_list, device=torch.device("cpu")):
    for g in gradient_list:
        if g['compressed']:
            raise ValueError("gradient should be decompress before aggregate.")

    new_gradient_list = dcopy(gradient_list[0])
    for k in new_gradient_list['gradient'].keys():
        new_gradient_list['gradient'][k] = None

    for k in gradient_list[0]["gradient"].keys():
        result = torch.sum(torch.stack([g["gradient"][k].to(device) for g in gradient_list]), dim=0)
        new_gradient_list["gradient"][k] = result

    return [new_gradient_list]
