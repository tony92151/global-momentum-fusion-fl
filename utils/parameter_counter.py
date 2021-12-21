import base64
import pickle
import sys
from copy import deepcopy as dcopy
import torch


def parameter_count(gradient_dict=None):
    if isinstance(gradient_dict, torch.nn.Module):  # value is a model
        return sum([l[1].numel() for l in gradient_dict.state_dict().items()])
    elif isinstance(gradient_dict, dict):
        # gradient_dict = {"compressed":True, "gradient": (tensor, (shape, mask, numel))}
        # sum the mask to get number of parameters in each layer and sum these numbers
        return sum([len(gradient_dict["gradient"][k]) for k in gradient_dict["gradient"].keys()])
