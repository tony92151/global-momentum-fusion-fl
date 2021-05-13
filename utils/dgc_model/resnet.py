# modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import collections

import torch
import torch.nn as nn

from torchvision.models.resnet import BasicBlock


__all__ = ['CifarResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
           'resnet110', 'resnet1202']


# modified from https://github.com/facebookarchive/fb.resnet.torch
class Shortcut(nn.Module):
    def __init__(self, stride, out_channels):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        s = list(x.shape)
        s[1] = self.out_channels - s[1]
        return torch.cat([x, torch.zeros(*s, device=x.device, dtype=x.dtype)],
                         dim=1)


class GlobalAvgPool2d(nn.Module):
    def forward(self, inputs):
        return inputs.mean(-1).mean(-1)


class CifarResNet(nn.Sequential):
    def __init__(self, params, num_classes=10, zero_init_residual=False):
        in_channels = params[0][0]
        layers = [
            nn.Conv2d(3, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        ]

        for out_channels, num_blocks, strides in params:
            for stride in [strides] + [1] * (num_blocks - 1):
                if stride != 1 or in_channels != out_channels:
                    shortcut = Shortcut(stride=stride, 
                                        out_channels=out_channels)
                else:
                    shortcut = None
                layers.append(BasicBlock(in_channels, out_channels, 
                                         stride=stride,
                                         downsample=shortcut))
                in_channels = out_channels
        layers.append(GlobalAvgPool2d())

        super().__init__(collections.OrderedDict([
            ('features', nn.Sequential(*layers)),
            ('classifier', nn.Linear(in_channels, num_classes))
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


def resnet20(**kwargs):
    return CifarResNet(params=[(16, 3, 1), (32, 3, 2), (64, 3, 2)], **kwargs)


def resnet32(**kwargs):
    return CifarResNet(params=[(16, 5, 1), (32, 5, 2), (64, 5, 2)], **kwargs)


def resnet44(**kwargs):
    return CifarResNet(params=[(16, 7, 1), (32, 7, 2), (64, 7, 2)], **kwargs)


def resnet56(**kwargs):
    return CifarResNet(params=[(16, 9, 1), (32, 9, 2), (64, 9, 2)], **kwargs)


def resnet110(**kwargs):
    return CifarResNet(params=[(16, 18, 1), (32, 18, 2), (64, 18, 2)], **kwargs)


def resnet1202(**kwargs):
    return CifarResNet(params=[(16, 200, 1), (32, 200, 2), (64, 200, 2)], 
                       **kwargs)
