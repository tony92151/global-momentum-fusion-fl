import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.dgc_model.resnet import CifarResNet
from utils.shakespeare_model import stacked_lstm
from utils.configer import Configer


class ResNet9_cifar(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet9_cifar, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[1, 1, 1, 1],
                                            num_classes=10)


class ResNet18_cifar(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet18_cifar, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[2, 2, 2, 2],
                                             num_classes=10)


class ResNet50_cifar(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet50_cifar, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[3, 4, 6, 3],
                                             num_classes=10)


class ResNet101_cifar(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet101_cifar, self).__init__(block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 23, 3],
                                              num_classes=10)


class ResNet110_cifar_gdc(CifarResNet):
    def __init__(self):
        super(ResNet110_cifar_gdc, self).__init__(params=[(16, 18, 1), (32, 18, 2), (64, 18, 2)], num_classes=10)


##########################################################################################
class ResNet9_femnist(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet9_femnist, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[1, 1, 1, 1],
                                              num_classes=62)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


class ResNet18_femnist(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet18_femnist, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[2, 2, 2, 2],
                                               num_classes=62)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


class ResNet50_femnist(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet50_femnist, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[3, 4, 6, 3],
                                               num_classes=62)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


class ResNet101_femnist(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet101_femnist, self).__init__(block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 23, 3],
                                                num_classes=62)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


class Net_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net_femnist(nn.Module):
    def __init__(self):
        super(Net_femnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 62)
        self.same_padding = nn.ReflectionPad2d(2)

    def forward(self, x):
        x = self.same_padding(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.same_padding(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#############################################
MODELS_TABLE = {
    # for cifar10
    "small_cifar": Net_cifar,
    "resnet9_cifar": ResNet9_cifar,
    "resnet18_cifar": ResNet18_cifar,
    "resnet50_cifar": ResNet50_cifar,
    "resnet101_cifar": ResNet101_cifar,
    "resnet110_cifar": ResNet110_cifar_gdc,
    # for femnist
    "small_femnist": Net_femnist,
    "resnet9_femnist": ResNet9_femnist,
    "resnet18_femnist": ResNet18_femnist,
    "resnet50_femnist": ResNet50_femnist,
    "resnet101_femnist": ResNet101_femnist,
    # for shakespeare
    "lstm_shakespeare": stacked_lstm,
}


def MODELS(config: Configer = None):
    if config is None:
        raise ValueError("config shouldn't be none")
    if config.trainer.get_model() not in MODELS.keys():
        raise ValueError("model not define in {}".format(MODELS.keys()))
    return MODELS_TABLE[config.trainer.get_model()]
