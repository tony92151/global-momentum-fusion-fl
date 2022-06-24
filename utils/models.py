import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.dgc_model.resnet import CifarResNet
from utils.shakespeare_model.stacked_lstm import LSTM_shakespeare_1L, LSTM_shakespeare_2L
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


# These models inheritance from CifarResNet are a little bit different different from official one
class ResNet20_cifar_gdc(CifarResNet):
    def __init__(self):
        super(ResNet20_cifar_gdc, self).__init__(params=[(16, 3, 1), (32, 3, 2), (64, 3, 2)], num_classes=10)


class ResNet56_cifar_gdc(CifarResNet):
    def __init__(self):
        super(ResNet56_cifar_gdc, self).__init__(params=[(16, 9, 1), (32, 9, 2), (64, 9, 2)], num_classes=10)


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
                                                num_claisses=62)


#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
#
# # These models inheritance from CifarResNet are a little bit different dfferent from official one

class ResNet56_femnist_gdc(CifarResNet):
    def __init__(self):
        super(ResNet56_femnist_gdc, self).__init__(params=[(16, 9, 1), (32, 9, 2), (64, 9, 2)], num_classes=62)
        self._modules['features'][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


class ResNet110_femnist_gdc(CifarResNet):
    def __init__(self):
        super(ResNet110_femnist_gdc, self).__init__(params=[(16, 18, 1), (32, 18, 2), (64, 18, 2)], num_classes=62)
        self._modules['features'][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


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

##########################################################################################
class ResNet9_mnist(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet9_mnist, self).__init__(block=torchvision.models.resnet.BasicBlock,
                                            layers=[1, 1, 1, 1],
                                            num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

class ResNet18_mnist(torchvision.models.resnet.ResNet):
    def __init__(self):
        super(ResNet18_mnist, self).__init__(block=torchvision.models.resnet.BasicBlock,
                                            layers=[2, 2, 2, 2],
                                            num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

class ResNet20_mnist_gdc(CifarResNet):
    def __init__(self):
        super(ResNet20_mnist_gdc, self).__init__(params=[(16, 3, 1), (32, 3, 2), (64, 3, 2)], num_classes=10)

        self.features[0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
##########################################################################################

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


class Net_femnist_afo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=True)
        self.re = nn.ReLU()
        self.max2d = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.max2d(x)
        x = self.drop1(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


#############################################
MODELS_TABLE = {
    # for cifar10
    "small_cifar": Net_cifar,
    "resnet9_cifar": ResNet9_cifar,
    "resnet18_cifar": ResNet18_cifar,
    "resnet50_cifar": ResNet50_cifar,
    "resnet101_cifar": ResNet101_cifar,
    "resnet20_cifar_gdc": ResNet20_cifar_gdc,
    "resnet56_cifar_gdc": ResNet56_cifar_gdc,
    "resnet110_cifar_gdc": ResNet110_cifar_gdc,
    # for femnist
    "small_femnist": Net_femnist,
    "net_femnist_afo": Net_femnist_afo,
    "resnet9_femnist": ResNet9_femnist,
    "resnet18_femnist": ResNet18_femnist,
    "resnet50_femnist": ResNet50_femnist,
    "resnet101_femnist": ResNet101_femnist,
    "resnet56_femnist_gdc": ResNet56_femnist_gdc,
    "resnet110_femnist_gdc": ResNet110_femnist_gdc,
    # for shakespeare
    "lstm_shakespeare_1L": LSTM_shakespeare_1L,
    "lstm_shakespeare_2L": LSTM_shakespeare_2L,
    # for fashionmnist
    "resnet9_mnist": ResNet9_mnist,
    "resnet18_mnist": ResNet18_mnist,
    "resnet9_fashionmnist":ResNet9_mnist,
    "resnet18_fashionmnist":ResNet18_mnist,
    "resnet20_fashionmnist":ResNet20_mnist_gdc
}


def MODELS(config: Configer = None):
    if config is None:
        raise ValueError("config shouldn't be none")
    if config.trainer.get_model() not in MODELS_TABLE.keys():
        raise ValueError("model not define in {}".format(MODELS_TABLE.keys()))
    return MODELS_TABLE[config.trainer.get_model()]
