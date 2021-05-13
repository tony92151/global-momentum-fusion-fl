import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.dgc_model.resnet import CifarResNet


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


class Net(nn.Module):
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
    
    
    
class ResNet110_cifar_gdc(CifarResNet):
    def __init__(self):
        super(ResNet110_cifar_gdc, self).__init__(params=[(16, 18, 1), (32, 18, 2), (64, 18, 2)])
        
