import json
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np


def cifar_dataloaders(root="./data/cifar10", index_path="./cifar10/niid/index.json", batch_size=128):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform_test)

    file_ = open(index_path, 'r')
    context = json.load(file_)
    file_.close()

    trainloaders = []

    for i in range(len(context.keys())):
        trainloaders.append(torch.utils.data.DataLoader(trainset,
                                                        batch_size=batch_size,
                                                        num_workers=2,
                                                        sampler=torch.utils.data.SubsetRandomSampler(context[str(i)])))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    return {"test": testloader, "train": trainloaders}


class MNISTDataset(Dataset):
    """EMNIST dataset"""

    def __init__(self, feature, target, transform=None):
        self.X = []
        self.Y = target
        if transform is not None:
            for i in range(len(feature)):
                self.X.append(transform(feature[i]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def femnist_dataloaders(root="./data/femnist", batch_size=128, clients=10):
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = torch.load(os.path.join(root, "train_data.pt"))
    test_data = torch.load(os.path.join(root, "test_data.pt"))

    train_data['users'] = train_data['users'][:clients]

    test_data['users'] = test_data['users'][:clients]

    #############################################################################
    trainloaders = []
    xs = np.array([])
    ys = np.array([])
    for i in train_data["users"]:
        x = np.array(train_data["user_data"][i]["x"])
        y = np.array(train_data["user_data"][i]["y"])
        x = x.reshape(-1, 28, 28)
        train_set = MNISTDataset(torch.from_numpy(x), torch.from_numpy(y), transform=data_transform)
        trainloaders.append(torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        num_workers=2,
                                                        shuffle=True))
        if len(xs) == 0:
            xs = x
            ys = y
            continue
        np.append(xs, x, axis=0)
        np.append(ys, y, axis=0)
    train_set = MNISTDataset(torch.from_numpy(xs), torch.from_numpy(ys), transform=data_transform)
    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              num_workers=2,
                                              shuffle=True)

    #############################################################################
    testloaders = []
    xs = np.array([])
    ys = np.array([])
    for i in test_data["users"]:
        x = np.array(test_data["user_data"][i]["x"])
        y = np.array(test_data["user_data"][i]["y"])
        x = x.reshape(-1, 28, 28)
        test_set = MNISTDataset(torch.from_numpy(x), torch.from_numpy(y), transform=data_transform)
        testloaders.append(torch.utils.data.DataLoader(test_set,
                                                       batch_size=batch_size,
                                                       num_workers=2,
                                                       shuffle=True))
        if len(xs) == 0:
            xs = x
            ys = y
            continue
        np.append(xs, x, axis=0)
        np.append(ys, y, axis=0)

    test_set = MNISTDataset(torch.from_numpy(xs), torch.from_numpy(ys), transform=data_transform)
    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             num_workers=2,
                                             shuffle=True)
    # test: single loader
    # train: single loader
    # train_s: loaders
    # test_s: loaders
    return {"test": testloader, "train_s": trainloaders, "test_s": testloaders, "train": trainloader}
