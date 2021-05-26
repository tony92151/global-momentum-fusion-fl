import json
import os, copy
import torch
from torch import tensor
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from torch.utils.data import SubsetRandomSampler
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def cifar_dataloaders(root="./data/cifar10", index_path="./cifar10/niid/index.json", batch_size=128, show=True):
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=2, shuffle=True)

    trainloaders = []
    for i in range(len(context.keys())):
        trainloaders.append(torch.utils.data.DataLoader(trainset,
                                                        batch_size=batch_size,
                                                        num_workers=2,
                                                        sampler=torch.utils.data.SubsetRandomSampler(context[str(i)])))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    if show:
        for j in range(len(context.keys())):
            ans = [0 for i in range(10)]
            for i in context[str(j)]:
                ans[trainset.targets[i]] += 1
            print("client: {} , {}, sum: {}".format(j, ans, sum(ans)))

    return {"test": testloader, "train_s": trainloaders, "train": trainloader}


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

    if clients > len(train_data['users']):
        raise ValueError("Request clients({}) larger then dataset provide({}).".format(clients, len(train_data['users'])))

    train_data['users'] = train_data['users'][:clients]
    test_data['users'] = test_data['users'][:clients]

    # train_data_ = copy.deepcopy(train_data)
    # test_data_ = copy.deepcopy(test_data)
    #############################################################################
    train_data_all_x = []
    train_data_all_y = []
    train_idx = []
    for i in train_data["users"]:
        train_data_all_x += train_data["user_data"][i]["x"]
        train_data_all_y += train_data["user_data"][i]["y"]
        train_idx.append(len(train_data["user_data"][i]["y"]))

    train_dataset = MNISTDataset(torch.tensor(train_data_all_x).view(-1, 28, 28),
                                 torch.tensor(train_data_all_y), transform=data_transform)

    train_idx = len_to_index(train_idx)
    trainloaders = [torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        sampler=SubsetRandomSampler(train_idx[i]))
        for i in range(len(train_idx))]

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    #############################################################################
    test_data_all_x = []
    test_data_all_y = []
    test_idx = []
    for i in test_data["users"]:
        test_data_all_x += test_data["user_data"][i]["x"]
        test_data_all_y += test_data["user_data"][i]["y"]
        test_idx.append(len(test_data["user_data"][i]["y"]))

    test_dataset = MNISTDataset(torch.tensor(test_data_all_x).view(-1, 28, 28),
                                torch.tensor(test_data_all_y), transform=data_transform)

    test_idx = len_to_index(test_idx)
    testloaders = [torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        sampler=SubsetRandomSampler(test_idx[i]))
        for i in range(len(test_idx))]

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    # test: single loader
    # train: single loader
    # train_s: loaders
    # test_s: loaders
    return {"test": testloader, "train_s": trainloaders,
            "test_s": testloaders, "train": trainloader}
    # return {"test": copy.deepcopy(testloader), "train": copy.deepcopy(trainloader)}


def len_to_index(train_idx):
    l = list(range(sum(train_idx)))
    ll = []
    for i in train_idx:
        ll.append(l[:i])
        l = l[i:]
    return ll
