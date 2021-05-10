import json
import os
import torch
import torchvision
from torchvision import transforms


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
