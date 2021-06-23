import json
import os, copy
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler
import torch.multiprocessing
from utils.configer import Configer

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

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform_test)
    ################################################################################################
    file_ = open(index_path.replace("datatype", "niid"), 'r')
    context = json.load(file_)
    file_.close()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    trainloaders = []
    for i in range(len(context.keys())):
        trainloaders.append(torch.utils.data.DataLoader(trainset,
                                                        batch_size=batch_size,
                                                        sampler=torch.utils.data.SubsetRandomSampler(context[str(i)])))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    ################################################################################################
    file_ = open(index_path.replace("datatype", "iid"), 'r')
    context2 = json.load(file_)
    file_.close()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    trainloaders_iid = []
    for i in range(len(context2.keys())):
        trainloaders_iid.append(torch.utils.data.DataLoader(trainset,
                                                            batch_size=batch_size,
                                                            sampler=torch.utils.data.SubsetRandomSampler(
                                                                context2[str(i)])))
    ################################################################################################
    if show:
        for j in range(len(context.keys())):
            ans = [0 for i in range(10)]
            for i in context[str(j)]:
                ans[trainset.targets[i]] += 1
            print("client: {} , {}, sum: {}".format(j, ans, sum(ans)))

    # test: single loader
    # train: single loader
    # train_s: loaders
    # test_s: loaders
    # train_s_iid: loaders with iid data
    return {"test": testloader,
            "train": trainloader,
            "train_s": trainloaders,
            "test_s": None,
            "train_s_iid": trainloaders_iid}


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
        raise ValueError(
            "Request clients({}) larger then dataset provide({}).".format(clients, len(train_data['users'])))

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
        sampler=SubsetRandomSampler(train_idx[i]))
        for i in range(len(train_idx))]

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        sampler=SubsetRandomSampler(test_idx[i]))
        for i in range(len(test_idx))]

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # iid dataloaders
    idx = list(range(len(train_data_all_x)))
    random.shuffle(idx)
    new_train_data_all_x = [train_data_all_x[i] for i in idx]
    new_train_data_all_y = [train_data_all_y[i] for i in idx]
    train_dataset_iid = MNISTDataset(torch.tensor(new_train_data_all_x).view(-1, 28, 28),
                                     torch.tensor(new_train_data_all_y), transform=data_transform)
    trainloaders_iid = [torch.utils.data.DataLoader(
        train_dataset_iid,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx[i]))
        for i in range(len(train_idx))]

    # test: single loader
    # train: single loader
    # train_s: loaders
    # test_s: loaders
    # train_s_iid: loaders with iid data
    return {"test": testloader, "train_s": trainloaders,
            "test_s": testloaders, "train": trainloader,
            "train_s_iid": trainloaders_iid}
    # return {"test": copy.deepcopy(testloader), "train": copy.deepcopy(trainloader)}


def len_to_index(train_idx):
    l = list(range(sum(train_idx)))
    ll = []
    for i in train_idx:
        ll.append(l[:i])
        l = l[i:]
    return ll


class SHDataset(Dataset):
    def __init__(self, feature, target, transform=None):
        self.X = torch.tensor(feature)
        self.Y = torch.tensor(target)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def shakespeare_dataloaders(root="./data/femnist", batch_size=128, clients=10):
    train_data = torch.load(os.path.join(root, "train_data.pt"))
    test_data = torch.load(os.path.join(root, "test_data.pt"))
    # ['users', 'num_samples', 'user_data', 'hierarchies']
    if clients > len(train_data['users']):
        raise ValueError(
            "Request clients({}) larger then dataset provide({}).".format(clients, len(train_data['users'])))
    train_data['users'] = train_data['users'][:clients]
    test_data['users'] = test_data['users'][:clients]
    #############################################################################
    train_data_all_x = []
    train_data_all_y = []
    train_idx = []
    for i in train_data["users"]:
        train_data_all_x += [word_to_indices(sen) for sen in train_data["user_data"][i]["x"]]
        train_data_all_y += [word_to_indices(sen)[0] for sen in train_data["user_data"][i]["y"]]
        train_idx.append(len(train_data["user_data"][i]["y"]))

    train_dataset = SHDataset(train_data_all_x, train_data_all_y)

    train_idx = len_to_index(train_idx)
    trainloaders = [torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx[i]))
        for i in range(len(train_idx))]

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #############################################################################
    test_data_all_x = []
    test_data_all_y = []
    test_idx = []
    for i in test_data["users"]:
        test_data_all_x += [word_to_indices(sen) for sen in test_data["user_data"][i]["x"]]
        test_data_all_y += [word_to_indices(sen)[0] for sen in test_data["user_data"][i]["y"]]
        test_idx.append(len(test_data["user_data"][i]["y"]))

    test_dataset = SHDataset(test_data_all_x, test_data_all_y)

    test_idx = len_to_index(test_idx)
    testloaders = [torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_idx[i]))
        for i in range(len(test_idx))]

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # iid dataloaders
    idx = list(range(len(train_data_all_x)))
    random.shuffle(idx)
    new_train_data_all_x = [train_data_all_x[i] for i in idx]
    new_train_data_all_y = [train_data_all_y[i] for i in idx]
    train_dataset_iid = SHDataset(new_train_data_all_x, new_train_data_all_y)
    trainloaders_iid = [torch.utils.data.DataLoader(
        train_dataset_iid,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx[i]))
        for i in range(len(train_idx))]

    # test: single loader
    # train: single loader
    # train_s: loaders
    # test_s: loaders
    # train_s_iid: loaders with iid data
    return {"test": testloader, "train_s": trainloaders,
            "test_s": testloaders, "train": trainloader,
            "train_s_iid": trainloaders_iid}


def word_to_indices(word):
    ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


#################################################################
def DATALOADER(config: Configer = None):
    # DATALOADER_TABLE = ["cifar10", "femnist", "shakespeare"]
    if config is None:
        raise ValueError("config shouldn't be none")
    if "cifar10" in config.trainer.get_dataset_path():
        dataloaders = cifar_dataloaders(root=config.trainer.get_dataset_path(),
                                        index_path=os.path.join(config.trainer.get_dataset_path(),
                                                                "datatype", "index.json"),
                                        batch_size=config.trainer.get_local_bs())
    elif "femnist" in config.trainer.get_dataset_path():
        dataloaders = femnist_dataloaders(root=config.trainer.get_dataset_path(),
                                          batch_size=config.trainer.get_local_bs(),
                                          clients=config.general.get_nodes())
    elif "shakespeare" in config.trainer.get_dataset_path():
        dataloaders = shakespeare_dataloaders(root=config.trainer.get_dataset_path(),
                                              batch_size=config.trainer.get_local_bs(),
                                              clients=config.general.get_nodes())

    print("Total train images: {}".format(len(dataloaders["train"].dataset)))
    print("Total test images: {}".format(len(dataloaders["test"].dataset)))
    return dataloaders
