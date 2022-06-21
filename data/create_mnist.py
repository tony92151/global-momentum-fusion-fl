import pickle
import os, json
import random, copy
import sys
import time

import numpy as np
import argparse
from shutil import copyfile
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler
import numpy as np

sys.path.append("../")
from utils.weight_divergence.emd import earth_moving_distance
from utils.dataloaders import mnist_dataloaders


def set_seed(value):
    np.random.seed(value)
    random.seed(value)


def unpickle(files):
    data = []
    data_c = [[] for i in range(10)]
    data_index = [[] for i in range(10)]
    idx = 0
    for f in files:
        print("Read data: {}".format(f))
        with open(f, 'rb') as fo:
            d = pickle.load(fo, encoding='latin1')

        for i in range(len(d['labels'])):
            data.append(d['labels'][i])
            data_c[d['labels'][i]].append((d['labels'][i], d['data'][i], d['filenames'][i], idx))
            idx += 1

    for d in range(len(data)):
        data_index[data[d]].append(d)

    # data_c: [[(labels, data, filenames, idx),(),...], [(labels, data, filenames, idx),(),...],... (10 classes)]
    #           |______________class0_______________|    |______________class1_______________|
    # data_index: [[21,74,6,787,4741,...], [211,7,64,987,8774,...],...(10 classes)]
    #              |_______class0_______|  |_______class1_______|
    return data_c, data_index


def pickling(f, value):
    with open(f, 'wb') as fo:
        pickle.dump(value, fo)


def divide_part(value, parts):
    # [value1, value2, ...]
    vlaue = copy.deepcopy(value)
    # random.shuffle(value)
    c = 0
    d = [[] for i in range(parts)]
    for i in vlaue:
        d[c % parts].append(i)
        c += 1
    random.shuffle(d)
    # [[value0, value21, value1876,...], [value7527, value123, value74,...],...(%n parts)]
    #   |___________part0____________|    |____________part1____________|
    return d


cifar_path = ["cifar-10-batches-py/data_batch_1",
              "cifar-10-batches-py/data_batch_2",
              "cifar-10-batches-py/data_batch_3",
              "cifar-10-batches-py/data_batch_4",
              "cifar-10-batches-py/data_batch_5"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20, help="Separate to n datasets")
    parser.add_argument('--data', type=str, default="./mnist", help="Path to dataset")
    parser.add_argument('--save', type=str, default="test", help="Path to test dataset")
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--num_shards', type=int, default=100, help="Default num_shards=100. Smaller num_shards will "
                                                                    "make the dataset more non-iid.[200,100,80,50,40,"
                                                                    "20]")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('-f')
    args = parser.parse_args()

    path = os.path.abspath(os.path.abspath("./mnist"))
    number_of_client = args.n

    if args.download:
        if os.path.exists(path):
            raise ValueError("Folder: \"{}\" exist".format(path))
        os.makedirs(path)
        import gdown, tarfile

        # download
        dataset = torchvision.datasets.MNIST(root=path, train=True, download=True)
        url = 'https://drive.google.com/uc?id=1ZdpVyGLOMr23kwf-1xpRgDvBLsaVTaZf'
        output = os.path.join(path, 'mnist_c40_6test.tar.gz')
        print("\nDownload ...")
        gdown.download(url, output, quiet=False)
        # check
        md5 = '9589546ba333305fa5e166dbaf68f575'
        gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)
        time.sleep(3)
        # extraction
        print("\nExtracting ...")
        tar = tarfile.open(output, 'r:gz')
        tar.extractall()

        time.sleep(1)
        # print data
        # file_ = open(os.path.join(path, "niid", "index.json"), 'r')
        # context_niid = json.load(file_)
        # file_.close()

        # for j in range(len(context_niid.keys())):
        #     ans = [0 for i in range(10)]
        #     for i in context_niid[str(j)]:
        #         ans[dataset.targets[i]] += 1
        #     print("client: {} , {}, sum: {}".format(j, ans, sum(ans)))
        emds = []
        for i in range(0, 6):
            cdataloders = mnist_dataloaders(root="./mnist",
                                            index_path="./mnist/test{}/index.json".format(i),
                                            show=False)
            emds.append(earth_moving_distance(dataloaders=cdataloders["train_s"], number_of_class=10))

        # print("\nEarth moving distance: ", emd)
        print("\n{:<10} {:<25}".format('Dataset', 'Earth moving distance'))
        for i, val in enumerate(emds):
            print("{:<10} {:<25}".format('test{}'.format(i), round(val, 2)))

        exit()

    path = os.path.abspath(args.data)

    # Download
    transform_train = transforms.Compose([transforms.ToTensor()])
    train_datasets = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform_train)


    classes = list(set(train_datasets.targets.tolist()))
    targets = train_datasets.targets.tolist()
    idx_list = [i for i in range(len(targets))]
    datas = []
    data_class = [[] for i in range(len(classes))]

    for i in classes:
        for l, idx in zip(targets, idx_list):
            if l == i:
                data_class[i].append((l, idx))
                datas.append((l, idx))

    # data: [[(labels, idx),(),...], [(labels, idx),(),...],... (10 classes)]
    #         |______class0_______|   |______class1_______|


    ######
    set_seed(args.seed)
    ######

    ############################################################
    # iid save
    for i in data_class:
        random.shuffle(i)
        print(len(i))

    data_class_sep = [divide_part(i, number_of_client) for i in data_class]

    packages = [[] for _ in range(number_of_client)]
    for i in range(number_of_client):
        for data in data_class_sep:
            info = data.pop(0)
            info = [p[1] for p in info]
            packages[i] += info
    print(len(packages))
    clients_json = {}
    for i, info in enumerate(packages):
        clients_json["{}".format(i)] = info
    save_path = os.path.join(path, "test0", "index.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fo:
        json.dump(clients_json, fo)

    ############################################################


    num_shards = args.num_shards
    number_of_client= args.n
    ############################################################
    # Total 60000 images
    num_shards, num_imgs = num_shards, int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    ############################################################

    packages = []

    for i in range(number_of_client):
        list_of_containt = []

        rand_set = set(np.random.choice(idx_shard, int(num_shards / number_of_client), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            list_of_containt = list_of_containt + datas[rand * num_imgs:(rand + 1) * num_imgs]

        # random.shuffle(list_of_containt)
        packages.append(list_of_containt)

    clients_json = {}

    for i,info in enumerate(packages):
        clients_json["{}".format(i)] = [p[1] for p in info]


    # 
    save_path = os.path.join(path, args.save, "index.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fo:
        json.dump(clients_json, fo)

    # print data
    total_used = 0
    for j in range(number_of_client):
        ans = [0 for i in range(10)]
        for i in packages[int(j)]:
            ans[i[0]] += 1
        total_used += sum(ans)
        print("client: {} ".format(j), ans, sum(ans))
    print("Image used: {}/{} ".format(total_used,len(targets)))
    
    #############################################
    #############################################

    context = []
    for i in packages:
        l = [idx[1] for idx in i]
        context.append(l)

    trainloaders = []
    for c in context:
        trainloaders.append(torch.utils.data.DataLoader(train_datasets,
                                                batch_size=10,
                                                sampler=SubsetRandomSampler(c)))
    
    emd = earth_moving_distance(dataloaders=trainloaders, number_of_class=number_of_client)
    print("\nEarth moving distance: ", emd)
