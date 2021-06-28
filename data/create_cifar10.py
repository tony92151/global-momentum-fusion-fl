import pickle
import os, json
import random, copy
import numpy as np
import argparse
from shutil import copyfile

import torchvision


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
    parser.add_argument('--data', type=str, default="./cifar10")
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('-f')
    args = parser.parse_args()

    path = os.path.abspath(args.data)
    number_of_client = args.n

    # Download
    _ = torchvision.datasets.CIFAR10(root=path, train=True, download=True)

    paths = [os.path.join(path, i) for i in cifar_path]
    data, data_index = unpickle(paths)

    # data: [[(labels, data, filenames, idx),(),...], [(labels, data, filenames, idx),(),...],... (10 classes)]
    #         |______________class0_______________|    |______________class1_______________|
    # data_index: [[21,74,6,787,4741,...], [211,7,64,987,8774,...],...(10 classes)]
    #              |_______class0_______|  |_______class1_______|

    os.makedirs(os.path.join(path, "iid"), exist_ok=True)
    os.makedirs(os.path.join(path, "niid"), exist_ok=True)

    k = {'batch_label': '', 'labels': [], 'data': [], 'filenames': []}

    chunks = [[] for i in range(number_of_client)]

    # for d in data_index:
    for d in data:
        dp = divide_part(d, number_of_client)

        for i in range(len(dp)):
            chunks[i] = chunks[i] + dp[i]

    clients_json = {}

    for i in range(len(chunks)):
        clients_json[str(i)] = [d[3] for d in chunks[i]]

    with open(os.path.join(path, "iid", "index.json"), 'w') as fo:
        json.dump(clients_json, fo)

    # for i in range(len(chunks)):
    #     k_ = copy.deepcopy(k)
    #     for d in chunks[i]:
    #         k_["labels"].append(d[0])
    #         k_["data"].append(d[1])
    #         k_["filenames"].append(d[2])
    #     pickling(os.path.join(path, "iid", "cifar10_{}.pkl".format(i)), k_)
    #     print("Save result: {}".format(os.path.join(path, "iid", "cifar10_{}.pkl".format(i))))

    # copyfile(os.path.join(path, "cifar-10-batches-py", "test_batch"), os.path.join(path, "iid", "cifar10_test.pkl"))

    ############################################################
    # Total 50000 images
    num_shards, num_imgs = 100, 500
    idx_shard = [i for i in range(num_shards)]
    ############################################################
    datas = []
    for d in data:
        datas = datas + d
    datas = [(d[0], d[3]) for d in datas]  # only label and index

    packages = []

    for i in range(number_of_client):
        list_of_containt = []

        rand_set = set(np.random.choice(idx_shard, int(num_shards / number_of_client), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            list_of_containt = list_of_containt + datas[rand * num_imgs:(rand + 1) * num_imgs]

        # random.shuffle(list_of_containt)
        packages.append(list_of_containt)

    lients_json = {}

    for i in range(len(packages)):
        clients_json["{}".format(i)] = [p[1] for p in packages[i]]

    with open(os.path.join(path, "niid", "index.json"), 'w') as fo:
        json.dump(clients_json, fo)

    # print data
    for j in range(number_of_client):
        ans = [0 for i in range(10)]
        for i in packages[int(j)]: 
            ans[i[0]] +=1
        print("client: {} ".format(j), ans, sum(ans))
