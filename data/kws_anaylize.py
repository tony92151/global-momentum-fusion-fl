import glob
import pickle
import os, json
import random, copy
import sys
import time
import numpy as np


def earth_moving_distance(data_statistics, number_of_class=10):
    # data_statistics = [
    #  [11,56,23,...10],   -> Dataloader1 have 11 images in class 0 ...
    #  [31,51,13,...1],    -> Dataloader2 have 31 images in class 0 ...
    #  ...
    #  ]
    # data_statistics = [[0 for _ in range(number_of_class)] for _ in dataloaders]
    #
    # for i, v in enumerate(dataloaders):
    #     for x, y in dataloaders[i]:
    #         for target in y:
    #             data_statistics[i][int(target)] += 1

    # data_statistics_sum = [1341,5226,1234,...3210],   -> Total have 1341 images in class 0 ...
    data_statistics_sum = np.array(data_statistics).sum(axis=0).tolist()
    total_data = sum(data_statistics_sum)

    def P(data_statistic, total_mount_data, c: int):
        p_tmp = []
        for data in data_statistic:
            p_tmp.append((sum(data) / total_mount_data) * (data[c] / sum(data)))
        return sum(p_tmp)

    def PK(data_statistic, c: int, k: int):
        return data_statistic[k][c] / sum(data_statistic[k])

    emd_in_each_loader = []
    for k in range(len(data_statistics)):
        e_tmp = []
        for c in range(number_of_class):
            e_tmp.append(abs(PK(data_statistic=data_statistics, c=c, k=k) - P(data_statistic=data_statistics,
                                                                              total_mount_data=total_data, c=c)))
        emd_in_each_loader.append(sum(e_tmp))

    return sum(emd_in_each_loader) / len(emd_in_each_loader)


classes = ["hey_android",
           "hey_snapdragon",
           "hi_galaxy",
           "hi_lumina"]

# label = 0,1,2,3
if __name__ == '__main__':
    # path = "/Users/tonyguo-mba/Downloads/qualcomm_keyword_speech_dataset"
    # users = set()
    # for c in classes:
    #     users = users.union(set([n.split("/")[-1] for n in glob.glob(os.path.join(path, c, "*"))]))
    # users = list(users)
    #
    # # find all .wav file
    # wav_file = set()
    # for c in classes:
    #     for u in users:
    #         if os.path.isdir(os.path.join(path, c, u)):
    #             wav_file = wav_file.union(set(glob.glob(os.path.join(path, c, u, "*.wav"))))
    # wav_file = list(wav_file)
    #
    # # data = {
    # #   "users": ["1", "2",...],
    # #   "num_samples": [12,34,26,8,...]
    # #   "user_data": {"0": {"x":[[0.49, 1.56,...],[0.59, 3.56,...],... ]
    # #                       "y":[1,6,2,...]
    # #                       },
    # #                 "1":......
    # #                 }
    # # }
    #
    # data = {
    #     "users": [],
    #     "num_samples": [],
    #     "user_data": {},
    # }
    #
    # for u in users:
    #     count = 0
    #     for f in wav_file:
    #         if u in os.path.dirname(f):
    #             count += 1
    #
    #     labels = []
    #     for f in wav_file:
    #         for i, c in enumerate(classes):
    #             if c in f and u in os.path.dirname(f):
    #                 labels.append(i)
    #                 break
    #     # print(labels)
    #     data["users"].append(u)
    #     data["num_samples"].append(count)
    #
    #     # udata = {"x": [f.replace(path, "") for f in wav_file], "y": labels}
    #     udata = {u: {"y": labels}}
    #     data["user_data"].update(udata)

    # print(data)

    # del data
    with open(sys.argv[1], "r") as file:
        data = json.load(file)

    data_statistics = []

    for k in data["user_data"].keys():
        print(data["user_data"][k]["y"])


    for k in data["user_data"].keys():
        clist = []
        for i in range(20):
            d = np.array(data["user_data"][k]["y"])
            clist.append(len(d[d == i]))
        data_statistics.append(clist)

    print(data_statistics)
    # print(earth_moving_distance(data_statistics, 20))
