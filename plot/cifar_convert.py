import argparse
import os
import pandas as pd
from tqdm import tqdm
EMD = [0, 0.48, 0.76, 0.87, 0.99, 1.18, 1.35]


def recursive_find(path=None, file_filter=["png", "bmp", "jpg", "jpeg"]):
    all_image_path = []
    if os.path.isdir(path) and path is not None:
        for dirPath, dirNames, fileNames in os.walk(path):
            for f in fileNames:
                all_image_path.append(os.path.join(dirPath, f))

    filter_image_path = []  # version, name
    filter_image_name = []

    for p in all_image_path:
        if ("checkpoints" not in p) and (p.split("/")[-1].split(".")[1] in file_filter):
            filter_image_path.append(os.path.relpath(os.path.dirname(p), os.path.dirname(path)))
            filter_image_name.append(p.split("/")[-1])

    return [filter_image_path, filter_image_name]  # version, name


def data_avg(data):
    accs = []
    losses = []
    for i in data:
        d = data[str(i)]
        acc = []
        loss = []
        for a in d:
            acc.append(d[str(a)]["acc"])
            loss.append(d[str(a)]["loss"])

        accs.append(round(sum(acc) / len(acc), 4))
        losses.append(round(sum(loss) / len(loss), 5))
    return accs, losses


def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trensorboard_path', help="path to trensorboard", type=str, default=None, required=True)
    parser.add_argument('--output_csv', help="path to output", type=str, default=None, required=True)
    args = parser.parse_args()

    assert args.trensorboard_path is not None
    assert args.output_csv is not None

    os.makedirs(args.output_csv, exist_ok=True)

    DGC_target = []
    GMC_target = []
    DGCwGMF_target = []
    DGCwGM_target = []

    for t in range(7):
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            DGC_target.append("test{}_cifar10_r56_DGC_cr{}".format(t, cr))

    for t in range(7):
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            GMC_target.append("test{}_cifar10_r56_GMC_cr{}".format(t, cr))

    for t in range(7):
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            DGCwGMF_target.append("test{}_cifar10_r56_GFDGC_cr{}".format(t, cr))

    for t in range(7):
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            DGCwGM_target.append("test{}_cifar10_r56_DGCwGM_cr{}".format(t, cr))

    # read
    import json

    json_path_list = recursive_find(args.trensorboard_path, file_filter=["json"])

    DGC_target_info = {}
    GMC_target_info = {}
    DGCwGMF_target_info = {}
    DGCwGM_target_info = {}
    for p, f in zip(json_path_list[0], json_path_list[1]):
        with open(os.path.join(p, f), newline='') as jsonfile:
            data = json.load(jsonfile)
        if list(data.keys())[0] in DGC_target:
            DGC_target_info.update(data)
        if list(data.keys())[0] in GMC_target:
            GMC_target_info.update(data)
        if list(data.keys())[0] in DGCwGMF_target:
            DGCwGMF_target_info.update(data)
        if list(data.keys())[0] in DGCwGM_target:
            DGCwGM_target_info.update(data)

    for i in tqdm(range(7)):

        dict_train_acc = {}
        dict_train_loss = {}

        dict_test_acc = {}
        dict_test_loss = {}

        # train
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            train_acc, train_loss = data_avg(
                DGC_target_info["test{}_cifar10_r56_DGC_cr{}".format(i, cr)]["train"])
            dict_train_acc["DGC({})".format(cr)] = train_acc
            dict_train_loss["DGC({})".format(cr)] = train_loss

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            train_acc, train_loss = data_avg(
                GMC_target_info["test{}_cifar10_r56_GMC_cr{}".format(i, cr)]["train"])
            dict_train_acc["GMC({})".format(cr)] = train_acc
            dict_train_loss["GMC({})".format(cr)] = train_loss

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            train_acc, train_loss = data_avg(
                DGCwGMF_target_info["test{}_cifar10_r56_GFDGC_cr{}".format(i, cr)]["train"])
            dict_train_acc["DGCwGMF({})".format(cr)] = train_acc
            dict_train_loss["DGCwGMF({})".format(cr)] = train_loss

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            train_acc, train_loss = data_avg(
                DGCwGM_target_info["test{}_cifar10_r56_DGCwGM_cr{}".format(i, cr)]["train"])
            dict_train_acc["DGCwGM({})".format(cr)] = train_acc
            dict_train_loss["DGCwGM({})".format(cr)] = train_loss

        # test
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            info_data = DGC_target_info["test{}_cifar10_r56_DGC_cr{}".format(i, cr)]["test"]
            train_acc = [info_data[i]["acc"] for i in info_data]
            train_loss = [info_data[i]["loss"] for i in info_data]
            dict_test_acc["DGC({})".format(cr)] = train_acc
            dict_test_loss["DGC({})".format(cr)] = train_loss

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            info_data = GMC_target_info["test{}_cifar10_r56_GMC_cr{}".format(i, cr)]["test"]
            train_acc = [info_data[i]["acc"] for i in info_data]
            train_loss = [info_data[i]["loss"] for i in info_data]
            dict_test_acc["GMC({})".format(cr)] = train_acc
            dict_test_loss["GMC({})".format(cr)] = train_loss

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            info_data = DGCwGMF_target_info["test{}_cifar10_r56_GFDGC_cr{}".format(i, cr)]["test"]
            train_acc = [info_data[i]["acc"] for i in info_data]
            train_loss = [info_data[i]["loss"] for i in info_data]
            dict_test_acc["DGCwGMF({})".format(cr)] = train_acc
            dict_test_loss["DGCwGMF({})".format(cr)] = train_loss

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            info_data = DGCwGM_target_info["test{}_cifar10_r56_DGCwGM_cr{}".format(i, cr)]["test"]
            train_acc = [info_data[i]["acc"] for i in info_data]
            train_loss = [info_data[i]["loss"] for i in info_data]
            dict_test_acc["DGCwGM({})".format(cr)] = train_acc
            dict_test_loss["DGCwGM({})".format(cr)] = train_loss


        # save
        pd.DataFrame(dict_train_acc).to_csv(os.path.join(args.output_csv, "test{}_train_acc.csv".format(i)))
        pd.DataFrame(dict_train_loss).to_csv(os.path.join(args.output_csv, "test{}_train_loss.csv".format(i)))
        pd.DataFrame(dict_test_acc).to_csv(os.path.join(args.output_csv, "test{}_test_acc.csv".format(i)))
        pd.DataFrame(dict_test_loss).to_csv(os.path.join(args.output_csv, "test{}_test_loss.csv".format(i)))
