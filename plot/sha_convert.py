import argparse
import os
import pandas as pd
from tqdm import tqdm

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

    for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
        DGC_target.append("sha_1l_DGC_cr{}".format(cr))
        GMC_target.append("sha_1l_GMC_cr{}".format(cr))
        DGCwGMF_target.append("sha_1l_GFDGC_cr{}".format(cr))
        DGCwGM_target.append("sha_1l_DGCwGM_cr{}".format(cr))

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


    dict_train_acc = {}
    dict_train_loss = {}

    dict_test_acc = {}
    dict_test_loss = {}

    # train
    for cr in tqdm([0.1, 0.3, 0.5, 0.7, 0.9]):
        train_acc, train_loss = data_avg(DGC_target_info["sha_1l_DGC_cr{}".format(cr)]["train"])
        dict_train_acc["DGC({})".format(cr)] = train_acc
        dict_train_loss["DGC({})".format(cr)] = train_loss

        train_acc, train_loss = data_avg(GMC_target_info["sha_1l_GMC_cr{}".format(cr)]["train"])
        dict_train_acc["GMC({})".format(cr)] = train_acc
        dict_train_loss["GMC({})".format(cr)] = train_loss

        train_acc, train_loss = data_avg(DGCwGMF_target_info["sha_1l_GFDGC_cr{}".format(cr)]["train"])
        dict_train_acc["DGCwGMF({})".format(cr)] = train_acc
        dict_train_loss["DGCwGMF({})".format(cr)] = train_loss

        train_acc, train_loss = data_avg(DGCwGM_target_info["sha_1l_DGCwGM_cr{}".format(cr)]["train"])
        dict_train_acc["DGCwGM({})".format(cr)] = train_acc
        dict_train_loss["DGCwGM({})".format(cr)] = train_loss

        # test
        info_data = DGC_target_info["sha_1l_DGC_cr{}".format(cr)]["test"]
        train_acc = [info_data[i]["acc"] for i in info_data]
        train_loss = [info_data[i]["loss"] for i in info_data]
        dict_test_acc["DGC({})".format(cr)] = train_acc
        dict_test_loss["DGC({})".format(cr)] = train_loss

        info_data = GMC_target_info["sha_1l_GMC_cr{}".format(cr)]["test"]
        train_acc = [info_data[i]["acc"] for i in info_data]
        train_loss = [info_data[i]["loss"] for i in info_data]
        dict_test_acc["GMC({})".format(cr)] = train_acc
        dict_test_loss["GMC({})".format(cr)] = train_loss

        info_data = DGCwGMF_target_info["sha_1l_GFDGC_cr{}".format(cr)]["test"]
        train_acc = [info_data[i]["acc"] for i in info_data]
        train_loss = [info_data[i]["loss"] for i in info_data]
        dict_test_acc["DGCwGMF({})".format(cr)] = train_acc
        dict_test_loss["DGCwGMF({})".format(cr)] = train_loss

        info_data = DGCwGM_target_info["sha_1l_DGCwGM_cr{}".format(cr)]["test"]
        train_acc = [info_data[i]["acc"] for i in info_data]
        train_loss = [info_data[i]["loss"] for i in info_data]
        dict_test_acc["DGCwGM({})".format(cr)] = train_acc
        dict_test_loss["DGCwGM({})".format(cr)] = train_loss

    # save
    pd.DataFrame(dict_train_acc).to_csv(os.path.join(args.output_csv, "train_acc.csv"))
    pd.DataFrame(dict_train_loss).to_csv(os.path.join(args.output_csv, "train_loss.csv"))
    pd.DataFrame(dict_test_acc).to_csv(os.path.join(args.output_csv, "test_acc.csv"))
    pd.DataFrame(dict_test_loss).to_csv(os.path.join(args.output_csv, "test_loss.csv"))
