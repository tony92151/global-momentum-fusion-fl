import argparse
import os
import pandas as pd
from tqdm import tqdm


def array_avg(l1, l2):
    assert len(l1) == len(l2)
    return [round((i1 + i2) / 2, 5) for i1, i2 in zip(l1, l2)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv1', help="path to csv1", type=str, default=None, required=True)
    parser.add_argument('--csv2', help="path to csv1", type=str, default=None, required=True)
    parser.add_argument('--output_csv', help="path to output", type=str, default=None, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_csv, exist_ok=True)

    # acc
    pd_csv1 = pd.read_csv(os.path.join(args.csv1, "train_acc.csv"), index_col=0)
    pd_csv2 = pd.read_csv(os.path.join(args.csv2, "train_acc.csv"), index_col=0)
    result = {}
    for tag in pd_csv1:
        result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
    pd.DataFrame(result).to_csv(
        os.path.join(args.output_csv, "train_acc.csv"))

    pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test_acc.csv"), index_col=0)
    pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test_acc.csv"), index_col=0)
    result = {}
    for tag in pd_csv1:
        result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
    pd.DataFrame(result).to_csv(
        os.path.join(args.output_csv, "test_acc.csv"))

    # loss
    pd_csv1 = pd.read_csv(os.path.join(args.csv1, "train_loss.csv"), index_col=0)
    pd_csv2 = pd.read_csv(os.path.join(args.csv2, "train_loss.csv"), index_col=0)
    result = {}
    for tag in pd_csv1:
        result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
    pd.DataFrame(result).to_csv(
        os.path.join(args.output_csv, "train_loss.csv"))

    pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test_loss.csv"), index_col=0)
    pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test_loss.csv"), index_col=0)
    result = {}
    for tag in pd_csv1:
        result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
    pd.DataFrame(result).to_csv(
        os.path.join(args.output_csv, "test_loss.csv"))

    # traffic
    for i in tqdm(range(7)):
        pd_csv1 = pd.read_csv(os.path.join(args.csv1, "traffic.csv"), index_col=0)
        pd_csv2 = pd.read_csv(os.path.join(args.csv2, "traffic.csv"), index_col=0)
        result = {}
        for tag in pd_csv1:
            result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
        pd.DataFrame(result).to_csv(
            os.path.join(args.output_csv, "traffic.csv"))
