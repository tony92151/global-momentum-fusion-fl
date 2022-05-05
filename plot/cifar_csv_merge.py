import argparse
import os
import pandas as pd
from tqdm import tqdm

EMD = [0, 0.48, 0.76, 0.87, 0.99, 1.18, 1.35]


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

    for i in tqdm(range(7)):
        # acc
        pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test{}_train_acc.csv".format(i)), index_col=0)
        pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test{}_train_acc.csv".format(i)), index_col=0)
        result = {}
        for tag in pd_csv1:
            result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
        pd.DataFrame(result).to_csv(
            os.path.join(args.output_csv, "test{}_train_acc.csv".format(i)))

        pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test{}_test_acc.csv".format(i)), index_col=0)
        pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test{}_test_acc.csv".format(i)), index_col=0)
        result = {}
        for tag in pd_csv1:
            result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
        pd.DataFrame(result).to_csv(
            os.path.join(args.output_csv, "test{}_test_acc.csv".format(i)))

        # loss
        pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test{}_train_loss.csv".format(i)), index_col=0)
        pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test{}_train_loss.csv".format(i)), index_col=0)
        result = {}
        for tag in pd_csv1:
            result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
        pd.DataFrame(result).to_csv(
            os.path.join(args.output_csv, "test{}_train_loss.csv".format(i)))

        pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test{}_test_loss.csv".format(i)), index_col=0)
        pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test{}_test_loss.csv".format(i)), index_col=0)
        result = {}
        for tag in pd_csv1:
            result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
        pd.DataFrame(result).to_csv(
            os.path.join(args.output_csv, "test{}_test_loss.csv".format(i)))

    # traffic
    for i in tqdm(range(7)):
        pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test{}_traffic.csv".format(i)), index_col=0)
        pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test{}_traffic.csv".format(i)), index_col=0)
        result = {}
        for tag in pd_csv1:
            result[tag] = array_avg(pd_csv1[tag].tolist(), pd_csv2[tag].tolist())
        pd.DataFrame(result).to_csv(
            os.path.join(args.output_csv, "test{}_traffic.csv".format(i)))
