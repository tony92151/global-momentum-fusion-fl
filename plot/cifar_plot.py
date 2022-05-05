import argparse
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

plt.rcParams["font.family"] = "Times New Roman"

EMD = [0, 0.48, 0.76, 0.87, 0.99, 1.18, 1.35]


def array_avg(l1, l2):
    assert len(l1) == len(l2)
    return [round((i1 + i2) / 2, 5) for i1, i2 in zip(l1, l2)]


def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


def save_both(data, test, CR, type=0, output="./plot", smooth_rate=0.5):
    datas = data["train"]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(12, 8))
    epoch = list(range(220))
    plt.plot(epoch,
             smooth(data["test"]["DGC"], smooth_rate),
             linewidth=2, markersize=10, color="#2079B6", label='(test)DGC')
    plt.plot(epoch,
             smooth(data["test"]["GMC"], smooth_rate),
             linewidth=2, markersize=10, color="#F77714", label='(test)GMC')
    plt.plot(epoch,
             smooth(data["test"]["DGCwGM"], smooth_rate),
             linewidth=2, markersize=10, color="#F44336", label='(test)DGCwGM')
    plt.plot(epoch,
             smooth(data["test"]["DGCwGMF"], smooth_rate),
             linewidth=2, markersize=10, color="#07AE06", label='(test)DGCwGMF')
    # plt.plot(epoch,
    #          smooth(data["test"]["DGCwGM"], smooth_rate),
    #          linewidth=2, markersize=10, color="#F44336", label='(test)DGCwGM')

    plt.plot(epoch,
             smooth(data["train"]["DGC"], smooth_rate),
             linewidth=2, markersize=10, linestyle='--', color="#2079B6", label='(train)DGC')
    plt.plot(epoch,
             smooth(data["train"]["GMC"], smooth_rate),
             linewidth=2, markersize=10, linestyle='--', color="#F77714", label='(train)GMC')
    plt.plot(epoch,
             smooth(data["train"]["DGCwGM"], smooth_rate),
             linewidth=2, markersize=10, linestyle='--', color="#F44336", label='(train)DGCwGM')
    plt.plot(epoch,
             smooth(data["train"]["DGCwGMF"], smooth_rate),
             linewidth=2, markersize=10, linestyle='--', color="#07AE06", label='(train)DGCwGMF')
    # plt.plot(epoch,
    #          smooth(data["train"]["DGCwGM"], smooth_rate),
    #          linewidth=2, markersize=10, linestyle='--', color="#F44336", label='(train)DGCwGM')

    if type == 0:
        plt.xlim(0, 220)
        plt.ylim(0.1, 1.0)
        plt.title('Top-1 Accuracy of Mod-Cifar10-{} (compression rate {})'.format(test, CR), fontsize=20, pad=10)
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Top-1 Accuracy', fontsize=20)
        plt.legend(loc='lower right', fontsize=16)
    elif type == 1:
        plt.xlim(0, 220)
        plt.ylim(0, 3.0)
        plt.title('Loss of Mod-Cifar10-{} (compression rate {})'.format(test, CR), fontsize=20, pad=10)
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.legend(loc='upper right', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid()
    save_type = "acc" if type == 0 else "loss"
    save_pdf_path = os.path.join(output, "figure", save_type, "mix_{}_plots_{}_{}.pdf".format(test, CR, save_type))
    os.makedirs(os.path.dirname(save_pdf_path), exist_ok=True)
    plt.savefig(save_pdf_path, transparent=True)


def save_baseline(data, test, output="./plot", smooth_rate=0.5):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True

    x = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.xlim(0.9, 0.1)
    plt.ylim(0.5, 0.85)

    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_xlabel('Compression rate', fontsize=20)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    ax1.plot(x, smooth(data["test"]["DGC"], smooth_rate),
             '-', linewidth=3, markersize=6, marker="o", color="#2079B6", label='DGC')
    ax1.plot(x, smooth(data["test"]["GMC"], smooth_rate),
             '-', linewidth=3, markersize=6, marker="s", color="#F77714", label='GMC')
    ax1.plot(x, smooth(data["test"]["DGCwGM"], smooth_rate),
             '-', linewidth=3, markersize=8, marker=".", color="#F44336", label='DGCwGM')
    ax1.plot(x, smooth(data["test"]["DGCwGMF"], smooth_rate),
             '-', linewidth=3, markersize=6, marker="^", color="#07AE06", label='DGCwGMF')
    # ax1.plot(x, smooth(data["test"]["DGCwGM"], smooth_rate),
    #          '-', linewidth=3, markersize=8, marker=".", color="#F44336", label='DGCwGM')

    ax1.grid()
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Communication overhead $(10^{9})$', fontsize=16)
    ax2.tick_params(axis='y', labelsize=16)

    ax2.plot(x, smooth(data["traffic"]["DGC"], smooth_rate),
             '--', linewidth=3, markersize=6, marker="o", color="#2079B6", label='DGC')
    ax2.plot(x, smooth(data["traffic"]["GMC"], smooth_rate),
             '--', linewidth=3, markersize=6, marker="s", color="#F77714", label='GMC')
    ax2.plot(x, smooth(data["traffic"]["DGCwGM"], smooth_rate),
             '--', linewidth=3, markersize=8, marker=".", color="#F44336", label='DGCwGM')
    ax2.plot(x, smooth(data["traffic"]["DGCwGMF"], smooth_rate),
             '--', linewidth=3, markersize=6, marker="^", color="#07AE06", label='DGCwGMF')
    # ax2.plot(x, smooth(data["traffic"]["DGCwGM"], smooth_rate),
    #          '--', linewidth=3, markersize=8, marker=".", color="#F44336", label='DGCwGM')

    plt.title('Mod-Cifar10-{} Dataset'.format(str(test)), fontsize=20, pad=10)
    plt.xlabel('Compression rate', fontsize=20)

    plt.xticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    leg1 = ax1.legend(loc='lower left', fontsize=16)
    leg1.set_title('Accuracy', prop={'size': 16})

    leg2 = ax2.legend(loc=(0.23, 0.018), fontsize=16)
    leg2.set_title('Communication overhead', prop={'size': 16})

    save_pdf_path = os.path.join(output, "figure", "baseline", "baseline_{}_plots.pdf".format(test))
    os.makedirs(os.path.dirname(save_pdf_path), exist_ok=True)
    plt.savefig(save_pdf_path, transparent=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help="path to csv", type=str, default=None, required=True)
    parser.add_argument('--output', help="path to output", type=str, default=None, required=True)
    parser.add_argument('--smooth_rate', type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    smooth_rate = args.smooth_rate
    # for i in tqdm(range(7)):
    #     pd_csv1 = pd.read_csv(os.path.join(args.csv1, "test{}_train_acc.csv".format(i)), index_col=0)
    #     pd_csv2 = pd.read_csv(os.path.join(args.csv2, "test{}_train_acc.csv".format(i)), index_col=0)

    print("====== Plot acc & loss ======")
    time.sleep(1)
    for i in tqdm(range(7)):
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # acc
            data = {"train": {}, "test": {}}

            train_csv = pd.read_csv(os.path.join(args.csv, "test{}_train_acc.csv".format(i)), index_col=0)
            test_csv = pd.read_csv(os.path.join(args.csv, "test{}_test_acc.csv".format(i)), index_col=0)

            data["train"] = {
                "DGC": train_csv["DGC({})".format(cr)].tolist(),
                "GMC": train_csv["GMC({})".format(cr)].tolist(),
                "DGCwGMF": train_csv["DGCwGMF({})".format(cr)].tolist(),
                "DGCwGM": train_csv["DGCwGM({})".format(cr)].tolist(),
            }

            data["test"] = {
                "DGC": test_csv["DGC({})".format(cr)].tolist(),
                "GMC": test_csv["GMC({})".format(cr)].tolist(),
                "DGCwGMF": test_csv["DGCwGMF({})".format(cr)].tolist(),
                "DGCwGM": test_csv["DGCwGM({})".format(cr)].tolist(),
            }

            print("=== test{} {} ===".format(i, cr))
            print("DGC : {}".format(data["test"]["DGC"][-1]))
            print("GMC : {}".format(data["test"]["GMC"][-1]))
            print("DGCwGMF : {}".format(data["test"]["DGCwGMF"][-1]))
            print("DGCwGM : {}".format(data["test"]["DGCwGM"][-1]))

            save_both(data=data, test=i, CR=cr, type=0, output=args.output, smooth_rate=smooth_rate)

            # loss
            data = {"train": {}, "test": {}}

            train_csv = pd.read_csv(os.path.join(args.csv, "test{}_train_loss.csv".format(i)), index_col=0)
            test_csv = pd.read_csv(os.path.join(args.csv, "test{}_test_loss.csv".format(i)), index_col=0)

            data["train"] = {
                "DGC": train_csv["DGC({})".format(cr)].tolist(),
                "GMC": train_csv["GMC({})".format(cr)].tolist(),
                "DGCwGMF": train_csv["DGCwGMF({})".format(cr)].tolist(),
                "DGCwGM": train_csv["DGCwGM({})".format(cr)].tolist(),
            }

            data["test"] = {
                "DGC": test_csv["DGC({})".format(cr)].tolist(),
                "GMC": test_csv["GMC({})".format(cr)].tolist(),
                "DGCwGMF": test_csv["DGCwGMF({})".format(cr)].tolist(),
                "DGCwGM": test_csv["DGCwGM({})".format(cr)].tolist(),
            }
            save_both(data=data, test=i, CR=cr, type=1, output=args.output, smooth_rate=smooth_rate)

    print("====== Plot baseline ======")
    time.sleep(1)
    for i in tqdm(range(7)):
        data = {
            "test": {
                "DGC": [],
                "GMC": [],
                "DGCwGMF": [],
                "DGCwGM": []
            },
            "traffic": {
                "DGC": [],
                "GMC": [],
                "DGCwGMF": [],
                "DGCwGM": []
            }
        }

        test_csv = pd.read_csv(os.path.join(args.csv, "test{}_test_acc.csv".format(i)), index_col=0)
        traffic_csv = pd.read_csv(os.path.join(args.csv, "test{}_traffic.csv".format(i)), index_col=0)
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # print("=== test{} {} ===".format(i, cr))
            for d in data["test"]:
                data["test"][d].append(test_csv["{}({})".format(d, cr)].tolist()[-1])
                data["traffic"][d].append(traffic_csv["{}({})".format(d, cr)].tolist()[-1] / 1000000000)

                # print("{} {}".format(d, traffic_csv["{}({})".format(d, cr)].tolist()[-1] / 1000000000))
        save_baseline(data=data, test=i, output=args.output, smooth_rate=smooth_rate)
