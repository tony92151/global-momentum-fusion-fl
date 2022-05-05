import argparse
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from tqdm import tqdm

# Extraction function
def tflog2pandas(path, filter=["traffic(number_of_parameters)"]):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            if not tag in filter:
                continue
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data, r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb_event', help="path to tb_event", type=str, default=None, required=True)
    parser.add_argument('--output_csv', help="path to tb_event", type=str, default=None, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_csv, exist_ok=True)

    for t in tqdm(range(7)):
        result = {}
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            path = os.path.join(args.tb_event, "test{}_cifar10_r56_DGC_cr{}".format(t, cr))
            _, r = tflog2pandas(path)
            result["DGC({})".format(cr)] = r["value"]

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            path = os.path.join(args.tb_event, "test{}_cifar10_r56_GMC_cr{}".format(t, cr))
            _, r = tflog2pandas(path)
            result["GMC({})".format(cr)] = r["value"]

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            path = os.path.join(args.tb_event, "test{}_cifar10_r56_GFDGC_cr{}".format(t, cr))
            _, r = tflog2pandas(path)
            result["DGCwGMF({})".format(cr)] = r["value"]

        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            path = os.path.join(args.tb_event, "test{}_cifar10_r56_DGCwGM_cr{}".format(t, cr))
            _, r = tflog2pandas(path)
            result["DGCwGM({})".format(cr)] = r["value"]

        pd.DataFrame(result).to_csv(os.path.join(args.output_csv, "test{}_traffic.csv".format(t)))