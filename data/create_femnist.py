import argparse
import os, json
import glob
import time

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/leaf/data/femnist/data")
    parser.add_argument('-f')
    args = parser.parse_args()

    path = os.path.abspath(args.data)

    train_list = glob.glob(os.path.join(path, "train", "all_data_*.json"))
    test_list = glob.glob(os.path.join(path, "test", "all_data_*.json"))

    train_data = {'users': [], 'num_samples': [], 'user_data': {}}
    test_data = {'users': [], 'num_samples': [], 'user_data': {}}

    for f in train_list:
        print("Load : {}".format(f))
        file_ = open(f, 'r')
        context = json.load(file_)
        file_.close()
        train_data['users'] += context['users']
        train_data['num_samples'] += context['num_samples']
        train_data['user_data'].update(context['user_data'])

    for f in test_list:
        print("Load : {}".format(f))
        file_ = open(f, 'r')
        context = json.load(file_)
        file_.close()
        test_data['users'] += context['users']
        test_data['num_samples'] += context['num_samples']
        test_data['user_data'].update(context['user_data'])

    os.makedirs(os.path.join(os.path.abspath("."), "femnist"), exist_ok=True)
    torch.save(train_data, os.path.join(os.path.abspath("."), "femnist", "train_data.pt"))
    torch.save(test_data, os.path.join(os.path.abspath("."), "femnist", "test_data.pt"))
    print("Save : {}".format(os.path.join(os.path.abspath("."), "femnist", "train_data.pt")))
    print("Save : {}".format(os.path.join(os.path.abspath("."), "femnist", "test_data.pt")))
    time.sleep(3)
    print("Done")
