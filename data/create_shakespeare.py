import argparse
import json
import os

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/leaf/data/shakespeare/data")
    parser.add_argument('-f')
    args = parser.parse_args()

    path = os.path.abspath(args.data)

    train_json_path = os.path.join(path, "train", os.listdir(os.path.join(path, "train"))[0])
    test_json_path = os.path.join(path, "test", os.listdir(os.path.join(path, "test"))[0])
    with open(os.path.join(train_json_path), "r") as file:
        train_context = json.load(file)
    with open(os.path.join(test_json_path), "r") as file:
        test_context = json.load(file)

    os.makedirs(os.path.join(os.path.abspath("."), "femnist"), exist_ok=True)
    torch.save(train_context, os.path.join(os.path.abspath("."), "shakespeare", "train_data.pt"))
    torch.save(test_context, os.path.join(os.path.abspath("."), "shakespeare", "test_data.pt"))
    print("Save : {}".format(os.path.join(os.path.abspath("."), "shakespeare", "train_data.pt")))
    print("Save : {}".format(os.path.join(os.path.abspath("."), "shakespeare", "test_data.pt")))
