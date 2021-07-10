import argparse
import json
import os, time

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/leaf/data/shakespeare/data")
    parser.add_argument('-f')
    args = parser.parse_args()

    if args.data == "download":
        if os.path.exists(os.path.abspath("./shakespeare")):
            raise ValueError("Folder: \"{}\" exist".format(os.path.abspath("./femnist")))
        os.makedirs(os.path.abspath("./shakespeare"))
        import gdown, tarfile
        # download
        url = 'https://drive.google.com/uc?id=1pG_tN1D874u5n3hrkkvYjKgCGWSHdF50'
        output = os.path.join(os.path.abspath("./shakespeare"), 'shakespeare_all.tar.gz')
        print("\nDownload ...")
        gdown.download(url, output, quiet=False)
        # check
        md5 = '2f6538edc4202638973a9f1d8141d262'
        gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)
        time.sleep(3)
        # extraction
        print("\nExtracting ...")
        tar = tarfile.open(output, 'r:gz')
        tar.extractall(path=os.path.abspath("./shakespeare"))

        path = os.path.abspath("./shakespeare/shakespeare_all")
    else:
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
    print("\nSave : {}".format(os.path.join(os.path.abspath("."), "shakespeare", "train_data.pt")))
    print("Save : {}".format(os.path.join(os.path.abspath("."), "shakespeare", "test_data.pt")))
