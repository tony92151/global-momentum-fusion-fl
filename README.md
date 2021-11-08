# global-fusion

## global-fusion fedml implement


## Training
### mod-cifar10 dataset

In the mod-cifar10 dataset, we separate cifar10 into 20 parts.
Each part represent a client.

We follow the intrusion of FedAvg and measure these datasets by Earth Moving Distance provided by this approach.

Dataset  |  Earth moving distance
:--------------:|:-----:
test0    |  0.00 (iid)
test1    |  0.48
test2    |  0.76
test3    |  0.87
test4    |  0.99
test5    |  1.18
test6    |  1.35

```shell=
# usage
In our experiment:
gpu=0 tbpath=./cifar10_repo_test compress_method="DGC" bash train_cifar10.sh
gpu=0 tbpath=./cifar10_repo_test compress_method="GFDGC" bash train_cifar10.sh
gpu=0 tbpath=./cifar10_repo_test compress_method="SGC" bash train_cifar10.sh
gpu=0 tbpath=./cifar10_repo_test compress_method="GFGC" bash train_cifar10.sh

```