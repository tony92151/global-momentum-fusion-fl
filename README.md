# global-fusion

## global-fusion fedml implement


## Training
### mod-cifar10 dataset

In the mod-cifar10 dataset, we separate cifar10 into 20 parts.
Each part represent a client.

We follow the intrusion of FedAvg and measure these datasets by Earth Moving Distance provided by this approach.

Dataset  |  Earth moving distance
:--------------:|:-----:
test0    |  0
test1    |  1.35
test2    |  1.1799999999999997
test3    |  0.9883012820512821
test4    |  0.8703525641025642
test5    |  0.7600000000000001
test6    |  0.4800000000000001

```shell=
# usage
pyenv=... gpu=0 name_prefix="test1" cifar10_subfolder="test1" datatype="niid" compress_method="DGC" bash train_cifar10.sh

# Or
# This will overwrite parameter in .ini file while initializing the dataloader
export dataset_path="./data/cifar10/test1"
python3 train.py --config ./configs/cifar10/config_wdv1.ini --output ./save/cifar10_final --pool 5 --gpu 0 --name_prefix ""

In our experiment:

gpu=0 name_prefix="test0_"  datatype="iid" cifar10_subfolder="test1" 
gpu=0 name_prefix="test1_" cifar10_subfolder="test1" datatype="iid" compress_method="DGC"  bash train_cifar10.sh
gpu=0 name_prefix="test1_" cifar10_subfolder="test1" datatype="iid" compress_method="GFDGC"  bash train_cifar10.sh

gpu=0 name_prefix="test1_"  cifar10_subfolder="test1" compress_method="DGC" bash train_cifar10.sh
gpu=0 name_prefix="test1_"  cifar10_subfolder="test1" compress_method="GFDGC" bash train_cifar10.sh

gpu=0 name_prefix="test1_"  cifar10_subfolder="test2" compress_method="DGC" bash train_cifar10.sh
gpu=0 name_prefix="test1_"  cifar10_subfolder="test2" compress_method="GFDGC" bash train_cifar10.sh

gpu=0 name_prefix="test1_"  cifar10_subfolder="test3" compress_method="DGC" bash train_cifar10.sh
gpu=0 name_prefix="test1_"  cifar10_subfolder="test3" compress_method="GFDGC" bash train_cifar10.sh

gpu=0 name_prefix="test1_"  cifar10_subfolder="test4" compress_method="DGC" bash train_cifar10.sh
gpu=0 name_prefix="test1_"  cifar10_subfolder="test4" compress_method="GFDGC" bash train_cifar10.sh

gpu=0 name_prefix="test1_"  cifar10_subfolder="test5" compress_method="DGC" bash train_cifar10.sh
gpu=0 name_prefix="test1_"  cifar10_subfolder="test5" compress_method="GFDGC" bash train_cifar10.sh

gpu=0 name_prefix="test1_"  cifar10_subfolder="test6" compress_method="DGC" bash train_cifar10.sh
gpu=0 name_prefix="test1_"  cifar10_subfolder="test6" compress_method="GFDGC" bash train_cifar10.sh

```