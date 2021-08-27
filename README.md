# global-fusion

## global-fusion fedml implement


## Training
### mod-cifar10 dataset

In the mod-cifar10 dataset, we separate cifar10 into 20 parts.
Each part represent a client.

We follow the intrusion of FedAvg and measure these datasets by Earth Moving Distance provided by this approach.

Dataset  |  Earth moving distance
:--------------:|:-----:
test1    |  1.35
test2    |  1.18
test3    |  0.9883012820512821
test4    |  0.8703525641025642
test5    |  0.7600000000000001

```shell=
# usage
pyenv=... gpu=0 name_prefix="test1"  cifar10_subfolder="test1" bash train_cifar10.sh

# Or
# This will overwrite parameter in .ini file while initializing the dataloader
export dataset_path="./data/cifar10/test1"
python3 train.py --config ./configs/cifar10/config_wdv1.ini --output ./save/cifar10_final --pool 5 --gpu 0 --name_prefix ""
```