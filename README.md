# global-momentum-fusion-fl

## Introduction

This repository is an official implementation of this paper. [arXiv](https://arxiv.org/abs/2211.09320)

`The code is still under refactoring`
```
@article{kuo2022improving,
  title={Improving Federated Learning Communication Efficiency with Global Momentum Fusion for Gradient Compression Schemes},
  author={Kuo, Chun-Chih and Kuo, Ted Tsei and Lin, Chia-Yu},
  journal={arXiv preprint arXiv:2211.09320},
  year={2022}
}
```

As description in the paper, we propose global-momentum-fusion method to reduce communication overheads. 
The following figure demonstrate how global-momentum-fusion work with DGC in federated learning trading flow.
![image](image/DGCwGMF_fig.jpg)


## Dataset prepare

```bash
cd data
python3 ./create_cifar10.py --download
python3 ./create_shakespeare.py --download
cd ..
```

[Detail](data/README.md)

## Training
### Cifar10 dataset

```shell=
# usage
In our cifar10 experiment:
seed=123 gpu=0 tbpath=./cifar10_repo_test compress_method="DGC" bash train_cifar10.sh
seed=123 gpu=0 tbpath=./cifar10_repo_test compress_method="GFDGC" bash train_cifar10.sh
seed=123 gpu=0 tbpath=./cifar10_repo_test compress_method="SGC" bash train_cifar10.sh
seed=123 gpu=0 tbpath=./cifar10_repo_test compress_method="GFGC" bash train_cifar10.sh
```
### Shakespeare dataset
```shell=
In our shakespeare experiment:
seed=123 gpu=0 tbpath=./sha_repo_exp compress_method="DGC" bash train_sha.sh
seed=123 gpu=0 tbpath=./sha_repo_exp compress_method="GFDGC" bash train_sha.sh
seed=123 gpu=0 tbpath=./sha_repo_exp compress_method="SGC" bash train_sha.sh
seed=123 gpu=0 tbpath=./sha_repo_exp compress_method="GFGC" bash train_sha.sh
```