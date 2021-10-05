#!/bin/bash

if [ -z "$pyenv" ]
then
  pyenv=$(which python3)
fi

if [ -z "$gpu" ]
then
  gpu="0"
fi

if [ -z "name_prefix" ]
then
  name_prefix=""
fi

if [ -z "tbpath" ]
then
  tbpath="cifar10_repo_exp"
fi

if [ -z "compress_method" ]
then
  compress_method="DGC"
fi




echo "Python interpreter: $pyenv"
echo "Torch version : $($pyenv -c 'import torch; print(torch.__version__)')"
echo "GPU : $gpu"
echo "tbpath : $tbpath"

#######################################################
# environment variable overwrite
# This will overwrite parameter in .ini file while initializing the dataloader

#export compress_method="DGC"
#######################################################

## fedavg iid
#$pyenv train.py --config ./configs/cifar10/config_fedavg_iid.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10
#
## fedavg niid
#$pyenv train.py --config ./configs/cifar10/config_fedavg_niid.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10

## dgc iid
#$pyenv train.py --config ./configs/cifar10/config_dgc_iid.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10
#
## dgc niid
#$pyenv train.py --config ./configs/cifar10/config_dgc_niid.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10

############################################################################################################
############################################################################################################
tbpath="cifar10_repo_exp"

# i = 0,1,2,3,4,5,6
for (( i = 0; i < 7; i++ ))
do
# compress ratio 0.1
export index_path="./data/cifar10/test$i/index.json"
$pyenv train_1.py \
--config ./configs/cifar10/$compress_method/config_1.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.1 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.1 \ \
--pool 5 \
--seed $i \
--gpu $gpu

# compress ratio 0.3
export index_path="./data/cifar10/test$i/index.json"
$pyenv train_1.py \
--config ./configs/cifar10/$compress_method/config_3.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.3 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.3 \ \
--pool 5 \
--seed $i \
--gpu $gpu

# compress ratio 0.5
export index_path="./data/cifar10/test$i/index.json"
$pyenv train_1.py \
--config ./configs/cifar10/$compress_method/config_5.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.5 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.5 \ \
--pool 5 \
--seed $i \
--gpu $gpu


# compress ratio 0.7
export index_path="./data/cifar10/test$i/index.json"
$pyenv train_1.py \
--config ./configs/cifar10/$compress_method/config_7.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.7 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.7 \ \
--pool 5 \
--seed $i \
--gpu $gpu


# compress ratio 0.9
export index_path="./data/cifar10/test$i/index.json"
$pyenv train_1.py \
--config ./configs/cifar10/$compress_method/config_9.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.9 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.9 \ \
--pool 5 \
--seed $i \
--gpu $gpu


done