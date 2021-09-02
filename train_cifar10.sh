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

if [ -z "cifar10_subfolder" ]
then
  name_prefix=""
fi

if [ -z "datatype" ]
then
  datatype="niid"
fi

if [ -z "compress_method" ]
then
  compress_method="DGC"
fi

echo "Python interpreter: $pyenv"
echo "Torch version : $($pyenv -c 'import torch; print(torch.__version__)')"
echo "GPU : $gpu"

#######################################################
# environment variable overwrite
# This will overwrite parameter in .ini file while initializing the dataloader
export dataset_path="./data/cifar10/$cifar10_subfolder"
export dataset_type=$datatype
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


# compress ratio 0.1
$pyenv train.py --config ./configs/cifar10/$compress_method/config_1.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
sleep 10
## dgc 0.2
#$pyenv train.py --config ./configs/cifar10/$compress_method/config_2.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10
# compress ratio 0.3
$pyenv train.py --config ./configs/cifar10/$compress_method/config_3.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
sleep 10
## compress ratio 0.4
#$pyenv train.py --config ./configs/cifar10/$compress_method/config_4.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10
# compress ratio 0.5
$pyenv train.py --config ./configs/cifar10/$compress_method/config_5.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
sleep 10

## compress ratio 0.6
#$pyenv train.py --config ./configs/cifar10/$compress_method/config_6.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10
# compress ratio 0.7
$pyenv train.py --config ./configs/cifar10/$compress_method/config_7.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
sleep 10
## compress ratio 0.8
#$pyenv train.py --config ./configs/cifar10/$compress_method/config_8.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10
# compress ratio 0.9
$pyenv train.py --config ./configs/cifar10/$compress_method/config_9.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
sleep 10
# compress ratio 1.0
#$pyenv train.py --config ./configs/cifar10/$compress_method/config_10.ini --output ./save/cifar10_final --pool 5 --gpu $gpu --name_prefix $name_prefix
#sleep 10
