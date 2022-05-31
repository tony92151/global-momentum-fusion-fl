#!/bin/bash

if [ -z "$pyenv" ]
then
  pyenv=$(which python3)
fi

if [ -z "$gpu" ]
then
  gpu="0"
fi

if [ -z "tbpath" ]
then
  tbpath="cifar10_tau_exp"
fi

if [ -z "compress_method" ]
then
  compress_method="DGC"
fi

if [ -z "seed" ]
then
  seed="123"
fi


echo "Python interpreter: $pyenv"
echo "Torch version : $($pyenv -c 'import torch; print(torch.__version__)')"
echo "GPU : $gpu"
echo "tbpath : $tbpath"
echo "seed : $seed"


# for (( i = 0; i < 7; i++ ))
# for (( i = 6; i >= 0; i-- ))
# do

export i=6
export index_path="./data/cifar10/test$i/index.json"

# for (( t = 1; t <= 1; i++ ))
# do
t=1
python3 train_tau_tuning.py \
--config ./configs/cifar10/GFDGC_tau_exp/t"$t"/config_1.ini \
--tensorboard_path ./"$tbpath" \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.1_t"$t" \
--gpu $gpu \
--pool 5 \
--seed 123

#python3 train.py \
#--config ./configs/cifar10/GFDGC_tau_exp/t"$t"/config_3.ini \
#--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.3_t"$t" \
#--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.3_t"$t" \
#--gpu $gpu \
#--pool 5 \
#--seed $seed
#
#python3 train.py \
#--config ./configs/cifar10/GFDGC_tau_exp/t"$t"/config_1.ini \
#--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.5_t"$t" \
#--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.5_t"$t" \
#--gpu $gpu \
#--pool 5 \
#--seed $seed

# done


# done
