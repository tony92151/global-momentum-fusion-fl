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
  tbpath="cifar10_repo_exp"
fi

if [ -z "compress_method" ]
then
  compress_method="DGC"
fi

if [ -z "seed" ]
then
  seed=123
fi


echo "Python interpreter: $pyenv"
echo "Torch version : $($pyenv -c 'import torch; print(torch.__version__)')"
echo "GPU : $gpu"
echo "tbpath : $tbpath"
echo "seed : $seed"


# for (( i = 0; i < 7; i++ ))
for (( i = 6; i >= 0; i-- ))
do

# export i=6
export index_path="./data/cifar10/test$i/index.json"

python3 train.py \
--config ./configs/cifar10/$compress_method/config_1.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.1 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.1 \
--gpu $gpu \
--pool 5 \
--seed $seed

#python3 train.py \
#--config ./configs/cifar10/$compress_method/config_2.ini \
#--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.2 \
#--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.2 \
#--gpu $gpu \
# --pool 5 \
# --seed $seed

python3 train.py \
--config ./configs/cifar10/$compress_method/config_3.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.3 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.3 \
--gpu $gpu \
--pool 5 \
--seed $seed

#python3 train.py \
#--config ./configs/cifar10/$compress_method/config_4.ini \
#--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.4 \
#--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.4 \
#--gpu $gpu \
# --pool 5 \
# --seed $seed

python3 train.py \
--config ./configs/cifar10/$compress_method/config_5.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.5 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.5 \
--gpu $gpu \
--pool 5 \
--seed $seed

#python3 train.py \
#--config ./configs/cifar10/$compress_method/config_6.ini \
#--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.6 \
#--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.6 \
#--gpu $gpu \
# --pool 5 \
# --seed $seed


python3 train.py \
--config ./configs/cifar10/$compress_method/config_7.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.7 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.7 \
--gpu $gpu \
--pool 5 \
--seed 123

#python3 train.py \
#--config ./configs/cifar10/$compress_method/config_8.ini \
#--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.8 \
#--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.8 \
#--gpu $gpu \
#--pool 5 \
#--seed 123

python3 train.py \
--config ./configs/cifar10/$compress_method/config_9.ini \
--tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.9 \
--output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr0.9 \
--gpu $gpu \
--pool 5 \
--seed $seed

# python3 train.py \
# --config ./configs/cifar10/$compress_method/config_10.ini \
# --tensorboard_path ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr1.0 \
# --output ./"$tbpath"/test"$i"_cifar10_r56_"$compress_method"_cr1.0 \
# --gpu $gpu \
# --pool 5 \
# --seed $seed

done
