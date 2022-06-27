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
  tbpath="mnist_repo_exp"
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


for (( i = 5; i >= 0; i-- ))
do

# export i=1
export index_path="./data/fashionmnist/test$i/index.json"


for (( t = 1; t <= 10; t++ ))
do


python3 train.py \
--config ./configs/fashionmnist/GFDGC_tau_exp/t"$t"/config_1.ini \
--tensorboard_path ./"$tbpath"/test"$i"_fashionmnist_r20_"$compress_method"_cr0.1_t"$t" \
--output ./"$tbpath"/test"$i"_fashionmnist_r20_"$compress_method"_cr0.1_t"$t" \
--gpu $gpu \
--pool 5 \
--seed $seed

# python3 train.py \
# --config ./configs/cifar10/GFDGC_tau_exp/t"$t"/config_3.ini \
# --tensorboard_path ./"$tbpath"/test"$i"_fashionmnist_r20_"$compress_method"_cr0.1_t"$t" \
# --output ./"$tbpath"/test"$i"_fashionmnist_r20_"$compress_method"_cr0.1_t"$t" \
# --gpu $gpu \
# --pool 5 \
# --seed $seed

# python3 train.py \
# --config ./configs/cifar10/GFDGC_tau_exp/t"$t"/config_5.ini \
# --tensorboard_path ./"$tbpath"/test"$i"_fashionmnist_r20_"$compress_method"_cr0.1_t"$t" \
# --output ./"$tbpath"/test"$i"_fashionmnist_r20_"$compress_method"_cr0.1_t"$t" \
# --gpu $gpu \
# --pool 5 \
# --seed $seed

done


done
