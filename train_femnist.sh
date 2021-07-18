#!/bin/bash

if [ -z "$pyenv" ]
then
  pyenv=$(which python3)
fi

if [ -z "$gpu" ]
then
  gpu="0"
fi

echo "Python interpreter: $pyenv"
echo "Torch version : $($pyenv -c 'import torch; print(torch.__version__)')"
echo "GPU : $gpu"

# fedavg iid
$pyenv train.py --config ./configs/femnist/config_fedavg_iid.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10

# fedavg niid
$pyenv train.py --config ./configs/femnist/config_fedavg_niid.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10

# dgc iid
$pyenv train.py --config ./configs/femnist/config_dgc_iid.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10

# dgc niid
$pyenv train.py --config ./configs/femnist/config_dgc_niid.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10

############################################################################################################
############################################################################################################


# dgc 0.1
$pyenv train.py --config ./configs/femnist/config_wdv1.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 0.2
$pyenv train.py --config ./configs/femnist/config_wdv2.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 0.3
$pyenv train.py --config ./configs/femnist/config_wdv3.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 0.4
$pyenv train.py --config ./configs/femnist/config_wdv4.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 0.5
$pyenv train.py --config ./configs/femnist/config_wdv5.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10

# dgc 0.6
$pyenv train.py --config ./configs/femnist/config_wdv6.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 0.7
$pyenv train.py --config ./configs/femnist/config_wdv7.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 0.8
$pyenv train.py --config ./configs/femnist/config_wdv8.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 0.9
$pyenv train.py --config ./configs/femnist/config_wdv9.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# dgc 1.0
#$pyenv train.py --config ./configs/femnist/config_wdv10.ini --output ./save/femnist_final --pool 5 --gpu $gpu
#sleep 10

############################################################################################################
############################################################################################################

# gfdgc 0.1
$pyenv train.py --config ./configs/femnist/config_wdv11.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 0.2
$pyenv train.py --config ./configs/femnist/config_wdv12.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 0.3
$pyenv train.py --config ./configs/femnist/config_wdv13.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 0.4
$pyenv train.py --config ./configs/femnist/config_wdv14.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 0.5
$pyenv train.py --config ./configs/femnist/config_wdv15.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10

# gfdgc 0.6
$pyenv train.py --config ./configs/femnist/config_wdv16.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 0.7
$pyenv train.py --config ./configs/femnist/config_wdv17.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 0.8
$pyenv train.py --config ./configs/femnist/config_wdv18.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 0.9
$pyenv train.py --config ./configs/femnist/config_wdv19.ini --output ./save/femnist_final --pool 5 --gpu $gpu
sleep 10
# gfdgc 1.0
#$pyenv train.py --config ./configs/femnist/config_wdv20.ini --output ./save/femnist_final --pool 5 --gpu $gpu
#sleep 10

############################################################################################################
############################################################################################################