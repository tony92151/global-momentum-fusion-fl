# cifar_baseline_iid
python3 train.py --config ./configs/cifar10/config_baseline_0.ini --output ./save/cifar10_test --pool 4 --gpu 0
sleep 30
# cifar_baseline_niid
python3 train.py --config ./configs/cifar10/config_baseline_1.ini --output ./save/cifar10_test --pool 4 --gpu 0
sleep 30

# cifar_dgc_cr_scale
python3 train.py --config ./configs/cifar10/config_1.ini --output ./save/cifar10_test --pool 4 --gpu 0
sleep 30
# cifar_gfdgc_cr_scale_fr_scale
python3 train.py --config ./configs/cifar10/config_2.ini --output ./save/cifar10_test --pool 4 --gpu 0
sleep 30

