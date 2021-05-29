# femnist_fedavg_baseline
python3 train.py --config ./configs/femnist/config_baseline_1.ini --output ./save/femnist_test --pool 4 --gpu 0
sleep 30

# femnist_small_dgc_cr_scale
python3 train.py --config ./configs/femnist/config_1.ini --output ./save/femnist_test --pool 4 --gpu 0
sleep 30
# femnist_small_gfdgc_cr_scale_fr_scale
python3 train.py --config ./configs/femnist/config_2.ini --output ./save/femnist_test --pool 4 --gpu 0
sleep 30

