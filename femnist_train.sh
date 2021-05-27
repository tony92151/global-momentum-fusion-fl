# femnist_baseline
python3 train.py --config ./configs/femnist/config_0.ini --output ./save/femnist_test --pool auto --gpu 0 --baseline 1
sleep 30


# femnist_small_dgc_cr1.0_niid
python3 train.py --config ./configs/femnist/config_1.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30
# femnist_small_gfdgc_cr1.0_fr0.5_niid
python3 train.py --config ./configs/femnist/config_2.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30
# femnist_small_gfdgc_cr1.0_fr0.75_niid
python3 train.py --config ./configs/femnist/config_3.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30


# femnist_small_dgc_cr_scale_niid
python3 train.py --config ./configs/femnist/config_4.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30
# femnist_small_gfdgc_cr_scale_fr0.5_niid
python3 train.py --config ./configs/femnist/config_5.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30
# femnist_small_gfdgc_cr_scale_fr0.75_niid
python3 train.py --config ./configs/femnist/config_6.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30

