memory_checkpoint=./tmp/test_0

# femnist_baseline
python3 train.py --config ./configs/femnist/config_0.ini --output ./save/femnist_test --pool auto --gpu 0
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
# femnist_small_dgc_cr_scale_fr0.5_niid
python3 train.py --config ./configs/femnist/config_4.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30
# femnist_small_dgc_cr_scale_fr0.75_niid
python3 train.py --config ./configs/femnist/config_5.ini --output ./save/femnist_test --pool auto --gpu 0
sleep 30


## small_gfdgc_cr0.8_fu0.5_niid
#python3 train.py --config ./configs/femnist/config_6.ini --output ./save/femnist_test --pool auto --gpu 0
#sleep 30
## small_gfdgc_cr0.6_fu0.5_niid
#python3 train.py --config ./configs/femnist/config_7.ini --output ./save/femnist_test --pool auto --gpu 0
#sleep 30
## small_gfdgc_cr0.4_fu0.5_niid
#python3 train.py --config ./configs/femnist/config_8.ini --output ./save/femnist_test --pool auto --gpu 0
#sleep 30
