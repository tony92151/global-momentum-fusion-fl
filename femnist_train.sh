# small_dgc_cr1_iid
# python3 train.py --config ./configs/femnist/config_0.ini --output ./save/femnist_test
# sleep 30
# small_dgc_cr1_niid
python3 train.py --config ./configs/femnist/config_1.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
# small_dgc_cr0.8_niid
python3 train.py --config ./configs/femnist/config_2.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
# small_dgc_cr0.6_niid
python3 train.py --config ./configs/femnist/config_3.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
# small_dgc_cr0.4_niid
python3 train.py --config ./configs/femnist/config_4.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
# small_gfdgc_cr1_fu0.5_niid
python3 train.py --config ./configs/femnist/config_5.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
# small_gfdgc_cr0.8_fu0.5_niid
python3 train.py --config ./configs/femnist/config_6.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
# small_gfdgc_cr0.6_fu0.5_niid
python3 train.py --config ./configs/femnist/config_7.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
# small_gfdgc_cr0.4_fu0.5_niid
python3 train.py --config ./configs/femnist/config_8.ini --output ./save/femnist_test --pool 40 --gpu 0,1
sleep 30
