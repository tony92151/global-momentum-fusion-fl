# dataset generate

### cifar10

In the mod-cifar10 dataset, we separate cifar10 into 20 parts.
Each part represent a client.

We follow the intrusion of FedAvg and measure these datasets by Earth Moving Distance provided by this approach.

Dataset  |  Earth moving distance
:--------------:|:-----:
test0    |  0.00 (iid)
test1    |  0.48
test2    |  0.76
test3    |  0.87
test4    |  0.99
test5    |  1.18
test6    |  1.35

***
usage: create_cifar10.py [-h] [--n N] [--data DATA] [--download] [--num_shards NUM_SHARDS] [--seed SEED]
                         [-f F]

optional arguments:
  
-h, --help            show this help message and exit

  --n N                 Separate to n datasets

  --data DATA           Path to dataset

  --download

  --num_shards NUM_SHARDS
                        Default num_shards=100. Smaller num_shards will make the dataset more non-
                        iid.[200,100,80,50,40,20]
  
--seed SEED
  
-f F
***

This command will download cifar10 by pyrorch build-in function, and generate indexing json files for all clients.
```shell
python3 ./create_cifar10.py --n 20
```

This command will download cifar10 by pyrorch build-in function, and download indexing json files of 20 clients in our experiment.
```shell
python3 ./create_cifar10.py --download
```


### femnist

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist)

```shell
# might take 1.5 hour
./preprocess.sh -s niid --sf 1 -k 100 -t sample --smplseed 123 --spltseed 1234
```
this might take 1.5 hours

```shell
python3 ./create_femnist.py --data {path to}/leaf/data/femnist/data
```

Or just download it by
```shell
python3 ./create_femnist.py --data download
```
this has 3560 clients and at least 100 images each client.

### shakespeare

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare)

```shell
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 1
```
this might take 1 minute

```shell
python3 ./create_shakespeare.py --data {path to}/leaf/data/shakespeare/data
```
Or just download it by
```shell
python3 ./create_shakespeare.py --download
```

### Sentiment140

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/sent140)
```shell
./preprocess.sh -s niid --sf 0.01 -k 3 -t sample --smplseed 123 --spltseed 123
```

