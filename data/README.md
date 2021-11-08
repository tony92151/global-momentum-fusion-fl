# dataset generate

### cifar10

***
usage: create_cifar10.py [-h] [--n N] [--data DATA] [--download DOWNLOAD] [--num_shards NUM_SHARDS] [--seed SEED]
                         [-f F]

optional arguments:
  
-h, --help            show this help message and exit

  --n N                 Separate to n datasets

  --data DATA           Path to dataset

  --download DOWNLOAD

  --num_shards NUM_SHARDS
                        Default num_shards=100. Smaller num_shards will make the dataset more non-
                        iid.[200,100,80,50,40,20]
  
--seed SEED
  
-f F
***

This command will download cifar10 by pyrorch build-in function, and generate indexing json files for all clients.
```python
python3 ./create_cifar10.py --n 20
```

This command will download cifar10 by pyrorch build-in function, and download indexing json files of 20 clients in our experiment.
```python
python3 ./create_cifar10.py --download True
```


### femnist

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist)

```shell
# might take 1.5 hour
./preprocess.sh -s niid --sf 1 -k 100 -t sample --smplseed 123 --spltseed 1234
```
this might take 1.5 hours

```python
python3 ./create_femnist.py --data {path to}/leaf/data/femnist/data
```

Or just download it by
```python
python3 ./create_femnist.py --data download
```
this has 3560 clients and at least 100 images each client.

### shakespeare

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare)

```shell
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 1
```
this might take 1 minute

```python
python3 ./create_shakespeare.py --data {path to}/leaf/data/shakespeare/data
```
Or just download it by
```python
python3 ./create_shakespeare.py --data download
```

### Sentiment140

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/sent140)
```shell
./preprocess.sh -s niid --sf 0.01 -k 3 -t sample --smplseed 123 --spltseed 123
```

