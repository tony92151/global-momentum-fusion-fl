# dataset generate

### cifar10

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
# might take 1 hour
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
this has 175 clients and at least 200 images each client.

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