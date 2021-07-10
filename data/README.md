# dataset generate

### cifar10

```python
python3 ./create_cifar10.py --data ./cifar10 --n 4
```

### femnist

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist)

```shell
# might take 1 hour
./preprocess.sh -s niid --sf 0.05 -k 200 -t sample
```
this might take 1 hours

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