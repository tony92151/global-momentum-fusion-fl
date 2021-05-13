# dataset generate

### cifar10

```python
python3 ./create_cifar10.py --data ./cifar10 --n 4
```

### femnist

fellow this [repo](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist)

```shell
# might take 1 hour
./preprocess.sh -s niid --sf 0.05 -k 100 -t sample
```
this might take 1 hours

```python
python3 ./create_femnist.py --data {path to}/leaf/data/femnist/data
```