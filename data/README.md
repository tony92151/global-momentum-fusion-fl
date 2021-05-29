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

or just download it [here](https://drive.google.com/file/d/1EYBkvR_gvKdndffHGMEqe0GjcEDn54IQ/view?usp=sharing), this has 175 clients and at least 200 images each client.