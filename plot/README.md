# plot 

## usage

Assume we run our experimet two times with two folders.
    
#### Mod-cifar10

```bash=
# Mod-cifar10
cd <repo>
python3 plot/tb2csv.py --tb_event ./<path to result A> --output_csv ./cifar_plot/csv2
python3 plot/tb2csv.py --tb_event ./<path to result B> --output_csv ./cifar_plot/csv1

python3 plot/cifar_csv_merge.py --csv1 ./cifar_plot/csv1 --csv2 ./cifar_plot/csv2 --output_csv ./cifar_plot/csv_merged

python3 plot/cifar_plot.py --csv cifar_plot/csv_merged --output ./cifar_plot

```


#### sha

```bash=
# Mod-cifar10
cd <repo>

python3 plot/cifar_convert.py --trensorboard_path ./<path to result A> --output_csv ./cifar_plot/csv1
python3 plot/cifar_convert.py --trensorboard_path ./<path to result B> --output_csv ./cifar_plot/csv2


python3 plot/cifar_tb2csv.py --tb_event ./<path to result A> --output_csv ./cifar_plot/csv1
python3 plot/cifar_tb2csv.py --tb_event ./<path to result B> --output_csv ./cifar_plot/csv2

python3 plot/cifar_csv_merge.py --csv1 ./cifar_plot/csv1 --csv2 ./cifar_plot/csv2 --output_csv ./cifar_plot/csv_merged

python3 plot/sha_plot.py --csv sha_plot/csv_merged --output ./sha_plot

```


```bash=
python3 plot/sha_convert.py --trensorboard_path ./<path to result A> --output_csv ./sha_plot/csv1
python3 plot/sha_convert.py --trensorboard_path ./<path to result B> --output_csv ./sha_plot/csv2

python3 plot/sha_tb2csv.py --tb_event ./<path to result A> --output_csv ./sha_plot/csv1
python3 plot/sha_tb2csv.py --tb_event ./<path to result B> --output_csv ./sha_plot/csv2

python3 plot/sha_csv_merge.py --csv1 ./sha_plot/csv1 --csv2 ./sha_plot/csv2 --output_csv ./sha_plot/csv_merged
```
    