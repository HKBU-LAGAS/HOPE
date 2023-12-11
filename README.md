# Bipartite Graph Clustering

## Requirements
- Linux machine
- Python3
- numPy = 1.23.3
- scikit-learn = 1.1.2
- scipy = 1.9.2
- argparse = 1.1
- munkres = 1.1.4

## Large Datasets
Please download them from [link](https://drive.google.com/file/d/1Z8QCJ-6NtbMFKRbY3qOpg8B3nTyrd54C)

## Clustering and Evaluation
```shell
$ sh evaluate.sh BGC cora
$ sh evaluate.sh FNEM cora
$ sh evaluate.sh SNEM cora
```
or 
```shell
$ python3 -W ignore run.py --algo BGC --data cora --alpha 0.3 --dim 5
$ python3 -W ignore eval.py --algo BGC --data cora
$
$ python3 -W ignore run.py --algo FNEM --data cora --alpha 0.3 --dim 5
$ python3 -W ignore eval.py --algo FNEM --data cora
$
$ python3 -W ignore run.py --algo SNEM --data cora --alpha 0.3 --dim 5
$ python3 -W ignore eval.py --algo SNEM --data cora
```
Note that "--dim 5" means that the beta parameter in the paper is set to 5*k.
The clustering results can be found in the folder "cluster".
