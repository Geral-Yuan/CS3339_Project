# SJTU CS3339 Machine Learning 2024 Fall Project

**Author**: [Geral-Yuan](https://github.com/Geral-Yuan)

## Installation

```sh
git clone https://github.com/Geral-Yuan/CS3339_Project.git
cd CS3339_Project
conda create -n <env_name> python=3.12
conda activate <env_name>
pip install -r requirements.txt
```

## File Tree

```
YuanJiale-521370910130/
├── Report.pdf
└── Code/
    ├── README.pdf
    ├── requirements.txt
    ├── main.py
    ├── validation.py
    ├── test.py
    ├── model
    │   ├── mlp.py
    │   ├── lr.py
    │   └── svm.py
    └── scripts
        ├── mlp.sh
        ├── lr.sh
        ├── svm.sh
        ├── test_mlp.sh
        ├── test_lr.sh
        └── test_svm.sh
```

## Preparing Data

Download train and test data, and put them under directory `data`. 

## Quick Test

To reproduce my results, run test scripts as follows. Each will generate a CSV file under directory `submission` as the prediction results on test set, which can be submitted to Kaggle to get the prediction accuracy.

```sh
bash scripts/test_mlp.sh
bash scripts/test_lr.sh
bash scripts/test_svm.sh
```

## Train & Validation

To train with grid search and cross validation, run train scripts as follows. Each will generate a log file under directory `logs` logging the output and a CSV file under directory `results` recording all combinations of hyperparameters as well as train and validation accuracy obtained throught cross validation.

```sh
bash scripts/mlp.sh
bash scripts/lr.sh
bash scripts/svm.sh
```
