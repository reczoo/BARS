## EASE_gowalla_x0 

A notebook to benchmark EASE_r on gowalla_x0 dataset.

Author: [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

    ```python
    CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
    RAM: 500G+
    ```
+ Software

    ```python
    python: 3.6.5
    pandas: 1.0.0
    numpy: 1.18.1
    ```

### Dataset
gowalla_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code

1. Downalod the dataset from LightGCN repo and run the preprocessing script for format transformation

    ```python
    cd data/Gowalla/gowalla_x0
    python ENMF_data_process.py
    cd benchmarks/EASE_r
    ```

2. Run the following script to reproduce the result

    ```python
    # Hyper-parameter l2_reg = 80, searched among [1, 10, 20, 50, 80, 100, 150, 200, 500]
    python -u main.py --train_data ../../data/Gowalla/gowalla_x0/train_enmf.txt --test_data ../../data/Gowalla/gowalla_x0/test_enmf.txt --l2_reg 80
    ```


### Results
```python
[Metrics] Recall(k=20): 0.176493 - Recall(k=50): 0.270126 - NDCG(k=20): 0.146657 - NDCG(k=50): 0.175974 - HitRate(k=20): 0.572677 - HitRate(k=50): 0.708051
```


### Logs
```python
2020-12-07 19:43:17.365952 Fitting EASE model...
2020-12-07 20:04:39.106220 Evaluating metrics...
2020-12-07 20:05:00.711707 Evaluating metrics for 29858 users...
2020-12-07 20:06:22.041540 [Metrics] Recall(k=20): 0.176493 - Recall(k=50): 0.270126 - NDCG(k=20): 0.146657 - NDCG(k=50): 0.175974 - HitRate(k=20): 0.572677 - HitRate(k=50): 0.708051
```


