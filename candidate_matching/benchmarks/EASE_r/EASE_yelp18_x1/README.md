## EASE_yelp18_x0 

A notebook to benchmark EASE_r on yelp18_x0 dataset.

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
yelp18_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code

1. Downalod the dataset from LightGCN repo and run the preprocessing script for format transformation.

    ```python
    cd data/Yelp18/yelp18_x0
    python ENMF_data_process.py
    cd benchmarks/EASE_r
    ```

2. Run the following script to reproduce the result.

    ```python
    # Hyper-parameter l2_reg = 300, searched among [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1, 10, 20, 50, 100, 200, 300, 500, 1000]
    python -u main.py --train_data ../../data/Yelp18/yelp18_x0/train_enmf.txt --test_data ../../data/Yelp18/yelp18_x0/test_enmf.txt --l2_reg 300
    ```

### Results
```python
[Metrics] Recall(k=20): 0.065707 - Recall(k=50): 0.122483 - NDCG(k=20): 0.055216 - NDCG(k=50): 0.076210 - HitRate(k=20): 0.396615 - HitRate(k=50): 0.583902
```

### Logs

```python
2020-12-06 17:41:49.628423 Fitting EASE model...
2020-12-06 18:17:07.053317 Evaluating metrics...
2020-12-06 18:17:57.353290 Evaluating metrics for 31668 users...
2020-12-06 18:19:21.229441 [Metrics] Recall(k=20): 0.065707 - Recall(k=50): 0.122483 - NDCG(k=20): 0.055216 - NDCG(k=50): 0.076210 - HitRate(k=20): 0.396615 - HitRate(k=50): 0.583902
```