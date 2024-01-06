## EASE_amazonbooks_x0 

A notebook to benchmark EASE_r on amazonbooks_x0 dataset.

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
amazonbooks_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code

1. Downalod the dataset from LightGCN repo and run the preprocessing script for format transformation

    ```python
    cd data/AmazonBooks/amazonbooks_x0
    python ENMF_data_process.py
    cd benchmarks/EASE_r
    ```

2. Run the following script to reproduce the result

    ```python
    # Hyper-parameter l2_reg = 50, searched among [1.e-1, 1, 10, 20, 50, 100, 200, 500]
    python -u main.py --train_data ../../data/AmazonBooks/amazonbooks_x0/train_enmf.txt --test_data ../../data/AmazonBooks/amazonbooks_x0/test_enmf.txt --l2_reg 50
    ```


### Results
```python
[Metrics] Recall(k=20): 0.070962 - Recall(k=50): 0.117733 - NDCG(k=20): 0.056734 - NDCG(k=50): 0.074354 - HitRate(k=20): 0.371008 - HitRate(k=50): 0.529282
```


### Logs
```python
2020-12-07 08:50:52.729758 Fitting EASE model...
2020-12-07 12:33:17.605835 Evaluating metrics...
2020-12-07 12:36:11.780981 Evaluating metrics for 52643 users...
2020-12-07 12:42:06.627166 [Metrics] Recall(k=20): 0.070962 - Recall(k=50): 0.117733 - NDCG(k=20): 0.056734 - NDCG(k=50): 0.074354 - HitRate(k=20): 0.371008 - HitRate(k=50): 0.529282
```