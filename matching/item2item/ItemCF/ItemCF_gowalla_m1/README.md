
## ItemKNN_gowalla_x0 

A notebook to benchmark ItemKNN on gowalla_x0 dataset.

Author: Jinpeng Wang, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Logs](#Logs) | [Results](#Results)

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
    cd benchmarks/ItemKNN
    ```

2. Run the following script to reproduce the result

    ```python
    from ItemKNN import ItemKNN

    params = {"train_data": "../../data/Gowalla/gowalla_x0/train_enmf.txt",
              "test_data": "../../data/Gowalla/gowalla_x0/test_enmf.txt",
              "similarity_measure": "cosine", # searched in [pearson, cosine]
              "num_neighbors": 100, # searched in [10, 20, 50, 100, 150, 200]
              "min_similarity_threshold": 0,
              "renormalize_similarity": False,
              "enable_average_bias": True,
              "metrics": ["F1(k=20)", "Recall(k=20)", "Recall(k=50)", "NDCG(k=20)", "NDCG(k=50)", "HitRate(k=20)", "HitRate(k=50)"]}
    model = ItemKNN(params)
    model.fit()
    model.evaluate()
    ```

### Logs
```python
2020-11-04 11:28:06.282303 Params: {'train_data': '../../data/Gowalla/gowalla_x0/train_enmf.txt', 'test_data': '../../data/Gowalla/gowalla_x0/test_enmf.txt', 'similarity_measure': 'cosine', 'num_neighbors': 100, 'min_similarity_threshold': 0, 'renormalize_similarity': False, 'enable_average_bias': True, 'metrics': ['F1(k=20)', 'Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)', 'HitRate(k=50)']}
2020-11-04 11:28:06.282428 Reading dataset...
2020-11-04 11:28:07.492770 Number of users: 29858, number of items: 40981
2020-11-04 11:28:08.928641 Start computing similarity matrix...
2020-11-04 11:38:57.550758 Finished similarity matrix computation.
2020-11-04 11:38:57.609390 Start predicting preference...
2020-11-04 11:49:35.260725 Evaluating metrics for 29858 users...
2020-11-04 11:51:24.018456 [Metrics] F1(k=20): 0.065288 - Recall(k=20): 0.156994 - Recall(k=50): 0.254896 - NDCG(k=20): 0.121364 - NDCG(k=50): 0.152704 - HitRate(k=20): 0.509445 - HitRate(k=50): 0.665048
```

### Results
```python
[Metrics] F1(k=20): 0.065288 - Recall(k=20): 0.156994 - Recall(k=50): 0.254896 - NDCG(k=20): 0.121364 - NDCG(k=50): 0.152704 - HitRate(k=20): 0.509445 - HitRate(k=50): 0.665048
```
