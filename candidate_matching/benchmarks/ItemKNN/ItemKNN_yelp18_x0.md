
## ItemKNN_yelp18_x0 

A notebook to benchmark ItemKNN on yelp18_x0 dataset.

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
yelp18_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code

1. Downalod the dataset from LightGCN repo and run the preprocessing script for format transformation

    ```python
    cd data/Yelp18/yelp18_x0
    python ENMF_data_process.py
    cd benchmarks/ItemKNN
    ```

2. Run the following script to reproduce the result

    ```python
    params = {"train_data": "../../data/Yelp18/yelp18_x0/train_enmf.txt",
              "test_data": "../../data/Yelp18/yelp18_x0/test_enmf.txt",
              "similarity_measure": "pearson", # searched in [pearson, cosine]
              "num_neighbors": 150, # searched in [10, 20, 50, 100, 150, 200]
              "min_similarity_threshold": 0,
              "renormalize_similarity": False,
              "enable_average_bias": True,
              "metrics": ['Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)', 'HitRate(k=50)']}
    model = ItemKNN(params)
    model.fit()
    model.evaluate()
    ```

### Logs

```python
2020-11-04 13:23:01.382845 Params: {'train_data': '../../data/Yelp18/yelp18_x0/train_enmf.txt', 'test_data': '../../data/Yelp18/yelp18_x0/test_enmf.txt', 'similarity_measure': 'pearson', 'num_neighbors': 150, 'min_similarity_threshold': 0, 'renormalize_similarity': False, 'enable_average_bias': True, 'metrics': ['Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)', 'HitRate(k=50)']}
2020-11-04 13:23:01.382983 Reading dataset...
2020-11-04 13:23:02.456546 Number of users: 31668, number of items: 38048
2020-11-04 13:23:03.536160 Start computing similarity matrix...
2020-11-04 13:34:40.975208 Finished similarity matrix computation.
2020-11-04 13:34:41.100311 Start predicting preference...
2020-11-04 13:49:51.267282 Evaluating metrics for 31668 users...
2020-11-04 13:56:25.141725 [Metrics] Recall(k=20): 0.063913 - Recall(k=50): 0.121876 - NDCG(k=20): 0.053094 - NDCG(k=50): 0.074579 - HitRate(k=20): 0.387552 - HitRate(k=50): 0.575313
```

### Results
```python
[Metrics] Recall(k=20): 0.063913 - Recall(k=50): 0.121876 - NDCG(k=20): 0.053094 - NDCG(k=50): 0.074579 - HitRate(k=20): 0.387552 - HitRate(k=50): 0.575313
```
