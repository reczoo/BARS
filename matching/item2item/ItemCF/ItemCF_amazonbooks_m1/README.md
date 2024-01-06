
## ItemKNN_amazonbooks_x0 

A notebook to benchmark ItemKNN on amazonbooks_x0 dataset.

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
amazonbooks_x0 follows the data split and preprocessing steps in NGCF and LightGCN. We directly transform the formats of the data from their [repo](https://github.com/kuandeng/LightGCN/tree/master/Data).


### Code

1. Downalod the dataset from LightGCN repo and run the preprocessing script for format transformation

    ```python
    cd data/AmazonBooks/amazonbooks_x0
    python ENMF_data_process.py
    cd benchmarks/ItemKNN
    ```

2. Run the following script to reproduce the result
    ```python
    from ItemKNN import ItemKNN

    params = {"train_data": "../../data/AmazonBooks/amazonbooks_x0/train_enmf.txt",
              "test_data": "../../data/AmazonBooks/amazonbooks_x0/test_enmf.txt",
              "similarity_measure": "pearson", # searched in [pearson, cosine]
              "num_neighbors": 10, # searched in [5, 10, 20, 50, 100, 150, 200]
              "min_similarity_threshold": 0,
              "renormalize_similarity": False,
              "enable_average_bias": True,
              "metrics": ["Recall(k=20)", "Recall(k=50)", "NDCG(k=20)", "NDCG(k=50)", "HitRate(k=20)", "HitRate(k=50)"]}
    model = ItemKNN(params)
    model.fit()
    model.evaluate()
    ```

### Logs
```python
2020-11-04 13:19:02.423178 Params: {'train_data': '../../data/AmazonBooks/amazonbooks_x0/train_enmf.txt', 'test_data': '../../data/AmazonBooks/amazonbooks_x0/test_enmf.txt', 'similarity_measure': 'pearson', 'num_neighbors': 10, 'min_similarity_threshold': 0, 'renormalize_similarity': False, 'enable_average_bias': True, 'metrics': ['Recall(k=20)', 'Recall(k=50)', 'NDCG(k=20)', 'NDCG(k=50)', 'HitRate(k=20)', 'HitRate(k=50)']}
2020-11-04 13:19:02.423251 Reading dataset...
2020-11-04 13:19:03.179263 Number of users: 52643, number of items: 91599
2020-11-04 13:19:03.809791 Start computing similarity matrix...
2020-11-04 14:16:10.129253 Finished similarity matrix computation.
2020-11-04 14:16:10.349181 Start predicting preference...
2020-11-04 15:09:53.323412 Evaluating metrics for 52643 users...
2020-11-04 15:16:26.010032 [Metrics] Recall(k=20): 0.073649 - Recall(k=50): 0.117522 - NDCG(k=20): 0.060600 - NDCG(k=50): 0.077055 - HitRate(k=20): 0.376536 - HitRate(k=50): 0.523431
```

### Results
```python
[Metrics] Recall(k=20): 0.073649 - Recall(k=50): 0.117522 - NDCG(k=20): 0.060600 - NDCG(k=50): 0.077055 - HitRate(k=20): 0.376536 - HitRate(k=50): 0.523431
```
