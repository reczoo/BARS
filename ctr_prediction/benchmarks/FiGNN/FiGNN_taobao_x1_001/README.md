## FiGNN_Taobao_x0_001

A notebook to benchmark FiGNN on Taobao_x0_001 dataset.

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


### Code




### Results
```python
[Metrics] logloss: 0.193429 - AUC: 0.644102
```


### Logs
```python
2020-07-06 10:40:57,840 P34044 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gnn_layers": "3",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FiGNN",
    "model_id": "FiGNN_taobao_x0_005_eb86f676",
    "model_root": "./Taobao/FiGNN_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/Taobao_x0/test.csv",
    "train_data": "../data/Taobao/Taobao_x0/train.csv",
    "use_gru": "True",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Taobao/Taobao_x0/test.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-06 10:40:57,841 P34044 INFO Set up feature encoder...
2020-07-06 10:40:57,841 P34044 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-07-06 10:40:58,922 P34044 INFO Total number of parameters: 21668352.
2020-07-06 10:40:58,923 P34044 INFO Loading data...
2020-07-06 10:40:58,926 P34044 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-07-06 10:41:07,536 P34044 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-06 10:41:09,600 P34044 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-07-06 10:41:09,719 P34044 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-06 10:41:09,719 P34044 INFO Loading train data done.
2020-07-06 10:41:13,045 P34044 INFO Start training: 2371 batches/epoch
2020-07-06 10:41:13,046 P34044 INFO ************ Epoch=1 start ************
2020-07-06 10:50:21,577 P34044 INFO [Metrics] logloss: 0.193429 - AUC: 0.644102
2020-07-06 10:50:21,581 P34044 INFO Save best model: monitor(max): 0.450673
2020-07-06 10:50:21,676 P34044 INFO --- 2371/2371 batches finished ---
2020-07-06 10:50:21,740 P34044 INFO Train loss: 0.198877
2020-07-06 10:50:21,740 P34044 INFO ************ Epoch=1 end ************
2020-07-06 10:59:34,293 P34044 INFO [Metrics] logloss: 0.196430 - AUC: 0.637472
2020-07-06 10:59:34,295 P34044 INFO Monitor(max) STOP: 0.441042 !
2020-07-06 10:59:34,295 P34044 INFO Reduce learning rate on plateau: 0.000100
2020-07-06 10:59:34,295 P34044 INFO --- 2371/2371 batches finished ---
2020-07-06 10:59:34,361 P34044 INFO Train loss: 0.190839
2020-07-06 10:59:34,361 P34044 INFO ************ Epoch=2 end ************
2020-07-06 11:08:40,876 P34044 INFO [Metrics] logloss: 0.212093 - AUC: 0.626490
2020-07-06 11:08:40,878 P34044 INFO Monitor(max) STOP: 0.414397 !
2020-07-06 11:08:40,878 P34044 INFO Reduce learning rate on plateau: 0.000010
2020-07-06 11:08:40,878 P34044 INFO Early stopping at epoch=3
2020-07-06 11:08:40,878 P34044 INFO --- 2371/2371 batches finished ---
2020-07-06 11:08:40,930 P34044 INFO Train loss: 0.179178
2020-07-06 11:08:40,930 P34044 INFO Training finished.
2020-07-06 11:08:40,930 P34044 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/FiGNN_taobao/taobao_x0_87391c5c/FiGNN_taobao_x0_005_eb86f676_model.ckpt
2020-07-06 11:08:41,058 P34044 INFO ****** Train/validation evaluation ******
2020-07-06 11:09:05,779 P34044 INFO [Metrics] logloss: 0.193429 - AUC: 0.644102
2020-07-06 11:09:05,846 P34044 INFO ******** Test evaluation ********
2020-07-06 11:09:05,846 P34044 INFO Loading data...
2020-07-06 11:09:05,846 P34044 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-06 11:09:06,914 P34044 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-06 11:09:06,915 P34044 INFO Loading test data done.
2020-07-06 11:09:31,644 P34044 INFO [Metrics] logloss: 0.193429 - AUC: 0.644102
```