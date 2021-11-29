## FLEN_Taobao_x0_001

A notebook to benchmark FLEN on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.194370 - AUC: 0.639660
```


### Logs
```python
2020-06-29 20:58:55,498 P6855 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_1fc975da",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1000, 1000]",
    "embedding_dim": "16",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'source': 'item', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'source': 'user', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand'], 'source': 'item', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'pid', 'source': 'context', 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'source': 'user', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'source': 'context', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'source': 'user', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FLEN",
    "model_id": "FLEN_taobao_x0_002_07952678",
    "model_root": "./Taobao/FLEN_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/Taobao_x0/test.csv",
    "train_data": "../data/Taobao/Taobao_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Taobao/Taobao_x0/test.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-29 20:58:55,499 P6855 INFO Set up feature encoder...
2020-06-29 20:58:55,499 P6855 INFO Reading file: ../data/Taobao/Taobao_x0/train.csv
2020-06-29 21:00:09,992 P6855 INFO Preprocess feature columns...
2020-06-29 21:00:54,595 P6855 INFO Fit feature encoder...
2020-06-29 21:00:54,596 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'userid', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:00,633 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'cms_segid', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:04,038 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'cms_group_id', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:07,313 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'final_gender_code', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:10,614 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'age_level', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:13,962 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'pvalue_level', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:16,502 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'shopping_level', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:19,813 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'occupation', 'source': 'user', 'type': 'categorical'}
2020-06-29 21:01:23,000 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'adgroup_id', 'source': 'item', 'type': 'categorical'}
2020-06-29 21:01:32,655 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'cate_id', 'source': 'item', 'type': 'categorical'}
2020-06-29 21:01:36,800 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'campaign_id', 'source': 'item', 'type': 'categorical'}
2020-06-29 21:01:45,557 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'customer', 'source': 'item', 'type': 'categorical'}
2020-06-29 21:01:53,958 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'brand', 'source': 'item', 'type': 'categorical'}
2020-06-29 21:01:59,207 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'pid', 'source': 'context', 'type': 'categorical'}
2020-06-29 21:02:02,752 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'source': 'context', 'type': 'categorical'}
2020-06-29 21:02:08,041 P6855 INFO Processing column: {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'source': 'user', 'type': 'sequence'}
2020-06-29 21:02:08,041 P6855 INFO Set feature index...
2020-06-29 21:02:08,041 P6855 INFO Pickle feature_encode: ../data/Taobao/taobao_x0_1fc975da/feature_encoder.pkl
2020-06-29 21:02:08,459 P6855 INFO Save feature_map to json: ../data/Taobao/taobao_x0_1fc975da/feature_map.json
2020-06-29 21:02:08,460 P6855 INFO Set feature encoder done.
2020-06-29 21:02:13,136 P6855 INFO Total number of parameters: 20117493.
2020-06-29 21:02:13,137 P6855 INFO Loading data...
2020-06-29 21:02:13,140 P6855 INFO Reading file: ../data/Taobao/Taobao_x0/train.csv
2020-06-29 21:03:27,930 P6855 INFO Preprocess feature columns...
2020-06-29 21:04:12,040 P6855 INFO Transform feature columns...
2020-06-29 21:08:42,411 P6855 INFO Saving data to h5: ../data/Taobao/taobao_x0_1fc975da/train.h5
2020-06-29 21:08:56,909 P6855 INFO Reading file: ../data/Taobao/Taobao_x0/test.csv
2020-06-29 21:09:07,061 P6855 INFO Preprocess feature columns...
2020-06-29 21:09:12,559 P6855 INFO Transform feature columns...
2020-06-29 21:09:52,132 P6855 INFO Saving data to h5: ../data/Taobao/taobao_x0_1fc975da/test.h5
2020-06-29 21:09:54,531 P6855 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-29 21:09:54,619 P6855 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-29 21:09:54,619 P6855 INFO Loading train data done.
2020-06-29 21:09:57,117 P6855 INFO Start training: 2371 batches/epoch
2020-06-29 21:09:57,117 P6855 INFO ************ Epoch=1 start ************
2020-06-29 21:14:28,649 P6855 INFO [Metrics] logloss: 0.194370 - AUC: 0.639660
2020-06-29 21:14:28,651 P6855 INFO Save best model: monitor(max): 0.445290
2020-06-29 21:14:28,725 P6855 INFO --- 2371/2371 batches finished ---
2020-06-29 21:14:28,971 P6855 INFO Train loss: 0.197617
2020-06-29 21:14:28,971 P6855 INFO ************ Epoch=1 end ************
2020-06-29 21:18:57,690 P6855 INFO [Metrics] logloss: 0.202697 - AUC: 0.629403
2020-06-29 21:18:57,691 P6855 INFO Monitor(max) STOP: 0.426706 !
2020-06-29 21:18:57,691 P6855 INFO Reduce learning rate on plateau: 0.000100
2020-06-29 21:18:57,692 P6855 INFO --- 2371/2371 batches finished ---
2020-06-29 21:18:57,941 P6855 INFO Train loss: 0.181291
2020-06-29 21:18:57,941 P6855 INFO ************ Epoch=2 end ************
2020-06-29 21:23:25,955 P6855 INFO [Metrics] logloss: 0.241611 - AUC: 0.607010
2020-06-29 21:23:25,957 P6855 INFO Monitor(max) STOP: 0.365399 !
2020-06-29 21:23:25,957 P6855 INFO Reduce learning rate on plateau: 0.000010
2020-06-29 21:23:25,957 P6855 INFO Early stopping at epoch=3
2020-06-29 21:23:25,957 P6855 INFO --- 2371/2371 batches finished ---
2020-06-29 21:23:26,209 P6855 INFO Train loss: 0.152987
2020-06-29 21:23:26,209 P6855 INFO Training finished.
2020-06-29 21:23:26,209 P6855 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/FLEN_taobao/taobao_x0_1fc975da/FLEN_taobao_x0_002_07952678_model.ckpt
2020-06-29 21:23:26,315 P6855 INFO ****** Train/validation evaluation ******
2020-06-29 21:23:43,904 P6855 INFO [Metrics] logloss: 0.194370 - AUC: 0.639660
2020-06-29 21:23:43,939 P6855 INFO ******** Test evaluation ********
2020-06-29 21:23:43,939 P6855 INFO Loading data...
2020-06-29 21:23:43,939 P6855 INFO Loading data from h5: ../data/Taobao/taobao_x0_1fc975da/test.h5
2020-06-29 21:23:44,797 P6855 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-29 21:23:44,798 P6855 INFO Loading test data done.
2020-06-29 21:24:03,315 P6855 INFO [Metrics] logloss: 0.194370 - AUC: 0.639660


```