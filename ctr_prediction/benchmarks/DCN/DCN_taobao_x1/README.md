## DCN_Taobao_x0_001

A notebook to benchmark DCN on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193778 - AUC: 0.642852
```

### Logs
```python
2020-06-22 22:57:04,498 P16248 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "3",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_9fefb51c",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[500, 500]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DCN",
    "model_id": "DCN_taobao_x0_016_124de1c1",
    "model_root": "./Taobao/DCN_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
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
2020-06-22 22:57:04,498 P16248 INFO Set up feature encoder...
2020-06-22 22:57:04,499 P16248 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_9fefb51c/feature_encoder.pkl
2020-06-22 22:57:05,381 P16248 INFO Total number of parameters: 20961005.
2020-06-22 22:57:05,381 P16248 INFO Loading data...
2020-06-22 22:57:05,383 P16248 INFO Loading data from h5: ../data/Taobao/taobao_x0_9fefb51c/train.h5
2020-06-22 22:57:12,156 P16248 INFO Loading data from h5: ../data/Taobao/taobao_x0_9fefb51c/test.h5
2020-06-22 22:57:13,674 P16248 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-22 22:57:13,757 P16248 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-22 22:57:13,757 P16248 INFO Loading train data done.
2020-06-22 22:57:17,719 P16248 INFO Start training: 2371 batches/epoch
2020-06-22 22:57:17,719 P16248 INFO ************ Epoch=1 start ************
2020-06-22 23:05:12,018 P16248 INFO [Metrics] logloss: 0.193778 - AUC: 0.642852
2020-06-22 23:05:12,021 P16248 INFO Save best model: monitor(max): 0.449074
2020-06-22 23:05:12,102 P16248 INFO --- 2371/2371 batches finished ---
2020-06-22 23:05:12,148 P16248 INFO Train loss: 0.202951
2020-06-22 23:05:12,148 P16248 INFO ************ Epoch=1 end ************
2020-06-22 23:12:55,465 P16248 INFO [Metrics] logloss: 0.196382 - AUC: 0.636106
2020-06-22 23:12:55,467 P16248 INFO Monitor(max) STOP: 0.439723 !
2020-06-22 23:12:55,467 P16248 INFO Reduce learning rate on plateau: 0.000100
2020-06-22 23:12:55,467 P16248 INFO --- 2371/2371 batches finished ---
2020-06-22 23:12:55,515 P16248 INFO Train loss: 0.196262
2020-06-22 23:12:55,516 P16248 INFO ************ Epoch=2 end ************
2020-06-22 23:20:23,847 P16248 INFO [Metrics] logloss: 0.204402 - AUC: 0.629005
2020-06-22 23:20:23,850 P16248 INFO Monitor(max) STOP: 0.424602 !
2020-06-22 23:20:23,851 P16248 INFO Reduce learning rate on plateau: 0.000010
2020-06-22 23:20:23,851 P16248 INFO Early stopping at epoch=3
2020-06-22 23:20:23,851 P16248 INFO --- 2371/2371 batches finished ---
2020-06-22 23:20:23,897 P16248 INFO Train loss: 0.184127
2020-06-22 23:20:23,898 P16248 INFO Training finished.
2020-06-22 23:20:23,898 P16248 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/DCN_taobao/taobao_x0_9fefb51c/DCN_taobao_x0_016_124de1c1_model.ckpt
2020-06-22 23:20:24,082 P16248 INFO ****** Train/validation evaluation ******
2020-06-22 23:20:42,635 P16248 INFO [Metrics] logloss: 0.193778 - AUC: 0.642852
2020-06-22 23:20:42,680 P16248 INFO ******** Test evaluation ********
2020-06-22 23:20:42,680 P16248 INFO Loading data...
2020-06-22 23:20:42,680 P16248 INFO Loading data from h5: ../data/Taobao/taobao_x0_9fefb51c/test.h5
2020-06-22 23:20:43,655 P16248 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-22 23:20:43,655 P16248 INFO Loading test data done.
2020-06-22 23:21:02,932 P16248 INFO [Metrics] logloss: 0.193778 - AUC: 0.642852


```