## FiGNN_Avazu_x4_002

A notebook to benchmark FiGNN on Avazu_x4_002 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default <OOV> token. Note that we found that min_category_count=1 performs the best, which is surprising.

We fix embedding_dim=40 following the existing FGCNN work.
### Code




### Results
```python
[Metrics] logloss: 0.371134 - AUC: 0.794416
```


### Logs
```python
2020-01-23 22:24:57,876 P55093 INFO {
    "batch_size": "2000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gnn_layers": "10",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiGNN",
    "model_id": "FiGNN_avazu_x4_005_eb8e07e1",
    "model_root": "./Avazu/FiGNN_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_gru": "True",
    "use_hdf5": "True",
    "use_residual": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "min_categr_count": "1",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "version": "pytorch",
    "gpu": "1"
}
2020-01-23 22:24:57,877 P55093 INFO Set up feature encoder...
2020-01-23 22:24:57,877 P55093 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x4_001_d45ad60e/feature_encoder.pkl
2020-01-23 22:25:04,653 P55093 INFO Loading data...
2020-01-23 22:25:04,658 P55093 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-01-23 22:25:07,477 P55093 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-01-23 22:25:08,801 P55093 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-01-23 22:25:08,973 P55093 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-01-23 22:25:08,973 P55093 INFO Loading train data done.
2020-01-23 22:25:19,691 P55093 INFO **** Start training: 16172 batches/epoch ****
2020-01-24 00:50:38,576 P55093 INFO [Metrics] logloss: 0.371311 - AUC: 0.794095
2020-01-24 00:50:38,685 P55093 INFO Save best model: monitor(max): 0.422783
2020-01-24 00:50:40,105 P55093 INFO --- 16172/16172 batches finished ---
2020-01-24 00:50:40,167 P55093 INFO Train loss: 0.379974
2020-01-24 00:50:40,167 P55093 INFO ************ Epoch=1 end ************
2020-01-24 03:16:00,678 P55093 INFO [Metrics] logloss: 0.415388 - AUC: 0.763071
2020-01-24 03:16:00,778 P55093 INFO Monitor(max) STOP: 0.347683 !
2020-01-24 03:16:00,778 P55093 INFO Reduce learning rate on plateau: 0.000100
2020-01-24 03:16:00,778 P55093 INFO --- 16172/16172 batches finished ---
2020-01-24 03:16:00,865 P55093 INFO Train loss: 0.286294
2020-01-24 03:16:00,865 P55093 INFO ************ Epoch=2 end ************
2020-01-24 05:41:22,050 P55093 INFO [Metrics] logloss: 0.470917 - AUC: 0.749049
2020-01-24 05:41:22,151 P55093 INFO Monitor(max) STOP: 0.278132 !
2020-01-24 05:41:22,151 P55093 INFO Reduce learning rate on plateau: 0.000010
2020-01-24 05:41:22,151 P55093 INFO Early stopping at epoch=3
2020-01-24 05:41:22,151 P55093 INFO --- 16172/16172 batches finished ---
2020-01-24 05:41:22,270 P55093 INFO Train loss: 0.257096
2020-01-24 05:41:22,270 P55093 INFO Training finished.
2020-01-24 05:41:22,271 P55093 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Avazu/FiGNN_avazu/avazu_x4_001_d45ad60e/FiGNN_avazu_x4_005_eb8e07e1_avazu_x4_001_d45ad60e_model.ckpt
2020-01-24 05:41:24,113 P55093 INFO ****** Train/validation evaluation ******
2020-01-24 06:21:50,300 P55093 INFO [Metrics] logloss: 0.326500 - AUC: 0.859686
2020-01-24 06:26:54,399 P55093 INFO [Metrics] logloss: 0.371311 - AUC: 0.794095
2020-01-24 06:26:54,657 P55093 INFO ******** Test evaluation ********
2020-01-24 06:26:54,657 P55093 INFO Loading data...
2020-01-24 06:26:54,657 P55093 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-01-24 06:26:55,138 P55093 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-01-24 06:26:55,138 P55093 INFO Loading test data done.
2020-01-24 06:31:58,643 P55093 INFO [Metrics] logloss: 0.371134 - AUC: 0.794416


```
