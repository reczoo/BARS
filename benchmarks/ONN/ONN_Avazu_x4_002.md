## ONN_Avazu_x4_002

A notebook to benchmark ONN on Avazu_x4_002 dataset.

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
[Metrics] logloss: 0.367666 - AUC: 0.800119
```


### Logs
```python
2020-02-04 04:28:37,379 P3555 INFO {
    "batch_norm": "True",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "2",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "ONN",
    "model_id": "ONN_avazu_x4_036_15e5fc4d",
    "model_root": "./Avazu/ONN_avazu/",
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
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-02-04 04:28:37,384 P3555 INFO Set up feature encoder...
2020-02-04 04:28:37,384 P3555 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-04 04:28:37,384 P3555 INFO Loading data...
2020-02-04 04:28:37,388 P3555 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-04 04:29:19,590 P3555 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-04 04:29:26,580 P3555 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-04 04:29:26,732 P3555 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-04 04:29:26,733 P3555 INFO Loading train data done.
2020-02-04 04:29:54,681 P3555 INFO **** Start training: 3235 batches/epoch ****
2020-02-04 06:21:28,582 P3555 INFO [Metrics] logloss: 0.367837 - AUC: 0.799815
2020-02-04 06:21:28,676 P3555 INFO Save best model: monitor(max): 0.431978
2020-02-04 06:21:29,946 P3555 INFO --- 3235/3235 batches finished ---
2020-02-04 06:21:30,097 P3555 INFO Train loss: 0.377756
2020-02-04 06:21:30,097 P3555 INFO ************ Epoch=1 end ************
2020-02-04 08:13:22,723 P3555 INFO [Metrics] logloss: 0.402126 - AUC: 0.770331
2020-02-04 08:13:22,789 P3555 INFO Monitor(max) STOP: 0.368204 !
2020-02-04 08:13:22,790 P3555 INFO Reduce learning rate on plateau: 0.000100
2020-02-04 08:13:22,790 P3555 INFO --- 3235/3235 batches finished ---
2020-02-04 08:13:22,958 P3555 INFO Train loss: 0.268413
2020-02-04 08:13:22,958 P3555 INFO ************ Epoch=2 end ************
2020-02-04 10:05:04,068 P3555 INFO [Metrics] logloss: 0.525691 - AUC: 0.759325
2020-02-04 10:05:04,160 P3555 INFO Monitor(max) STOP: 0.233633 !
2020-02-04 10:05:04,160 P3555 INFO Reduce learning rate on plateau: 0.000010
2020-02-04 10:05:04,160 P3555 INFO Early stopping at epoch=3
2020-02-04 10:05:04,160 P3555 INFO --- 3235/3235 batches finished ---
2020-02-04 10:05:04,375 P3555 INFO Train loss: 0.221371
2020-02-04 10:05:04,376 P3555 INFO Training finished.
2020-02-04 10:05:04,376 P3555 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Avazu/ONN_avazu/avazu_x4_001_d45ad60e/ONN_avazu_x4_036_15e5fc4d_avazu_x4_001_d45ad60e_model.ckpt
2020-02-04 10:05:06,670 P3555 INFO ****** Train/validation evaluation ******
2020-02-04 10:12:01,235 P3555 INFO [Metrics] logloss: 0.310772 - AUC: 0.881562
2020-02-04 10:12:46,289 P3555 INFO [Metrics] logloss: 0.367837 - AUC: 0.799815
2020-02-04 10:12:47,091 P3555 INFO ******** Test evaluation ********
2020-02-04 10:12:47,091 P3555 INFO Loading data...
2020-02-04 10:12:47,091 P3555 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-04 10:12:47,682 P3555 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-04 10:12:47,682 P3555 INFO Loading test data done.
2020-02-04 10:13:29,817 P3555 INFO [Metrics] logloss: 0.367666 - AUC: 0.800119

```
