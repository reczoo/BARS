## ONN_Avazu_x4_001 

A notebook to benchmark ONN on Avazu_x4_001 dataset.

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
In this setting, we preprocess the data split by removing the ``id`` field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default ``<OOV>`` token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless ``id`` field nor specially preprocess the timestamp field.

To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.


### Code


### Results
```python
[Metrics] logloss: 0.368328 - AUC: 0.799150
```


### Logs
```python
2020-07-17 20:42:09,450 P33536 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "8",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "ONN",
    "model_id": "ONN_avazu_x4_3bbbc4c9_006_d669ec93",
    "model_root": "./Avazu/ONN_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-17 20:42:09,451 P33536 INFO Set up feature encoder...
2020-07-17 20:42:09,451 P33536 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-17 20:42:37,582 P33536 INFO Total number of parameters: 723660201.
2020-07-17 20:42:37,583 P33536 INFO Loading data...
2020-07-17 20:42:37,587 P33536 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-17 20:42:44,520 P33536 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-17 20:42:46,077 P33536 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-17 20:42:46,186 P33536 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-17 20:42:46,187 P33536 INFO Loading train data done.
2020-07-17 20:43:19,816 P33536 INFO Start training: 3235 batches/epoch
2020-07-17 20:43:19,816 P33536 INFO ************ Epoch=1 start ************
2020-07-17 22:44:36,166 P33536 INFO [Metrics] logloss: 0.368429 - AUC: 0.798979
2020-07-17 22:44:36,167 P33536 INFO Save best model: monitor(max): 0.430550
2020-07-17 22:44:38,918 P33536 INFO --- 3235/3235 batches finished ---
2020-07-17 22:44:39,066 P33536 INFO Train loss: 0.377100
2020-07-17 22:44:39,066 P33536 INFO ************ Epoch=1 end ************
2020-07-18 00:46:03,776 P33536 INFO [Metrics] logloss: 0.387986 - AUC: 0.787972
2020-07-18 00:46:03,781 P33536 INFO Monitor(max) STOP: 0.399986 !
2020-07-18 00:46:03,781 P33536 INFO Reduce learning rate on plateau: 0.000100
2020-07-18 00:46:03,781 P33536 INFO --- 3235/3235 batches finished ---
2020-07-18 00:46:03,935 P33536 INFO Train loss: 0.306951
2020-07-18 00:46:03,935 P33536 INFO ************ Epoch=2 end ************
2020-07-18 02:47:23,378 P33536 INFO [Metrics] logloss: 0.477172 - AUC: 0.768023
2020-07-18 02:47:23,382 P33536 INFO Monitor(max) STOP: 0.290850 !
2020-07-18 02:47:23,382 P33536 INFO Reduce learning rate on plateau: 0.000010
2020-07-18 02:47:23,382 P33536 INFO Early stopping at epoch=3
2020-07-18 02:47:23,382 P33536 INFO --- 3235/3235 batches finished ---
2020-07-18 02:47:23,532 P33536 INFO Train loss: 0.251349
2020-07-18 02:47:23,532 P33536 INFO Training finished.
2020-07-18 02:47:23,532 P33536 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Avazu/ONN_avazu/min2/avazu_x4_3bbbc4c9/ONN_avazu_x4_3bbbc4c9_006_d669ec93_model.ckpt
2020-07-18 02:47:27,255 P33536 INFO ****** Train/validation evaluation ******
2020-07-18 02:52:42,613 P33536 INFO [Metrics] logloss: 0.322398 - AUC: 0.866123
2020-07-18 02:53:13,833 P33536 INFO [Metrics] logloss: 0.368429 - AUC: 0.798979
2020-07-18 02:53:13,896 P33536 INFO ******** Test evaluation ********
2020-07-18 02:53:13,896 P33536 INFO Loading data...
2020-07-18 02:53:13,896 P33536 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-18 02:53:14,520 P33536 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-18 02:53:14,521 P33536 INFO Loading test data done.
2020-07-18 02:53:45,602 P33536 INFO [Metrics] logloss: 0.368328 - AUC: 0.799150
```
