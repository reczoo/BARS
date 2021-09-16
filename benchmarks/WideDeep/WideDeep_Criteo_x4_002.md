## WideDeep_Criteo_x4_002

A notebook to benchmark WideDeep on Criteo_x4_002 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=2.

We fix embedding_dim=40 in this setting.
### Code




### Results
```python
[Metrics] logloss: 0.438892 - AUC: 0.812913
```


### Logs
```python
2020-03-04 12:13:56,397 P1308 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-6)",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "2",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "WideDeep",
    "model_id": "WideDeep_criteo_x4_007_5305f0a1",
    "model_root": "./Criteo/WideDeep_criteo/",
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
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-03-04 12:13:56,399 P1308 INFO Set up feature encoder...
2020-03-04 12:13:56,400 P1308 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-03-04 12:13:56,400 P1308 INFO Loading data...
2020-03-04 12:13:56,422 P1308 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-03-04 12:14:03,056 P1308 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-03-04 12:14:04,852 P1308 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-03-04 12:14:04,975 P1308 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-04 12:14:04,975 P1308 INFO Loading train data done.
2020-03-04 12:14:16,494 P1308 INFO **** Start training: 3668 batches/epoch ****
2020-03-04 12:25:02,021 P1308 INFO [Metrics] logloss: 0.443056 - AUC: 0.808503
2020-03-04 12:25:02,084 P1308 INFO Save best model: monitor(max): 0.365447
2020-03-04 12:25:03,715 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:25:03,792 P1308 INFO Train loss: 0.456710
2020-03-04 12:25:03,792 P1308 INFO ************ Epoch=1 end ************
2020-03-04 12:35:40,277 P1308 INFO [Metrics] logloss: 0.440805 - AUC: 0.811012
2020-03-04 12:35:40,337 P1308 INFO Save best model: monitor(max): 0.370207
2020-03-04 12:35:42,167 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:35:42,251 P1308 INFO Train loss: 0.449778
2020-03-04 12:35:42,251 P1308 INFO ************ Epoch=2 end ************
2020-03-04 12:46:18,349 P1308 INFO [Metrics] logloss: 0.439688 - AUC: 0.812136
2020-03-04 12:46:18,418 P1308 INFO Save best model: monitor(max): 0.372448
2020-03-04 12:46:20,265 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:46:20,335 P1308 INFO Train loss: 0.447703
2020-03-04 12:46:20,335 P1308 INFO ************ Epoch=3 end ************
2020-03-04 12:56:57,083 P1308 INFO [Metrics] logloss: 0.439318 - AUC: 0.812417
2020-03-04 12:56:57,145 P1308 INFO Save best model: monitor(max): 0.373100
2020-03-04 12:56:58,993 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:56:59,060 P1308 INFO Train loss: 0.446406
2020-03-04 12:56:59,060 P1308 INFO ************ Epoch=4 end ************
2020-03-04 13:07:37,644 P1308 INFO [Metrics] logloss: 0.440403 - AUC: 0.811830
2020-03-04 13:07:37,715 P1308 INFO Monitor(max) STOP: 0.371428 !
2020-03-04 13:07:37,715 P1308 INFO Reduce learning rate on plateau: 0.000100
2020-03-04 13:07:37,715 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 13:07:37,795 P1308 INFO Train loss: 0.445365
2020-03-04 13:07:37,795 P1308 INFO ************ Epoch=5 end ************
2020-03-04 13:18:13,294 P1308 INFO [Metrics] logloss: 0.450589 - AUC: 0.805841
2020-03-04 13:18:13,357 P1308 INFO Monitor(max) STOP: 0.355253 !
2020-03-04 13:18:13,357 P1308 INFO Reduce learning rate on plateau: 0.000010
2020-03-04 13:18:13,357 P1308 INFO Early stopping at epoch=6
2020-03-04 13:18:13,357 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 13:18:13,426 P1308 INFO Train loss: 0.417406
2020-03-04 13:18:13,426 P1308 INFO Training finished.
2020-03-04 13:18:13,426 P1308 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/WideDeep_criteo/criteo_x4_001_be98441d/WideDeep_criteo_x4_007_5305f0a1_criteo_x4_001_be98441d_model.ckpt
2020-03-04 13:18:15,161 P1308 INFO ****** Train/validation evaluation ******
2020-03-04 13:23:26,752 P1308 INFO [Metrics] logloss: 0.419569 - AUC: 0.833701
2020-03-04 13:24:02,752 P1308 INFO [Metrics] logloss: 0.439318 - AUC: 0.812417
2020-03-04 13:24:02,949 P1308 INFO ******** Test evaluation ********
2020-03-04 13:24:02,949 P1308 INFO Loading data...
2020-03-04 13:24:02,949 P1308 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-03-04 13:24:04,114 P1308 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-04 13:24:04,115 P1308 INFO Loading test data done.
2020-03-04 13:24:36,674 P1308 INFO [Metrics] logloss: 0.438892 - AUC: 0.812913

```