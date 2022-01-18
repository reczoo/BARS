## DeepFM_Criteo_x0_001

A notebook to benchmark DeepFM on Criteo_x0_001 dataset.

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

This dataset split follows the setting in the AFN work. That is, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. The data preprocessing script is provided on Github and we directly download the preprocessed data.

Reproducing steps:
Step1: Download the preprocessed data via the [https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Avazu_x0/download_criteo_x0.py](script).

Criteo_x0_001
In this setting, we follow the AFN work to fix embedding_dim=16, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons.

### Code



### Results
```python
[Metrics] AUC: 0.813766 - logloss: 0.438075
```


### Logs
```python
2021-01-07 09:05:42,513 P20355 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x0_ace9c1b9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepFM",
    "model_id": "DeepFM_criteo_x0_005_e29587dc",
    "model_root": "./Criteo/DeepFM_criteo_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x0/test.csv",
    "train_data": "../data/Criteo/Criteo_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-01-07 09:05:42,513 P20355 INFO Set up feature encoder...
2021-01-07 09:05:42,513 P20355 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x0_ace9c1b9/feature_encoder.pkl
2021-01-07 09:05:44,949 P20355 INFO Total number of parameters: 23429477.
2021-01-07 09:05:44,950 P20355 INFO Loading data...
2021-01-07 09:05:44,952 P20355 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/train.h5
2021-01-07 09:05:50,462 P20355 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/valid.h5
2021-01-07 09:05:51,991 P20355 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2021-01-07 09:05:51,991 P20355 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2021-01-07 09:05:51,991 P20355 INFO Loading train data done.
2021-01-07 09:05:55,302 P20355 INFO Start training: 8058 batches/epoch
2021-01-07 09:05:55,302 P20355 INFO ************ Epoch=1 start ************
2021-01-07 09:53:18,677 P20355 INFO [Metrics] AUC: 0.802982 - logloss: 0.448156
2021-01-07 09:53:18,678 P20355 INFO Save best model: monitor(max): 0.802982
2021-01-07 09:53:18,954 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 09:53:19,035 P20355 INFO Train loss: 0.464105
2021-01-07 09:53:19,035 P20355 INFO ************ Epoch=1 end ************
2021-01-07 10:40:44,669 P20355 INFO [Metrics] AUC: 0.805123 - logloss: 0.446193
2021-01-07 10:40:44,670 P20355 INFO Save best model: monitor(max): 0.805123
2021-01-07 10:40:44,834 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 10:40:44,903 P20355 INFO Train loss: 0.458788
2021-01-07 10:40:44,903 P20355 INFO ************ Epoch=2 end ************
2021-01-07 11:28:05,805 P20355 INFO [Metrics] AUC: 0.806539 - logloss: 0.444836
2021-01-07 11:28:05,807 P20355 INFO Save best model: monitor(max): 0.806539
2021-01-07 11:28:05,984 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 11:28:06,077 P20355 INFO Train loss: 0.457252
2021-01-07 11:28:06,077 P20355 INFO ************ Epoch=3 end ************
2021-01-07 12:15:22,950 P20355 INFO [Metrics] AUC: 0.807059 - logloss: 0.444371
2021-01-07 12:15:22,951 P20355 INFO Save best model: monitor(max): 0.807059
2021-01-07 12:15:23,126 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 12:15:23,193 P20355 INFO Train loss: 0.456450
2021-01-07 12:15:23,194 P20355 INFO ************ Epoch=4 end ************
2021-01-07 13:02:34,433 P20355 INFO [Metrics] AUC: 0.807667 - logloss: 0.443869
2021-01-07 13:02:34,434 P20355 INFO Save best model: monitor(max): 0.807667
2021-01-07 13:02:34,603 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 13:02:34,703 P20355 INFO Train loss: 0.455951
2021-01-07 13:02:34,703 P20355 INFO ************ Epoch=5 end ************
2021-01-07 13:49:50,058 P20355 INFO [Metrics] AUC: 0.808079 - logloss: 0.443509
2021-01-07 13:49:50,059 P20355 INFO Save best model: monitor(max): 0.808079
2021-01-07 13:49:50,222 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 13:49:50,303 P20355 INFO Train loss: 0.455585
2021-01-07 13:49:50,303 P20355 INFO ************ Epoch=6 end ************
2021-01-07 14:37:04,109 P20355 INFO [Metrics] AUC: 0.808280 - logloss: 0.443244
2021-01-07 14:37:04,111 P20355 INFO Save best model: monitor(max): 0.808280
2021-01-07 14:37:04,282 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 14:37:04,351 P20355 INFO Train loss: 0.455306
2021-01-07 14:37:04,351 P20355 INFO ************ Epoch=7 end ************
2021-01-07 15:23:51,249 P20355 INFO [Metrics] AUC: 0.808496 - logloss: 0.443116
2021-01-07 15:23:51,251 P20355 INFO Save best model: monitor(max): 0.808496
2021-01-07 15:23:51,404 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 15:23:51,482 P20355 INFO Train loss: 0.455077
2021-01-07 15:23:51,482 P20355 INFO ************ Epoch=8 end ************
2021-01-07 16:12:54,803 P20355 INFO [Metrics] AUC: 0.808663 - logloss: 0.442906
2021-01-07 16:12:54,805 P20355 INFO Save best model: monitor(max): 0.808663
2021-01-07 16:12:54,947 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 16:12:55,027 P20355 INFO Train loss: 0.454941
2021-01-07 16:12:55,027 P20355 INFO ************ Epoch=9 end ************
2021-01-07 17:04:58,871 P20355 INFO [Metrics] AUC: 0.808699 - logloss: 0.442868
2021-01-07 17:04:58,873 P20355 INFO Save best model: monitor(max): 0.808699
2021-01-07 17:04:59,022 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 17:04:59,124 P20355 INFO Train loss: 0.454738
2021-01-07 17:04:59,124 P20355 INFO ************ Epoch=10 end ************
2021-01-07 17:58:22,982 P20355 INFO [Metrics] AUC: 0.808719 - logloss: 0.442782
2021-01-07 17:58:22,983 P20355 INFO Save best model: monitor(max): 0.808719
2021-01-07 17:58:23,150 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 17:58:23,239 P20355 INFO Train loss: 0.454626
2021-01-07 17:58:23,239 P20355 INFO ************ Epoch=11 end ************
2021-01-07 18:47:24,835 P20355 INFO [Metrics] AUC: 0.809056 - logloss: 0.442497
2021-01-07 18:47:24,837 P20355 INFO Save best model: monitor(max): 0.809056
2021-01-07 18:47:24,984 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 18:47:25,057 P20355 INFO Train loss: 0.454530
2021-01-07 18:47:25,057 P20355 INFO ************ Epoch=12 end ************
2021-01-07 19:36:29,597 P20355 INFO [Metrics] AUC: 0.808998 - logloss: 0.442563
2021-01-07 19:36:29,598 P20355 INFO Monitor(max) STOP: 0.808998 !
2021-01-07 19:36:29,598 P20355 INFO Reduce learning rate on plateau: 0.000100
2021-01-07 19:36:29,598 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 19:36:29,676 P20355 INFO Train loss: 0.454411
2021-01-07 19:36:29,676 P20355 INFO ************ Epoch=13 end ************
2021-01-07 20:25:29,834 P20355 INFO [Metrics] AUC: 0.812913 - logloss: 0.438966
2021-01-07 20:25:29,836 P20355 INFO Save best model: monitor(max): 0.812913
2021-01-07 20:25:29,990 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 20:25:30,080 P20355 INFO Train loss: 0.442469
2021-01-07 20:25:30,081 P20355 INFO ************ Epoch=14 end ************
2021-01-07 21:14:11,115 P20355 INFO [Metrics] AUC: 0.813396 - logloss: 0.438551
2021-01-07 21:14:11,116 P20355 INFO Save best model: monitor(max): 0.813396
2021-01-07 21:14:11,282 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 21:14:11,363 P20355 INFO Train loss: 0.437978
2021-01-07 21:14:11,364 P20355 INFO ************ Epoch=15 end ************
2021-01-07 22:03:51,879 P20355 INFO [Metrics] AUC: 0.813447 - logloss: 0.438538
2021-01-07 22:03:51,880 P20355 INFO Save best model: monitor(max): 0.813447
2021-01-07 22:03:52,037 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 22:03:52,107 P20355 INFO Train loss: 0.436183
2021-01-07 22:03:52,107 P20355 INFO ************ Epoch=16 end ************
2021-01-07 22:54:30,365 P20355 INFO [Metrics] AUC: 0.813323 - logloss: 0.438692
2021-01-07 22:54:30,366 P20355 INFO Monitor(max) STOP: 0.813323 !
2021-01-07 22:54:30,366 P20355 INFO Reduce learning rate on plateau: 0.000010
2021-01-07 22:54:30,366 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 22:54:30,449 P20355 INFO Train loss: 0.434855
2021-01-07 22:54:30,449 P20355 INFO ************ Epoch=17 end ************
2021-01-07 23:45:07,889 P20355 INFO [Metrics] AUC: 0.812838 - logloss: 0.439500
2021-01-07 23:45:07,890 P20355 INFO Monitor(max) STOP: 0.812838 !
2021-01-07 23:45:07,890 P20355 INFO Reduce learning rate on plateau: 0.000001
2021-01-07 23:45:07,891 P20355 INFO Early stopping at epoch=18
2021-01-07 23:45:07,891 P20355 INFO --- 8058/8058 batches finished ---
2021-01-07 23:45:07,968 P20355 INFO Train loss: 0.430421
2021-01-07 23:45:07,969 P20355 INFO Training finished.
2021-01-07 23:45:07,969 P20355 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Criteo/DeepFM_criteo_x0/criteo_x0_ace9c1b9/DeepFM_criteo_x0_005_e29587dc_model.ckpt
2021-01-07 23:45:08,374 P20355 INFO ****** Train/validation evaluation ******
2021-01-07 23:45:45,784 P20355 INFO [Metrics] AUC: 0.813447 - logloss: 0.438538
2021-01-07 23:45:45,826 P20355 INFO ******** Test evaluation ********
2021-01-07 23:45:45,827 P20355 INFO Loading data...
2021-01-07 23:45:45,827 P20355 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/test.h5
2021-01-07 23:45:46,488 P20355 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2021-01-07 23:45:46,489 P20355 INFO Loading test data done.
2021-01-07 23:46:06,825 P20355 INFO [Metrics] AUC: 0.813766 - logloss: 0.438075

```