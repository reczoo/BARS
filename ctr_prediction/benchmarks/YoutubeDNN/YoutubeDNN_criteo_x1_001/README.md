## YoutubeDNN_Criteo_x0_001 

A notebook to benchmark YoutubeDNN on Criteo_x0_001 dataset.

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
[Metrics] logloss: 0.438193 - AUC: 0.813561
```


### Logs
```python
2020-12-29 21:25:24,841 P34601 INFO {
    "batch_norm": "False",
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
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_criteo_x0_012_ad22ff24",
    "model_root": "./Criteo/DNN_criteo_x0/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
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
2020-12-29 21:25:24,843 P34601 INFO Set up feature encoder...
2020-12-29 21:25:24,843 P34601 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x0_ace9c1b9/feature_encoder.pkl
2020-12-29 21:25:30,559 P34601 INFO Total number of parameters: 21340761.
2020-12-29 21:25:30,560 P34601 INFO Loading data...
2020-12-29 21:25:30,567 P34601 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/train.h5
2020-12-29 21:25:40,876 P34601 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/valid.h5
2020-12-29 21:25:43,082 P34601 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2020-12-29 21:25:43,083 P34601 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2020-12-29 21:25:43,083 P34601 INFO Loading train data done.
2020-12-29 21:25:51,371 P34601 INFO Start training: 8058 batches/epoch
2020-12-29 21:25:51,371 P34601 INFO ************ Epoch=1 start ************
2020-12-29 21:57:40,339 P34601 INFO [Metrics] logloss: 0.447765 - AUC: 0.803683
2020-12-29 21:57:40,351 P34601 INFO Save best model: monitor(max): 0.355919
2020-12-29 21:57:40,671 P34601 INFO --- 8058/8058 batches finished ---
2020-12-29 21:57:41,252 P34601 INFO Train loss: 0.462266
2020-12-29 21:57:41,252 P34601 INFO ************ Epoch=1 end ************
2020-12-29 22:28:53,349 P34601 INFO [Metrics] logloss: 0.445400 - AUC: 0.806353
2020-12-29 22:28:53,360 P34601 INFO Save best model: monitor(max): 0.360953
2020-12-29 22:28:53,675 P34601 INFO --- 8058/8058 batches finished ---
2020-12-29 22:28:54,001 P34601 INFO Train loss: 0.456844
2020-12-29 22:28:54,002 P34601 INFO ************ Epoch=2 end ************
2020-12-29 23:01:30,883 P34601 INFO [Metrics] logloss: 0.444427 - AUC: 0.807261
2020-12-29 23:01:30,887 P34601 INFO Save best model: monitor(max): 0.362834
2020-12-29 23:01:31,242 P34601 INFO --- 8058/8058 batches finished ---
2020-12-29 23:01:31,567 P34601 INFO Train loss: 0.455390
2020-12-29 23:01:31,567 P34601 INFO ************ Epoch=3 end ************
2020-12-29 23:31:15,694 P34601 INFO [Metrics] logloss: 0.443646 - AUC: 0.807871
2020-12-29 23:31:15,698 P34601 INFO Save best model: monitor(max): 0.364225
2020-12-29 23:31:16,006 P34601 INFO --- 8058/8058 batches finished ---
2020-12-29 23:31:16,439 P34601 INFO Train loss: 0.454719
2020-12-29 23:31:16,440 P34601 INFO ************ Epoch=4 end ************
2020-12-30 00:02:21,458 P34601 INFO [Metrics] logloss: 0.443315 - AUC: 0.808390
2020-12-30 00:02:21,476 P34601 INFO Save best model: monitor(max): 0.365075
2020-12-30 00:02:21,727 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 00:02:22,043 P34601 INFO Train loss: 0.454256
2020-12-30 00:02:22,043 P34601 INFO ************ Epoch=5 end ************
2020-12-30 00:32:41,305 P34601 INFO [Metrics] logloss: 0.442961 - AUC: 0.808654
2020-12-30 00:32:41,318 P34601 INFO Save best model: monitor(max): 0.365694
2020-12-30 00:32:41,825 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 00:32:42,242 P34601 INFO Train loss: 0.453929
2020-12-30 00:32:42,243 P34601 INFO ************ Epoch=6 end ************
2020-12-30 01:05:51,574 P34601 INFO [Metrics] logloss: 0.443018 - AUC: 0.808930
2020-12-30 01:05:51,576 P34601 INFO Save best model: monitor(max): 0.365912
2020-12-30 01:05:51,919 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 01:05:52,233 P34601 INFO Train loss: 0.453680
2020-12-30 01:05:52,233 P34601 INFO ************ Epoch=7 end ************
2020-12-30 01:37:48,540 P34601 INFO [Metrics] logloss: 0.442656 - AUC: 0.808964
2020-12-30 01:37:48,548 P34601 INFO Save best model: monitor(max): 0.366308
2020-12-30 01:37:48,872 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 01:37:49,455 P34601 INFO Train loss: 0.453480
2020-12-30 01:37:49,455 P34601 INFO ************ Epoch=8 end ************
2020-12-30 02:09:44,425 P34601 INFO [Metrics] logloss: 0.442797 - AUC: 0.809188
2020-12-30 02:09:44,432 P34601 INFO Save best model: monitor(max): 0.366391
2020-12-30 02:09:44,789 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 02:09:45,113 P34601 INFO Train loss: 0.453301
2020-12-30 02:09:45,113 P34601 INFO ************ Epoch=9 end ************
2020-12-30 02:42:33,989 P34601 INFO [Metrics] logloss: 0.442814 - AUC: 0.809119
2020-12-30 02:42:33,992 P34601 INFO Monitor(max) STOP: 0.366305 !
2020-12-30 02:42:33,992 P34601 INFO Reduce learning rate on plateau: 0.000100
2020-12-30 02:42:33,992 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 02:42:34,411 P34601 INFO Train loss: 0.453192
2020-12-30 02:42:34,411 P34601 INFO ************ Epoch=10 end ************
2020-12-30 03:13:07,305 P34601 INFO [Metrics] logloss: 0.439281 - AUC: 0.812563
2020-12-30 03:13:07,316 P34601 INFO Save best model: monitor(max): 0.373282
2020-12-30 03:13:07,680 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 03:13:08,043 P34601 INFO Train loss: 0.442472
2020-12-30 03:13:08,043 P34601 INFO ************ Epoch=11 end ************
2020-12-30 03:45:45,314 P34601 INFO [Metrics] logloss: 0.438801 - AUC: 0.813102
2020-12-30 03:45:45,316 P34601 INFO Save best model: monitor(max): 0.374301
2020-12-30 03:45:45,669 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 03:45:46,034 P34601 INFO Train loss: 0.438305
2020-12-30 03:45:46,034 P34601 INFO ************ Epoch=12 end ************
2020-12-30 04:16:29,891 P34601 INFO [Metrics] logloss: 0.438638 - AUC: 0.813258
2020-12-30 04:16:29,893 P34601 INFO Save best model: monitor(max): 0.374620
2020-12-30 04:16:30,234 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 04:16:30,544 P34601 INFO Train loss: 0.436469
2020-12-30 04:16:30,544 P34601 INFO ************ Epoch=13 end ************
2020-12-30 04:43:17,333 P34601 INFO [Metrics] logloss: 0.438663 - AUC: 0.813245
2020-12-30 04:43:17,336 P34601 INFO Monitor(max) STOP: 0.374583 !
2020-12-30 04:43:17,336 P34601 INFO Reduce learning rate on plateau: 0.000010
2020-12-30 04:43:17,336 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 04:43:17,636 P34601 INFO Train loss: 0.435106
2020-12-30 04:43:17,637 P34601 INFO ************ Epoch=14 end ************
2020-12-30 05:04:12,968 P34601 INFO [Metrics] logloss: 0.439143 - AUC: 0.813013
2020-12-30 05:04:12,970 P34601 INFO Monitor(max) STOP: 0.373870 !
2020-12-30 05:04:12,971 P34601 INFO Reduce learning rate on plateau: 0.000001
2020-12-30 05:04:12,971 P34601 INFO Early stopping at epoch=15
2020-12-30 05:04:12,971 P34601 INFO --- 8058/8058 batches finished ---
2020-12-30 05:04:13,303 P34601 INFO Train loss: 0.431000
2020-12-30 05:04:13,303 P34601 INFO Training finished.
2020-12-30 05:04:13,303 P34601 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Criteo/DNN_criteo_x0/criteo_x0_ace9c1b9/DNN_criteo_x0_012_ad22ff24_model.ckpt
2020-12-30 05:04:13,704 P34601 INFO ****** Train/validation evaluation ******
2020-12-30 05:05:54,928 P34601 INFO [Metrics] logloss: 0.438638 - AUC: 0.813258
2020-12-30 05:05:55,223 P34601 INFO ******** Test evaluation ********
2020-12-30 05:05:55,224 P34601 INFO Loading data...
2020-12-30 05:05:55,227 P34601 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/test.h5
2020-12-30 05:05:56,694 P34601 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2020-12-30 05:05:56,694 P34601 INFO Loading test data done.
2020-12-30 05:06:54,839 P34601 INFO [Metrics] logloss: 0.438193 - AUC: 0.813561

```
