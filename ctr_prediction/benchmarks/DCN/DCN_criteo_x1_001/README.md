## DCN_Criteo_x0_001

A notebook to benchmark DCN on Criteo_x0_001 dataset.

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
[Metrics] logloss: 0.437904 - AUC: 0.813926
```

### Logs
```python
2020-12-27 13:13:41,486 P31211 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "3",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x0_ace9c1b9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_criteo_x0_014_53f4baa3",
    "model_root": "./Criteo/DCN_criteo_x0/",
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
2020-12-27 13:13:41,487 P31211 INFO Set up feature encoder...
2020-12-27 13:13:41,487 P31211 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x0_ace9c1b9/feature_encoder.pkl
2020-12-27 13:13:43,250 P31211 INFO Total number of parameters: 21343491.
2020-12-27 13:13:43,250 P31211 INFO Loading data...
2020-12-27 13:13:43,253 P31211 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/train.h5
2020-12-27 13:13:48,716 P31211 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/valid.h5
2020-12-27 13:13:50,454 P31211 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2020-12-27 13:13:50,454 P31211 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2020-12-27 13:13:50,454 P31211 INFO Loading train data done.
2020-12-27 13:13:55,090 P31211 INFO Start training: 8058 batches/epoch
2020-12-27 13:13:55,091 P31211 INFO ************ Epoch=1 start ************
2020-12-27 14:12:58,388 P31211 INFO [Metrics] logloss: 0.447059 - AUC: 0.804085
2020-12-27 14:12:58,389 P31211 INFO Save best model: monitor(max): 0.357026
2020-12-27 14:12:58,466 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 14:12:58,547 P31211 INFO Train loss: 0.461657
2020-12-27 14:12:58,547 P31211 INFO ************ Epoch=1 end ************
2020-12-27 15:02:52,736 P31211 INFO [Metrics] logloss: 0.445168 - AUC: 0.806487
2020-12-27 15:02:52,737 P31211 INFO Save best model: monitor(max): 0.361319
2020-12-27 15:02:52,873 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 15:02:52,958 P31211 INFO Train loss: 0.455967
2020-12-27 15:02:52,959 P31211 INFO ************ Epoch=2 end ************
2020-12-27 15:47:57,714 P31211 INFO [Metrics] logloss: 0.444223 - AUC: 0.807586
2020-12-27 15:47:57,716 P31211 INFO Save best model: monitor(max): 0.363363
2020-12-27 15:47:57,851 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 15:47:57,934 P31211 INFO Train loss: 0.454617
2020-12-27 15:47:57,934 P31211 INFO ************ Epoch=3 end ************
2020-12-27 16:36:06,487 P31211 INFO [Metrics] logloss: 0.443307 - AUC: 0.808256
2020-12-27 16:36:06,488 P31211 INFO Save best model: monitor(max): 0.364950
2020-12-27 16:36:06,630 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 16:36:06,714 P31211 INFO Train loss: 0.453948
2020-12-27 16:36:06,714 P31211 INFO ************ Epoch=4 end ************
2020-12-27 17:14:12,114 P31211 INFO [Metrics] logloss: 0.443120 - AUC: 0.808461
2020-12-27 17:14:12,116 P31211 INFO Save best model: monitor(max): 0.365341
2020-12-27 17:14:12,261 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 17:14:12,345 P31211 INFO Train loss: 0.453554
2020-12-27 17:14:12,346 P31211 INFO ************ Epoch=5 end ************
2020-12-27 17:48:52,255 P31211 INFO [Metrics] logloss: 0.442741 - AUC: 0.808832
2020-12-27 17:48:52,257 P31211 INFO Save best model: monitor(max): 0.366091
2020-12-27 17:48:52,397 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 17:48:52,479 P31211 INFO Train loss: 0.453250
2020-12-27 17:48:52,479 P31211 INFO ************ Epoch=6 end ************
2020-12-27 18:23:30,297 P31211 INFO [Metrics] logloss: 0.442675 - AUC: 0.808974
2020-12-27 18:23:30,298 P31211 INFO Save best model: monitor(max): 0.366299
2020-12-27 18:23:30,418 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 18:23:30,506 P31211 INFO Train loss: 0.453000
2020-12-27 18:23:30,506 P31211 INFO ************ Epoch=7 end ************
2020-12-27 19:02:37,888 P31211 INFO [Metrics] logloss: 0.442776 - AUC: 0.809184
2020-12-27 19:02:37,902 P31211 INFO Save best model: monitor(max): 0.366408
2020-12-27 19:02:38,069 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 19:02:38,161 P31211 INFO Train loss: 0.452781
2020-12-27 19:02:38,162 P31211 INFO ************ Epoch=8 end ************
2020-12-27 20:02:09,398 P31211 INFO [Metrics] logloss: 0.442581 - AUC: 0.809264
2020-12-27 20:02:09,400 P31211 INFO Save best model: monitor(max): 0.366683
2020-12-27 20:02:09,500 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 20:02:09,594 P31211 INFO Train loss: 0.452622
2020-12-27 20:02:09,594 P31211 INFO ************ Epoch=9 end ************
2020-12-27 21:01:33,977 P31211 INFO [Metrics] logloss: 0.442339 - AUC: 0.809386
2020-12-27 21:01:33,979 P31211 INFO Save best model: monitor(max): 0.367046
2020-12-27 21:01:34,124 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 21:01:34,213 P31211 INFO Train loss: 0.452513
2020-12-27 21:01:34,214 P31211 INFO ************ Epoch=10 end ************
2020-12-27 22:00:37,410 P31211 INFO [Metrics] logloss: 0.442111 - AUC: 0.809505
2020-12-27 22:00:37,425 P31211 INFO Save best model: monitor(max): 0.367394
2020-12-27 22:00:37,572 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 22:00:37,656 P31211 INFO Train loss: 0.452360
2020-12-27 22:00:37,657 P31211 INFO ************ Epoch=11 end ************
2020-12-27 22:55:30,599 P31211 INFO [Metrics] logloss: 0.442067 - AUC: 0.809600
2020-12-27 22:55:30,601 P31211 INFO Save best model: monitor(max): 0.367533
2020-12-27 22:55:30,737 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 22:55:30,819 P31211 INFO Train loss: 0.452260
2020-12-27 22:55:30,819 P31211 INFO ************ Epoch=12 end ************
2020-12-27 23:47:59,466 P31211 INFO [Metrics] logloss: 0.442065 - AUC: 0.809637
2020-12-27 23:47:59,468 P31211 INFO Save best model: monitor(max): 0.367572
2020-12-27 23:47:59,598 P31211 INFO --- 8058/8058 batches finished ---
2020-12-27 23:47:59,681 P31211 INFO Train loss: 0.452170
2020-12-27 23:47:59,681 P31211 INFO ************ Epoch=13 end ************
2020-12-28 00:40:52,590 P31211 INFO [Metrics] logloss: 0.442010 - AUC: 0.809582
2020-12-28 00:40:52,591 P31211 INFO Monitor(max) STOP: 0.367572 !
2020-12-28 00:40:52,591 P31211 INFO Reduce learning rate on plateau: 0.000100
2020-12-28 00:40:52,591 P31211 INFO --- 8058/8058 batches finished ---
2020-12-28 00:40:52,681 P31211 INFO Train loss: 0.452076
2020-12-28 00:40:52,681 P31211 INFO ************ Epoch=14 end ************
2020-12-28 01:29:35,926 P31211 INFO [Metrics] logloss: 0.439002 - AUC: 0.812899
2020-12-28 01:29:35,930 P31211 INFO Save best model: monitor(max): 0.373897
2020-12-28 01:29:36,164 P31211 INFO --- 8058/8058 batches finished ---
2020-12-28 01:29:36,385 P31211 INFO Train loss: 0.441425
2020-12-28 01:29:36,386 P31211 INFO ************ Epoch=15 end ************
2020-12-28 02:04:27,234 P31211 INFO [Metrics] logloss: 0.438487 - AUC: 0.813407
2020-12-28 02:04:27,236 P31211 INFO Save best model: monitor(max): 0.374919
2020-12-28 02:04:27,388 P31211 INFO --- 8058/8058 batches finished ---
2020-12-28 02:04:27,474 P31211 INFO Train loss: 0.437405
2020-12-28 02:04:27,474 P31211 INFO ************ Epoch=16 end ************
2020-12-28 02:32:12,570 P31211 INFO [Metrics] logloss: 0.438385 - AUC: 0.813570
2020-12-28 02:32:12,572 P31211 INFO Save best model: monitor(max): 0.375185
2020-12-28 02:32:12,706 P31211 INFO --- 8058/8058 batches finished ---
2020-12-28 02:32:12,795 P31211 INFO Train loss: 0.435645
2020-12-28 02:32:12,795 P31211 INFO ************ Epoch=17 end ************
2020-12-28 02:59:55,367 P31211 INFO [Metrics] logloss: 0.438483 - AUC: 0.813512
2020-12-28 02:59:55,369 P31211 INFO Monitor(max) STOP: 0.375029 !
2020-12-28 02:59:55,369 P31211 INFO Reduce learning rate on plateau: 0.000010
2020-12-28 02:59:55,369 P31211 INFO --- 8058/8058 batches finished ---
2020-12-28 02:59:55,458 P31211 INFO Train loss: 0.434309
2020-12-28 02:59:55,458 P31211 INFO ************ Epoch=18 end ************
2020-12-28 03:20:59,776 P31211 INFO [Metrics] logloss: 0.438989 - AUC: 0.813241
2020-12-28 03:20:59,778 P31211 INFO Monitor(max) STOP: 0.374252 !
2020-12-28 03:20:59,778 P31211 INFO Reduce learning rate on plateau: 0.000001
2020-12-28 03:20:59,778 P31211 INFO Early stopping at epoch=19
2020-12-28 03:20:59,778 P31211 INFO --- 8058/8058 batches finished ---
2020-12-28 03:20:59,862 P31211 INFO Train loss: 0.430456
2020-12-28 03:20:59,862 P31211 INFO Training finished.
2020-12-28 03:20:59,862 P31211 INFO Load best model: /home/zhujieming/zhujieming/FuxiCTR/benchmarks/Criteo/DCN_criteo_x0/criteo_x0_ace9c1b9/DCN_criteo_x0_014_53f4baa3_model.ckpt
2020-12-28 03:21:00,060 P31211 INFO ****** Train/validation evaluation ******
2020-12-28 03:21:35,528 P31211 INFO [Metrics] logloss: 0.438385 - AUC: 0.813570
2020-12-28 03:21:35,636 P31211 INFO ******** Test evaluation ********
2020-12-28 03:21:35,636 P31211 INFO Loading data...
2020-12-28 03:21:35,636 P31211 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/test.h5
2020-12-28 03:21:36,369 P31211 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2020-12-28 03:21:36,369 P31211 INFO Loading test data done.
2020-12-28 03:21:55,028 P31211 INFO [Metrics] logloss: 0.437904 - AUC: 0.813926



```
