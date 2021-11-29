## FM_Criteo_x0_001

A notebook to benchmark FM on Criteo_x0_001 dataset.

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
[Metrics] AUC: 0.802157 - logloss: 0.449063
```


### Logs
```python
2021-01-11 00:38:27,641 P40230 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x0_ace9c1b9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_criteo_x0_001_17295367",
    "model_root": "./Criteo/FM_criteo_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
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
2021-01-11 00:38:27,642 P40230 INFO Set up feature encoder...
2021-01-11 00:38:27,642 P40230 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x0_ace9c1b9/feature_encoder.pkl
2021-01-11 00:38:29,145 P40230 INFO Total number of parameters: 22949477.
2021-01-11 00:38:29,145 P40230 INFO Loading data...
2021-01-11 00:38:29,148 P40230 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/train.h5
2021-01-11 00:38:34,181 P40230 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/valid.h5
2021-01-11 00:38:35,449 P40230 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2021-01-11 00:38:35,450 P40230 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2021-01-11 00:38:35,450 P40230 INFO Loading train data done.
2021-01-11 00:38:39,011 P40230 INFO Start training: 8058 batches/epoch
2021-01-11 00:38:39,011 P40230 INFO ************ Epoch=1 start ************
2021-01-11 01:11:40,883 P40230 INFO [Metrics] AUC: 0.793791 - logloss: 0.456591
2021-01-11 01:11:40,885 P40230 INFO Save best model: monitor(max): 0.793791
2021-01-11 01:11:40,970 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 01:11:41,034 P40230 INFO Train loss: 0.471009
2021-01-11 01:11:41,035 P40230 INFO ************ Epoch=1 end ************
2021-01-11 01:44:48,985 P40230 INFO [Metrics] AUC: 0.795107 - logloss: 0.455468
2021-01-11 01:44:48,987 P40230 INFO Save best model: monitor(max): 0.795107
2021-01-11 01:44:49,146 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 01:44:49,210 P40230 INFO Train loss: 0.465477
2021-01-11 01:44:49,210 P40230 INFO ************ Epoch=2 end ************
2021-01-11 02:17:52,328 P40230 INFO [Metrics] AUC: 0.795598 - logloss: 0.454960
2021-01-11 02:17:52,329 P40230 INFO Save best model: monitor(max): 0.795598
2021-01-11 02:17:52,486 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 02:17:52,565 P40230 INFO Train loss: 0.464934
2021-01-11 02:17:52,565 P40230 INFO ************ Epoch=3 end ************
2021-01-11 02:50:55,150 P40230 INFO [Metrics] AUC: 0.795763 - logloss: 0.454858
2021-01-11 02:50:55,152 P40230 INFO Save best model: monitor(max): 0.795763
2021-01-11 02:50:55,302 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 02:50:55,376 P40230 INFO Train loss: 0.464727
2021-01-11 02:50:55,376 P40230 INFO ************ Epoch=4 end ************
2021-01-11 03:24:02,719 P40230 INFO [Metrics] AUC: 0.795982 - logloss: 0.454646
2021-01-11 03:24:02,720 P40230 INFO Save best model: monitor(max): 0.795982
2021-01-11 03:24:02,869 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 03:24:02,947 P40230 INFO Train loss: 0.464639
2021-01-11 03:24:02,947 P40230 INFO ************ Epoch=5 end ************
2021-01-11 03:57:14,146 P40230 INFO [Metrics] AUC: 0.795896 - logloss: 0.454714
2021-01-11 03:57:14,148 P40230 INFO Monitor(max) STOP: 0.795896 !
2021-01-11 03:57:14,148 P40230 INFO Reduce learning rate on plateau: 0.000100
2021-01-11 03:57:14,148 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 03:57:14,212 P40230 INFO Train loss: 0.464562
2021-01-11 03:57:14,212 P40230 INFO ************ Epoch=6 end ************
2021-01-11 04:30:24,995 P40230 INFO [Metrics] AUC: 0.799876 - logloss: 0.451159
2021-01-11 04:30:24,996 P40230 INFO Save best model: monitor(max): 0.799876
2021-01-11 04:30:25,133 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 04:30:25,225 P40230 INFO Train loss: 0.454956
2021-01-11 04:30:25,226 P40230 INFO ************ Epoch=7 end ************
2021-01-11 05:03:37,976 P40230 INFO [Metrics] AUC: 0.800415 - logloss: 0.450699
2021-01-11 05:03:37,977 P40230 INFO Save best model: monitor(max): 0.800415
2021-01-11 05:03:38,133 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 05:03:38,198 P40230 INFO Train loss: 0.452030
2021-01-11 05:03:38,198 P40230 INFO ************ Epoch=8 end ************
2021-01-11 05:36:49,926 P40230 INFO [Metrics] AUC: 0.800658 - logloss: 0.450475
2021-01-11 05:36:49,927 P40230 INFO Save best model: monitor(max): 0.800658
2021-01-11 05:36:50,061 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 05:36:50,130 P40230 INFO Train loss: 0.451344
2021-01-11 05:36:50,130 P40230 INFO ************ Epoch=9 end ************
2021-01-11 06:02:37,570 P40230 INFO [Metrics] AUC: 0.800839 - logloss: 0.450337
2021-01-11 06:02:37,571 P40230 INFO Save best model: monitor(max): 0.800839
2021-01-11 06:02:37,727 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 06:02:37,797 P40230 INFO Train loss: 0.450972
2021-01-11 06:02:37,798 P40230 INFO ************ Epoch=10 end ************
2021-01-11 06:15:14,363 P40230 INFO [Metrics] AUC: 0.800953 - logloss: 0.450240
2021-01-11 06:15:14,365 P40230 INFO Save best model: monitor(max): 0.800953
2021-01-11 06:15:14,521 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 06:15:14,589 P40230 INFO Train loss: 0.450701
2021-01-11 06:15:14,589 P40230 INFO ************ Epoch=11 end ************
2021-01-11 06:27:50,597 P40230 INFO [Metrics] AUC: 0.801014 - logloss: 0.450158
2021-01-11 06:27:50,598 P40230 INFO Save best model: monitor(max): 0.801014
2021-01-11 06:27:50,737 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 06:27:50,804 P40230 INFO Train loss: 0.450469
2021-01-11 06:27:50,804 P40230 INFO ************ Epoch=12 end ************
2021-01-11 06:40:30,291 P40230 INFO [Metrics] AUC: 0.801136 - logloss: 0.450061
2021-01-11 06:40:30,293 P40230 INFO Save best model: monitor(max): 0.801136
2021-01-11 06:40:30,443 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 06:40:30,521 P40230 INFO Train loss: 0.450266
2021-01-11 06:40:30,521 P40230 INFO ************ Epoch=13 end ************
2021-01-11 06:53:12,473 P40230 INFO [Metrics] AUC: 0.801240 - logloss: 0.449983
2021-01-11 06:53:12,475 P40230 INFO Save best model: monitor(max): 0.801240
2021-01-11 06:53:12,610 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 06:53:12,681 P40230 INFO Train loss: 0.450081
2021-01-11 06:53:12,681 P40230 INFO ************ Epoch=14 end ************
2021-01-11 07:05:43,338 P40230 INFO [Metrics] AUC: 0.801272 - logloss: 0.449947
2021-01-11 07:05:43,340 P40230 INFO Save best model: monitor(max): 0.801272
2021-01-11 07:05:43,476 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 07:05:43,544 P40230 INFO Train loss: 0.449894
2021-01-11 07:05:43,544 P40230 INFO ************ Epoch=15 end ************
2021-01-11 07:18:22,539 P40230 INFO [Metrics] AUC: 0.801339 - logloss: 0.449896
2021-01-11 07:18:22,541 P40230 INFO Save best model: monitor(max): 0.801339
2021-01-11 07:18:22,690 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 07:18:22,764 P40230 INFO Train loss: 0.449727
2021-01-11 07:18:22,764 P40230 INFO ************ Epoch=16 end ************
2021-01-11 07:30:52,346 P40230 INFO [Metrics] AUC: 0.801422 - logloss: 0.449828
2021-01-11 07:30:52,347 P40230 INFO Save best model: monitor(max): 0.801422
2021-01-11 07:30:52,479 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 07:30:52,567 P40230 INFO Train loss: 0.449564
2021-01-11 07:30:52,567 P40230 INFO ************ Epoch=17 end ************
2021-01-11 07:43:25,915 P40230 INFO [Metrics] AUC: 0.801474 - logloss: 0.449796
2021-01-11 07:43:25,917 P40230 INFO Save best model: monitor(max): 0.801474
2021-01-11 07:43:26,045 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 07:43:26,122 P40230 INFO Train loss: 0.449415
2021-01-11 07:43:26,122 P40230 INFO ************ Epoch=18 end ************
2021-01-11 07:56:03,015 P40230 INFO [Metrics] AUC: 0.801491 - logloss: 0.449771
2021-01-11 07:56:03,016 P40230 INFO Save best model: monitor(max): 0.801491
2021-01-11 07:56:03,157 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 07:56:03,234 P40230 INFO Train loss: 0.449265
2021-01-11 07:56:03,234 P40230 INFO ************ Epoch=19 end ************
2021-01-11 08:08:51,347 P40230 INFO [Metrics] AUC: 0.801552 - logloss: 0.449739
2021-01-11 08:08:51,348 P40230 INFO Save best model: monitor(max): 0.801552
2021-01-11 08:08:51,522 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 08:08:51,599 P40230 INFO Train loss: 0.449127
2021-01-11 08:08:51,600 P40230 INFO ************ Epoch=20 end ************
2021-01-11 08:21:23,441 P40230 INFO [Metrics] AUC: 0.801597 - logloss: 0.449683
2021-01-11 08:21:23,443 P40230 INFO Save best model: monitor(max): 0.801597
2021-01-11 08:21:23,583 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 08:21:23,653 P40230 INFO Train loss: 0.448997
2021-01-11 08:21:23,653 P40230 INFO ************ Epoch=21 end ************
2021-01-11 08:33:59,725 P40230 INFO [Metrics] AUC: 0.801595 - logloss: 0.449691
2021-01-11 08:33:59,727 P40230 INFO Monitor(max) STOP: 0.801595 !
2021-01-11 08:33:59,727 P40230 INFO Reduce learning rate on plateau: 0.000010
2021-01-11 08:33:59,727 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 08:33:59,803 P40230 INFO Train loss: 0.448870
2021-01-11 08:33:59,804 P40230 INFO ************ Epoch=22 end ************
2021-01-11 08:46:28,322 P40230 INFO [Metrics] AUC: 0.801839 - logloss: 0.449461
2021-01-11 08:46:28,324 P40230 INFO Save best model: monitor(max): 0.801839
2021-01-11 08:46:28,459 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 08:46:28,532 P40230 INFO Train loss: 0.446134
2021-01-11 08:46:28,532 P40230 INFO ************ Epoch=23 end ************
2021-01-11 08:58:55,673 P40230 INFO [Metrics] AUC: 0.801877 - logloss: 0.449429
2021-01-11 08:58:55,674 P40230 INFO Save best model: monitor(max): 0.801877
2021-01-11 08:58:55,813 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 08:58:55,888 P40230 INFO Train loss: 0.446013
2021-01-11 08:58:55,888 P40230 INFO ************ Epoch=24 end ************
2021-01-11 09:11:25,397 P40230 INFO [Metrics] AUC: 0.801892 - logloss: 0.449415
2021-01-11 09:11:25,399 P40230 INFO Save best model: monitor(max): 0.801892
2021-01-11 09:11:25,537 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 09:11:25,610 P40230 INFO Train loss: 0.445960
2021-01-11 09:11:25,611 P40230 INFO ************ Epoch=25 end ************
2021-01-11 09:23:54,771 P40230 INFO [Metrics] AUC: 0.801906 - logloss: 0.449403
2021-01-11 09:23:54,772 P40230 INFO Save best model: monitor(max): 0.801906
2021-01-11 09:23:54,921 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 09:23:55,007 P40230 INFO Train loss: 0.445922
2021-01-11 09:23:55,007 P40230 INFO ************ Epoch=26 end ************
2021-01-11 09:36:23,957 P40230 INFO [Metrics] AUC: 0.801910 - logloss: 0.449401
2021-01-11 09:36:23,959 P40230 INFO Save best model: monitor(max): 0.801910
2021-01-11 09:36:24,114 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 09:36:24,193 P40230 INFO Train loss: 0.445891
2021-01-11 09:36:24,194 P40230 INFO ************ Epoch=27 end ************
2021-01-11 09:48:52,300 P40230 INFO [Metrics] AUC: 0.801912 - logloss: 0.449398
2021-01-11 09:48:52,301 P40230 INFO Save best model: monitor(max): 0.801912
2021-01-11 09:48:52,442 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 09:48:52,518 P40230 INFO Train loss: 0.445861
2021-01-11 09:48:52,518 P40230 INFO ************ Epoch=28 end ************
2021-01-11 10:01:21,710 P40230 INFO [Metrics] AUC: 0.801917 - logloss: 0.449395
2021-01-11 10:01:21,711 P40230 INFO Save best model: monitor(max): 0.801917
2021-01-11 10:01:21,839 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 10:01:21,915 P40230 INFO Train loss: 0.445835
2021-01-11 10:01:21,915 P40230 INFO ************ Epoch=29 end ************
2021-01-11 10:13:50,546 P40230 INFO [Metrics] AUC: 0.801924 - logloss: 0.449392
2021-01-11 10:13:50,548 P40230 INFO Save best model: monitor(max): 0.801924
2021-01-11 10:13:50,728 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 10:13:50,803 P40230 INFO Train loss: 0.445810
2021-01-11 10:13:50,803 P40230 INFO ************ Epoch=30 end ************
2021-01-11 10:26:34,411 P40230 INFO [Metrics] AUC: 0.801928 - logloss: 0.449388
2021-01-11 10:26:34,412 P40230 INFO Save best model: monitor(max): 0.801928
2021-01-11 10:26:34,548 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 10:26:34,623 P40230 INFO Train loss: 0.445783
2021-01-11 10:26:34,623 P40230 INFO ************ Epoch=31 end ************
2021-01-11 10:39:02,513 P40230 INFO [Metrics] AUC: 0.801932 - logloss: 0.449383
2021-01-11 10:39:02,514 P40230 INFO Save best model: monitor(max): 0.801932
2021-01-11 10:39:02,649 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 10:39:02,728 P40230 INFO Train loss: 0.445757
2021-01-11 10:39:02,728 P40230 INFO ************ Epoch=32 end ************
2021-01-11 10:51:32,326 P40230 INFO [Metrics] AUC: 0.801932 - logloss: 0.449384
2021-01-11 10:51:32,327 P40230 INFO Monitor(max) STOP: 0.801932 !
2021-01-11 10:51:32,327 P40230 INFO Reduce learning rate on plateau: 0.000001
2021-01-11 10:51:32,327 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 10:51:32,399 P40230 INFO Train loss: 0.445729
2021-01-11 10:51:32,399 P40230 INFO ************ Epoch=33 end ************
2021-01-11 11:04:01,830 P40230 INFO [Metrics] AUC: 0.801936 - logloss: 0.449381
2021-01-11 11:04:01,832 P40230 INFO Save best model: monitor(max): 0.801936
2021-01-11 11:04:01,964 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 11:04:02,041 P40230 INFO Train loss: 0.445332
2021-01-11 11:04:02,041 P40230 INFO ************ Epoch=34 end ************
2021-01-11 11:27:08,598 P40230 INFO [Metrics] AUC: 0.801937 - logloss: 0.449380
2021-01-11 11:27:08,600 P40230 INFO Save best model: monitor(max): 0.801937
2021-01-11 11:27:08,753 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 11:27:08,833 P40230 INFO Train loss: 0.445329
2021-01-11 11:27:08,833 P40230 INFO ************ Epoch=35 end ************
2021-01-11 11:59:07,161 P40230 INFO [Metrics] AUC: 0.801938 - logloss: 0.449380
2021-01-11 11:59:07,163 P40230 INFO Monitor(max) STOP: 0.801938 !
2021-01-11 11:59:07,163 P40230 INFO Reduce learning rate on plateau: 0.000001
2021-01-11 11:59:07,163 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 11:59:07,437 P40230 INFO Train loss: 0.445327
2021-01-11 11:59:07,437 P40230 INFO ************ Epoch=36 end ************
2021-01-11 12:29:04,108 P40230 INFO [Metrics] AUC: 0.801938 - logloss: 0.449380
2021-01-11 12:29:04,110 P40230 INFO Monitor(max) STOP: 0.801938 !
2021-01-11 12:29:04,110 P40230 INFO Reduce learning rate on plateau: 0.000001
2021-01-11 12:29:04,110 P40230 INFO Early stopping at epoch=37
2021-01-11 12:29:04,110 P40230 INFO --- 8058/8058 batches finished ---
2021-01-11 12:29:04,499 P40230 INFO Train loss: 0.445326
2021-01-11 12:29:04,500 P40230 INFO Training finished.
2021-01-11 12:29:04,500 P40230 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Criteo/FM_criteo_x0/criteo_x0_ace9c1b9/FM_criteo_x0_001_17295367_model.ckpt
2021-01-11 12:29:04,805 P40230 INFO ****** Train/validation evaluation ******
2021-01-11 12:31:17,112 P40230 INFO [Metrics] AUC: 0.801937 - logloss: 0.449380
2021-01-11 12:31:17,158 P40230 INFO ******** Test evaluation ********
2021-01-11 12:31:17,158 P40230 INFO Loading data...
2021-01-11 12:31:17,159 P40230 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/test.h5
2021-01-11 12:31:17,844 P40230 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2021-01-11 12:31:17,844 P40230 INFO Loading test data done.
2021-01-11 12:32:27,253 P40230 INFO [Metrics] AUC: 0.802157 - logloss: 0.449063

```
