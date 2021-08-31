## FiGNN_Criteo_x4_001

A notebook to benchmark FiGNN on Criteo_x4_001 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.
### Code




### Results
```python
[Metrics] logloss: 0.438285 - AUC: 0.813803
```


### Logs
```python
2020-06-28 18:03:33,994 P1608 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gnn_layers": "4",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiGNN",
    "model_id": "FiGNN_criteo_x4_5c863b0f_006_b4207cb3",
    "model_root": "./Criteo/FiGNN_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_gru": "True",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-28 18:03:33,996 P1608 INFO Set up feature encoder...
2020-06-28 18:03:33,996 P1608 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-28 18:03:33,997 P1608 INFO Loading data...
2020-06-28 18:03:34,001 P1608 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-28 18:03:40,896 P1608 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-28 18:03:43,264 P1608 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-28 18:03:43,458 P1608 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-28 18:03:43,458 P1608 INFO Loading train data done.
2020-06-28 18:03:50,165 P1608 INFO Start training: 3668 batches/epoch
2020-06-28 18:03:50,166 P1608 INFO ************ Epoch=1 start ************
2020-06-28 18:35:53,816 P1608 INFO [Metrics] logloss: 0.448463 - AUC: 0.802507
2020-06-28 18:35:53,821 P1608 INFO Save best model: monitor(max): 0.354044
2020-06-28 18:35:53,958 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 18:35:54,029 P1608 INFO Train loss: 0.461990
2020-06-28 18:35:54,029 P1608 INFO ************ Epoch=1 end ************
2020-06-28 19:07:57,680 P1608 INFO [Metrics] logloss: 0.445560 - AUC: 0.805816
2020-06-28 19:07:57,692 P1608 INFO Save best model: monitor(max): 0.360256
2020-06-28 19:07:57,775 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 19:07:57,828 P1608 INFO Train loss: 0.453959
2020-06-28 19:07:57,829 P1608 INFO ************ Epoch=2 end ************
2020-06-28 19:40:01,956 P1608 INFO [Metrics] logloss: 0.444426 - AUC: 0.807096
2020-06-28 19:40:01,960 P1608 INFO Save best model: monitor(max): 0.362669
2020-06-28 19:40:02,043 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 19:40:02,100 P1608 INFO Train loss: 0.451988
2020-06-28 19:40:02,100 P1608 INFO ************ Epoch=3 end ************
2020-06-28 20:12:03,975 P1608 INFO [Metrics] logloss: 0.443639 - AUC: 0.807860
2020-06-28 20:12:03,976 P1608 INFO Save best model: monitor(max): 0.364221
2020-06-28 20:12:04,045 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 20:12:04,098 P1608 INFO Train loss: 0.450993
2020-06-28 20:12:04,098 P1608 INFO ************ Epoch=4 end ************
2020-06-28 20:44:06,081 P1608 INFO [Metrics] logloss: 0.443349 - AUC: 0.808236
2020-06-28 20:44:06,082 P1608 INFO Save best model: monitor(max): 0.364887
2020-06-28 20:44:06,148 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 20:44:06,209 P1608 INFO Train loss: 0.450321
2020-06-28 20:44:06,210 P1608 INFO ************ Epoch=5 end ************
2020-06-28 21:16:05,044 P1608 INFO [Metrics] logloss: 0.442854 - AUC: 0.808700
2020-06-28 21:16:05,045 P1608 INFO Save best model: monitor(max): 0.365846
2020-06-28 21:16:05,113 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 21:16:05,175 P1608 INFO Train loss: 0.449811
2020-06-28 21:16:05,175 P1608 INFO ************ Epoch=6 end ************
2020-06-28 21:48:07,154 P1608 INFO [Metrics] logloss: 0.442516 - AUC: 0.809043
2020-06-28 21:48:07,155 P1608 INFO Save best model: monitor(max): 0.366527
2020-06-28 21:48:07,236 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 21:48:07,290 P1608 INFO Train loss: 0.449446
2020-06-28 21:48:07,290 P1608 INFO ************ Epoch=7 end ************
2020-06-28 22:20:07,596 P1608 INFO [Metrics] logloss: 0.442228 - AUC: 0.809357
2020-06-28 22:20:07,598 P1608 INFO Save best model: monitor(max): 0.367129
2020-06-28 22:20:07,666 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 22:20:07,720 P1608 INFO Train loss: 0.449155
2020-06-28 22:20:07,720 P1608 INFO ************ Epoch=8 end ************
2020-06-28 22:52:05,485 P1608 INFO [Metrics] logloss: 0.441993 - AUC: 0.809666
2020-06-28 22:52:05,486 P1608 INFO Save best model: monitor(max): 0.367673
2020-06-28 22:52:05,554 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 22:52:05,607 P1608 INFO Train loss: 0.448893
2020-06-28 22:52:05,607 P1608 INFO ************ Epoch=9 end ************
2020-06-28 23:24:05,051 P1608 INFO [Metrics] logloss: 0.441881 - AUC: 0.809705
2020-06-28 23:24:05,053 P1608 INFO Save best model: monitor(max): 0.367824
2020-06-28 23:24:05,156 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 23:24:05,216 P1608 INFO Train loss: 0.448673
2020-06-28 23:24:05,216 P1608 INFO ************ Epoch=10 end ************
2020-06-28 23:56:10,899 P1608 INFO [Metrics] logloss: 0.441707 - AUC: 0.809921
2020-06-28 23:56:10,900 P1608 INFO Save best model: monitor(max): 0.368214
2020-06-28 23:56:10,981 P1608 INFO --- 3668/3668 batches finished ---
2020-06-28 23:56:11,044 P1608 INFO Train loss: 0.448468
2020-06-28 23:56:11,044 P1608 INFO ************ Epoch=11 end ************
2020-06-29 00:28:11,042 P1608 INFO [Metrics] logloss: 0.441551 - AUC: 0.810041
2020-06-29 00:28:11,043 P1608 INFO Save best model: monitor(max): 0.368490
2020-06-29 00:28:11,110 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 00:28:11,163 P1608 INFO Train loss: 0.448314
2020-06-29 00:28:11,163 P1608 INFO ************ Epoch=12 end ************
2020-06-29 01:00:17,003 P1608 INFO [Metrics] logloss: 0.441343 - AUC: 0.810416
2020-06-29 01:00:17,004 P1608 INFO Save best model: monitor(max): 0.369073
2020-06-29 01:00:17,073 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 01:00:17,131 P1608 INFO Train loss: 0.448140
2020-06-29 01:00:17,132 P1608 INFO ************ Epoch=13 end ************
2020-06-29 01:32:21,871 P1608 INFO [Metrics] logloss: 0.441241 - AUC: 0.810394
2020-06-29 01:32:21,872 P1608 INFO Save best model: monitor(max): 0.369153
2020-06-29 01:32:21,939 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 01:32:21,998 P1608 INFO Train loss: 0.448005
2020-06-29 01:32:21,998 P1608 INFO ************ Epoch=14 end ************
2020-06-29 02:04:28,979 P1608 INFO [Metrics] logloss: 0.441153 - AUC: 0.810508
2020-06-29 02:04:28,981 P1608 INFO Save best model: monitor(max): 0.369355
2020-06-29 02:04:29,092 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 02:04:29,154 P1608 INFO Train loss: 0.447864
2020-06-29 02:04:29,154 P1608 INFO ************ Epoch=15 end ************
2020-06-29 02:36:39,862 P1608 INFO [Metrics] logloss: 0.441003 - AUC: 0.810654
2020-06-29 02:36:39,863 P1608 INFO Save best model: monitor(max): 0.369651
2020-06-29 02:36:39,957 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 02:36:40,030 P1608 INFO Train loss: 0.447759
2020-06-29 02:36:40,030 P1608 INFO ************ Epoch=16 end ************
2020-06-29 03:08:47,980 P1608 INFO [Metrics] logloss: 0.441078 - AUC: 0.810645
2020-06-29 03:08:47,981 P1608 INFO Monitor(max) STOP: 0.369566 !
2020-06-29 03:08:47,981 P1608 INFO Reduce learning rate on plateau: 0.000100
2020-06-29 03:08:47,982 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 03:08:48,042 P1608 INFO Train loss: 0.447619
2020-06-29 03:08:48,042 P1608 INFO ************ Epoch=17 end ************
2020-06-29 03:40:50,309 P1608 INFO [Metrics] logloss: 0.438864 - AUC: 0.813043
2020-06-29 03:40:50,310 P1608 INFO Save best model: monitor(max): 0.374179
2020-06-29 03:40:50,376 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 03:40:50,430 P1608 INFO Train loss: 0.438812
2020-06-29 03:40:50,430 P1608 INFO ************ Epoch=18 end ************
2020-06-29 04:12:51,578 P1608 INFO [Metrics] logloss: 0.438707 - AUC: 0.813291
2020-06-29 04:12:51,580 P1608 INFO Save best model: monitor(max): 0.374584
2020-06-29 04:12:51,656 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 04:12:51,713 P1608 INFO Train loss: 0.435713
2020-06-29 04:12:51,714 P1608 INFO ************ Epoch=19 end ************
2020-06-29 04:44:50,690 P1608 INFO [Metrics] logloss: 0.438830 - AUC: 0.813216
2020-06-29 04:44:50,691 P1608 INFO Monitor(max) STOP: 0.374386 !
2020-06-29 04:44:50,691 P1608 INFO Reduce learning rate on plateau: 0.000010
2020-06-29 04:44:50,692 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 04:44:50,750 P1608 INFO Train loss: 0.434164
2020-06-29 04:44:50,750 P1608 INFO ************ Epoch=20 end ************
2020-06-29 05:16:51,268 P1608 INFO [Metrics] logloss: 0.439293 - AUC: 0.812954
2020-06-29 05:16:51,270 P1608 INFO Monitor(max) STOP: 0.373661 !
2020-06-29 05:16:51,270 P1608 INFO Reduce learning rate on plateau: 0.000001
2020-06-29 05:16:51,272 P1608 INFO Early stopping at epoch=21
2020-06-29 05:16:51,273 P1608 INFO --- 3668/3668 batches finished ---
2020-06-29 05:16:51,325 P1608 INFO Train loss: 0.430879
2020-06-29 05:16:51,325 P1608 INFO Training finished.
2020-06-29 05:16:51,325 P1608 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/FiGNN_criteo/min10/criteo_x4_5c863b0f/FiGNN_criteo_x4_5c863b0f_006_b4207cb3_model.ckpt
2020-06-29 05:16:51,453 P1608 INFO ****** Train/validation evaluation ******
2020-06-29 05:20:56,063 P1608 INFO [Metrics] logloss: 0.426656 - AUC: 0.825899
2020-06-29 05:21:24,008 P1608 INFO [Metrics] logloss: 0.438707 - AUC: 0.813291
2020-06-29 05:21:24,085 P1608 INFO ******** Test evaluation ********
2020-06-29 05:21:24,086 P1608 INFO Loading data...
2020-06-29 05:21:24,086 P1608 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-29 05:21:25,083 P1608 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-29 05:21:25,083 P1608 INFO Loading test data done.
2020-06-29 05:21:52,857 P1608 INFO [Metrics] logloss: 0.438285 - AUC: 0.813803


```