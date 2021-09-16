## HFM_Criteo_x4_002

A notebook to benchmark HFM on Criteo_x4_002 dataset.

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
[Metrics] logloss: 0.441036 - AUC: 0.810985
```


### Logs
```python
2020-05-14 16:20:04,709 P22672 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-5)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_convolution",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_criteo_x4_002_f397374f",
    "model_root": "./Criteo/HFM_criteo/",
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
    "use_dnn": "False",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-05-14 16:20:04,710 P22672 INFO Set up feature encoder...
2020-05-14 16:20:04,710 P22672 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-05-14 16:20:04,716 P22672 INFO Loading data...
2020-05-14 16:20:04,718 P22672 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-05-14 16:20:26,535 P22672 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-05-14 16:20:35,627 P22672 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-05-14 16:20:35,826 P22672 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-05-14 16:20:35,826 P22672 INFO Loading train data done.
2020-05-14 16:20:54,360 P22672 INFO **** Start training: 7335 batches/epoch ****
2020-05-14 17:32:02,813 P22672 INFO [Metrics] logloss: 0.450590 - AUC: 0.800010
2020-05-14 17:32:02,919 P22672 INFO Save best model: monitor(max): 0.349419
2020-05-14 17:32:03,764 P22672 INFO --- 7335/7335 batches finished ---
2020-05-14 17:32:03,874 P22672 INFO Train loss: 0.474985
2020-05-14 17:32:03,874 P22672 INFO ************ Epoch=1 end ************
2020-05-14 18:43:11,132 P22672 INFO [Metrics] logloss: 0.449762 - AUC: 0.801167
2020-05-14 18:43:11,256 P22672 INFO Save best model: monitor(max): 0.351405
2020-05-14 18:43:12,668 P22672 INFO --- 7335/7335 batches finished ---
2020-05-14 18:43:12,813 P22672 INFO Train loss: 0.472243
2020-05-14 18:43:12,813 P22672 INFO ************ Epoch=2 end ************
2020-05-14 19:54:18,886 P22672 INFO [Metrics] logloss: 0.449336 - AUC: 0.801439
2020-05-14 19:54:19,000 P22672 INFO Save best model: monitor(max): 0.352103
2020-05-14 19:54:20,430 P22672 INFO --- 7335/7335 batches finished ---
2020-05-14 19:54:20,576 P22672 INFO Train loss: 0.471811
2020-05-14 19:54:20,576 P22672 INFO ************ Epoch=3 end ************
2020-05-14 21:05:25,631 P22672 INFO [Metrics] logloss: 0.449340 - AUC: 0.801421
2020-05-14 21:05:25,734 P22672 INFO Monitor(max) STOP: 0.352081 !
2020-05-14 21:05:25,734 P22672 INFO Reduce learning rate on plateau: 0.000100
2020-05-14 21:05:25,734 P22672 INFO --- 7335/7335 batches finished ---
2020-05-14 21:05:25,895 P22672 INFO Train loss: 0.471594
2020-05-14 21:05:25,895 P22672 INFO ************ Epoch=4 end ************
2020-05-14 22:16:33,705 P22672 INFO [Metrics] logloss: 0.443870 - AUC: 0.807602
2020-05-14 22:16:33,799 P22672 INFO Save best model: monitor(max): 0.363732
2020-05-14 22:16:35,230 P22672 INFO --- 7335/7335 batches finished ---
2020-05-14 22:16:35,402 P22672 INFO Train loss: 0.452276
2020-05-14 22:16:35,402 P22672 INFO ************ Epoch=5 end ************
2020-05-14 23:27:42,300 P22672 INFO [Metrics] logloss: 0.442857 - AUC: 0.808740
2020-05-14 23:27:42,392 P22672 INFO Save best model: monitor(max): 0.365883
2020-05-14 23:27:43,844 P22672 INFO --- 7335/7335 batches finished ---
2020-05-14 23:27:43,993 P22672 INFO Train loss: 0.447207
2020-05-14 23:27:43,994 P22672 INFO ************ Epoch=6 end ************
2020-05-15 00:38:46,899 P22672 INFO [Metrics] logloss: 0.442285 - AUC: 0.809401
2020-05-15 00:38:47,000 P22672 INFO Save best model: monitor(max): 0.367116
2020-05-15 00:38:48,480 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 00:38:48,631 P22672 INFO Train loss: 0.445488
2020-05-15 00:38:48,631 P22672 INFO ************ Epoch=7 end ************
2020-05-15 01:50:05,056 P22672 INFO [Metrics] logloss: 0.442006 - AUC: 0.809689
2020-05-15 01:50:05,148 P22672 INFO Save best model: monitor(max): 0.367683
2020-05-15 01:50:06,617 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 01:50:06,766 P22672 INFO Train loss: 0.444304
2020-05-15 01:50:06,766 P22672 INFO ************ Epoch=8 end ************
2020-05-15 03:01:08,779 P22672 INFO [Metrics] logloss: 0.441827 - AUC: 0.809900
2020-05-15 03:01:08,905 P22672 INFO Save best model: monitor(max): 0.368073
2020-05-15 03:01:10,373 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 03:01:10,524 P22672 INFO Train loss: 0.443389
2020-05-15 03:01:10,524 P22672 INFO ************ Epoch=9 end ************
2020-05-15 04:12:14,125 P22672 INFO [Metrics] logloss: 0.441815 - AUC: 0.809998
2020-05-15 04:12:14,227 P22672 INFO Save best model: monitor(max): 0.368183
2020-05-15 04:12:15,723 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 04:12:15,871 P22672 INFO Train loss: 0.442660
2020-05-15 04:12:15,872 P22672 INFO ************ Epoch=10 end ************
2020-05-15 05:23:18,905 P22672 INFO [Metrics] logloss: 0.441740 - AUC: 0.810018
2020-05-15 05:23:18,998 P22672 INFO Save best model: monitor(max): 0.368278
2020-05-15 05:23:20,418 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 05:23:20,567 P22672 INFO Train loss: 0.442051
2020-05-15 05:23:20,568 P22672 INFO ************ Epoch=11 end ************
2020-05-15 06:34:23,231 P22672 INFO [Metrics] logloss: 0.441801 - AUC: 0.809972
2020-05-15 06:34:23,323 P22672 INFO Monitor(max) STOP: 0.368171 !
2020-05-15 06:34:23,323 P22672 INFO Reduce learning rate on plateau: 0.000010
2020-05-15 06:34:23,323 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 06:34:23,473 P22672 INFO Train loss: 0.441523
2020-05-15 06:34:23,473 P22672 INFO ************ Epoch=12 end ************
2020-05-15 07:45:24,609 P22672 INFO [Metrics] logloss: 0.441276 - AUC: 0.810634
2020-05-15 07:45:24,698 P22672 INFO Save best model: monitor(max): 0.369358
2020-05-15 07:45:27,125 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 07:45:27,303 P22672 INFO Train loss: 0.434193
2020-05-15 07:45:27,303 P22672 INFO ************ Epoch=13 end ************
2020-05-15 08:56:40,476 P22672 INFO [Metrics] logloss: 0.441283 - AUC: 0.810671
2020-05-15 08:56:40,566 P22672 INFO Save best model: monitor(max): 0.369388
2020-05-15 08:56:42,024 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 08:56:42,185 P22672 INFO Train loss: 0.433268
2020-05-15 08:56:42,185 P22672 INFO ************ Epoch=14 end ************
2020-05-15 10:07:54,364 P22672 INFO [Metrics] logloss: 0.441345 - AUC: 0.810627
2020-05-15 10:07:54,456 P22672 INFO Monitor(max) STOP: 0.369282 !
2020-05-15 10:07:54,456 P22672 INFO Reduce learning rate on plateau: 0.000001
2020-05-15 10:07:54,456 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 10:07:54,639 P22672 INFO Train loss: 0.432872
2020-05-15 10:07:54,640 P22672 INFO ************ Epoch=15 end ************
2020-05-15 11:19:07,456 P22672 INFO [Metrics] logloss: 0.441343 - AUC: 0.810646
2020-05-15 11:19:07,571 P22672 INFO Monitor(max) STOP: 0.369303 !
2020-05-15 11:19:07,571 P22672 INFO Reduce learning rate on plateau: 0.000001
2020-05-15 11:19:07,571 P22672 INFO Early stopping at epoch=16
2020-05-15 11:19:07,571 P22672 INFO --- 7335/7335 batches finished ---
2020-05-15 11:19:07,745 P22672 INFO Train loss: 0.431486
2020-05-15 11:19:07,746 P22672 INFO Training finished.
2020-05-15 11:19:07,746 P22672 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Criteo/HFM_criteo/criteo_x4_001_be98441d/HFM_criteo_x4_002_f397374f_criteo_x4_001_be98441d_model.ckpt
2020-05-15 11:19:09,165 P22672 INFO ****** Train/validation evaluation ******
2020-05-15 11:39:00,923 P22672 INFO [Metrics] logloss: 0.424450 - AUC: 0.828188
2020-05-15 11:41:27,021 P22672 INFO [Metrics] logloss: 0.441283 - AUC: 0.810671
2020-05-15 11:41:27,541 P22672 INFO ******** Test evaluation ********
2020-05-15 11:41:27,541 P22672 INFO Loading data...
2020-05-15 11:41:27,541 P22672 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-05-15 11:41:29,119 P22672 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-05-15 11:41:29,119 P22672 INFO Loading test data done.
2020-05-15 11:43:53,250 P22672 INFO [Metrics] logloss: 0.441036 - AUC: 0.810985

```