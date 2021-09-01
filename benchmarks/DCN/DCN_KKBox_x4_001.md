## DCN_KKBox_x4_001

A notebook to benchmark DCN on KKBox_x4_001 dataset.

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
In this setting, For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10.

To make a fair comparison, we fix embedding_dim=128, which performs well.


### Code




### Results
```python
[Metrics] logloss: 0.476084 - AUC: 0.853383
```


### Logs
```python
2020-04-11 17:37:07,399 P40782 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "4",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[5000, 5000]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DCN",
    "model_id": "DCN_kkbox_x4_005_5c63f623",
    "model_root": "./KKBox/DCN_kkbox/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.4",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x4/test.csv",
    "train_data": "../data/KKBox/KKBox_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-04-11 17:37:07,400 P40782 INFO Set up feature encoder...
2020-04-11 17:37:07,400 P40782 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_encoder.pkl
2020-04-11 17:37:07,541 P40782 INFO Loading data...
2020-04-11 17:37:07,546 P40782 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-04-11 17:37:07,941 P40782 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-04-11 17:37:08,138 P40782 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-11 17:37:08,157 P40782 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-11 17:37:08,157 P40782 INFO Loading train data done.
2020-04-11 17:37:14,116 P40782 INFO **** Start training: 591 batches/epoch ****
2020-04-11 17:44:03,100 P40782 INFO [Metrics] logloss: 0.556179 - AUC: 0.785663
2020-04-11 17:44:03,101 P40782 INFO Save best model: monitor(max): 0.229484
2020-04-11 17:44:03,299 P40782 INFO --- 591/591 batches finished ---
2020-04-11 17:44:03,347 P40782 INFO Train loss: 0.608789
2020-04-11 17:44:03,347 P40782 INFO ************ Epoch=1 end ************
2020-04-11 17:51:07,869 P40782 INFO [Metrics] logloss: 0.542583 - AUC: 0.798728
2020-04-11 17:51:07,870 P40782 INFO Save best model: monitor(max): 0.256145
2020-04-11 17:51:08,181 P40782 INFO --- 591/591 batches finished ---
2020-04-11 17:51:08,232 P40782 INFO Train loss: 0.590351
2020-04-11 17:51:08,232 P40782 INFO ************ Epoch=2 end ************
2020-04-11 17:54:57,501 P40782 INFO [Metrics] logloss: 0.534695 - AUC: 0.805944
2020-04-11 17:54:57,503 P40782 INFO Save best model: monitor(max): 0.271249
2020-04-11 17:54:57,843 P40782 INFO --- 591/591 batches finished ---
2020-04-11 17:54:57,895 P40782 INFO Train loss: 0.583235
2020-04-11 17:54:57,895 P40782 INFO ************ Epoch=3 end ************
2020-04-11 18:01:24,479 P40782 INFO [Metrics] logloss: 0.528044 - AUC: 0.811404
2020-04-11 18:01:24,480 P40782 INFO Save best model: monitor(max): 0.283360
2020-04-11 18:01:24,836 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:01:24,886 P40782 INFO Train loss: 0.578949
2020-04-11 18:01:24,887 P40782 INFO ************ Epoch=4 end ************
2020-04-11 18:07:47,351 P40782 INFO [Metrics] logloss: 0.523614 - AUC: 0.815199
2020-04-11 18:07:47,353 P40782 INFO Save best model: monitor(max): 0.291585
2020-04-11 18:07:47,690 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:07:47,742 P40782 INFO Train loss: 0.576046
2020-04-11 18:07:47,742 P40782 INFO ************ Epoch=5 end ************
2020-04-11 18:12:38,688 P40782 INFO [Metrics] logloss: 0.519921 - AUC: 0.818364
2020-04-11 18:12:38,689 P40782 INFO Save best model: monitor(max): 0.298443
2020-04-11 18:12:39,052 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:12:39,105 P40782 INFO Train loss: 0.573769
2020-04-11 18:12:39,105 P40782 INFO ************ Epoch=6 end ************
2020-04-11 18:19:13,569 P40782 INFO [Metrics] logloss: 0.516951 - AUC: 0.820672
2020-04-11 18:19:13,570 P40782 INFO Save best model: monitor(max): 0.303722
2020-04-11 18:19:13,911 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:19:13,964 P40782 INFO Train loss: 0.572344
2020-04-11 18:19:13,964 P40782 INFO ************ Epoch=7 end ************
2020-04-11 18:26:29,641 P40782 INFO [Metrics] logloss: 0.515758 - AUC: 0.821852
2020-04-11 18:26:29,643 P40782 INFO Save best model: monitor(max): 0.306094
2020-04-11 18:26:29,974 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:26:30,027 P40782 INFO Train loss: 0.570699
2020-04-11 18:26:30,027 P40782 INFO ************ Epoch=8 end ************
2020-04-11 18:31:37,836 P40782 INFO [Metrics] logloss: 0.513236 - AUC: 0.823744
2020-04-11 18:31:37,837 P40782 INFO Save best model: monitor(max): 0.310508
2020-04-11 18:31:38,147 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:31:38,206 P40782 INFO Train loss: 0.569583
2020-04-11 18:31:38,206 P40782 INFO ************ Epoch=9 end ************
2020-04-11 18:38:00,822 P40782 INFO [Metrics] logloss: 0.511549 - AUC: 0.825283
2020-04-11 18:38:00,823 P40782 INFO Save best model: monitor(max): 0.313734
2020-04-11 18:38:01,134 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:38:01,188 P40782 INFO Train loss: 0.568534
2020-04-11 18:38:01,188 P40782 INFO ************ Epoch=10 end ************
2020-04-11 18:43:34,318 P40782 INFO [Metrics] logloss: 0.509876 - AUC: 0.826778
2020-04-11 18:43:34,319 P40782 INFO Save best model: monitor(max): 0.316902
2020-04-11 18:43:34,621 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:43:34,674 P40782 INFO Train loss: 0.567527
2020-04-11 18:43:34,674 P40782 INFO ************ Epoch=11 end ************
2020-04-11 18:49:31,373 P40782 INFO [Metrics] logloss: 0.508628 - AUC: 0.827671
2020-04-11 18:49:31,375 P40782 INFO Save best model: monitor(max): 0.319042
2020-04-11 18:49:31,688 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:49:31,769 P40782 INFO Train loss: 0.566973
2020-04-11 18:49:31,769 P40782 INFO ************ Epoch=12 end ************
2020-04-11 18:56:11,922 P40782 INFO [Metrics] logloss: 0.506928 - AUC: 0.828857
2020-04-11 18:56:11,924 P40782 INFO Save best model: monitor(max): 0.321928
2020-04-11 18:56:12,281 P40782 INFO --- 591/591 batches finished ---
2020-04-11 18:56:12,335 P40782 INFO Train loss: 0.566196
2020-04-11 18:56:12,335 P40782 INFO ************ Epoch=13 end ************
2020-04-11 19:02:15,325 P40782 INFO [Metrics] logloss: 0.506097 - AUC: 0.829484
2020-04-11 19:02:15,326 P40782 INFO Save best model: monitor(max): 0.323387
2020-04-11 19:02:15,645 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:02:15,694 P40782 INFO Train loss: 0.565535
2020-04-11 19:02:15,695 P40782 INFO ************ Epoch=14 end ************
2020-04-11 19:08:19,233 P40782 INFO [Metrics] logloss: 0.505268 - AUC: 0.830412
2020-04-11 19:08:19,234 P40782 INFO Save best model: monitor(max): 0.325144
2020-04-11 19:08:19,549 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:08:19,601 P40782 INFO Train loss: 0.564906
2020-04-11 19:08:19,601 P40782 INFO ************ Epoch=15 end ************
2020-04-11 19:14:58,448 P40782 INFO [Metrics] logloss: 0.504191 - AUC: 0.831233
2020-04-11 19:14:58,450 P40782 INFO Save best model: monitor(max): 0.327042
2020-04-11 19:14:58,787 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:14:58,838 P40782 INFO Train loss: 0.564342
2020-04-11 19:14:58,839 P40782 INFO ************ Epoch=16 end ************
2020-04-11 19:22:14,539 P40782 INFO [Metrics] logloss: 0.502812 - AUC: 0.832187
2020-04-11 19:22:14,540 P40782 INFO Save best model: monitor(max): 0.329374
2020-04-11 19:22:14,887 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:22:14,938 P40782 INFO Train loss: 0.564020
2020-04-11 19:22:14,938 P40782 INFO ************ Epoch=17 end ************
2020-04-11 19:28:06,615 P40782 INFO [Metrics] logloss: 0.502611 - AUC: 0.832399
2020-04-11 19:28:06,617 P40782 INFO Save best model: monitor(max): 0.329788
2020-04-11 19:28:06,945 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:28:06,999 P40782 INFO Train loss: 0.563510
2020-04-11 19:28:06,999 P40782 INFO ************ Epoch=18 end ************
2020-04-11 19:34:59,372 P40782 INFO [Metrics] logloss: 0.502228 - AUC: 0.832764
2020-04-11 19:34:59,373 P40782 INFO Save best model: monitor(max): 0.330536
2020-04-11 19:34:59,688 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:34:59,741 P40782 INFO Train loss: 0.562946
2020-04-11 19:34:59,741 P40782 INFO ************ Epoch=19 end ************
2020-04-11 19:41:36,216 P40782 INFO [Metrics] logloss: 0.501895 - AUC: 0.833105
2020-04-11 19:41:36,217 P40782 INFO Save best model: monitor(max): 0.331210
2020-04-11 19:41:36,550 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:41:36,604 P40782 INFO Train loss: 0.562514
2020-04-11 19:41:36,604 P40782 INFO ************ Epoch=20 end ************
2020-04-11 19:47:04,225 P40782 INFO [Metrics] logloss: 0.500819 - AUC: 0.833760
2020-04-11 19:47:04,226 P40782 INFO Save best model: monitor(max): 0.332941
2020-04-11 19:47:04,533 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:47:04,585 P40782 INFO Train loss: 0.562277
2020-04-11 19:47:04,585 P40782 INFO ************ Epoch=21 end ************
2020-04-11 19:54:19,984 P40782 INFO [Metrics] logloss: 0.499973 - AUC: 0.834465
2020-04-11 19:54:19,986 P40782 INFO Save best model: monitor(max): 0.334492
2020-04-11 19:54:20,222 P40782 INFO --- 591/591 batches finished ---
2020-04-11 19:54:20,274 P40782 INFO Train loss: 0.561999
2020-04-11 19:54:20,275 P40782 INFO ************ Epoch=22 end ************
2020-04-11 20:00:54,930 P40782 INFO [Metrics] logloss: 0.499307 - AUC: 0.834875
2020-04-11 20:00:54,931 P40782 INFO Save best model: monitor(max): 0.335569
2020-04-11 20:00:55,281 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:00:55,333 P40782 INFO Train loss: 0.561477
2020-04-11 20:00:55,334 P40782 INFO ************ Epoch=23 end ************
2020-04-11 20:06:20,191 P40782 INFO [Metrics] logloss: 0.499257 - AUC: 0.835190
2020-04-11 20:06:20,192 P40782 INFO Save best model: monitor(max): 0.335933
2020-04-11 20:06:20,496 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:06:20,548 P40782 INFO Train loss: 0.561141
2020-04-11 20:06:20,548 P40782 INFO ************ Epoch=24 end ************
2020-04-11 20:13:00,446 P40782 INFO [Metrics] logloss: 0.498474 - AUC: 0.835703
2020-04-11 20:13:00,447 P40782 INFO Save best model: monitor(max): 0.337229
2020-04-11 20:13:00,806 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:13:00,857 P40782 INFO Train loss: 0.560711
2020-04-11 20:13:00,858 P40782 INFO ************ Epoch=25 end ************
2020-04-11 20:18:21,948 P40782 INFO [Metrics] logloss: 0.498332 - AUC: 0.836161
2020-04-11 20:18:21,949 P40782 INFO Save best model: monitor(max): 0.337829
2020-04-11 20:18:22,276 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:18:22,328 P40782 INFO Train loss: 0.560537
2020-04-11 20:18:22,328 P40782 INFO ************ Epoch=26 end ************
2020-04-11 20:25:01,892 P40782 INFO [Metrics] logloss: 0.497432 - AUC: 0.836884
2020-04-11 20:25:01,894 P40782 INFO Save best model: monitor(max): 0.339452
2020-04-11 20:25:02,202 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:25:02,264 P40782 INFO Train loss: 0.560266
2020-04-11 20:25:02,264 P40782 INFO ************ Epoch=27 end ************
2020-04-11 20:32:13,717 P40782 INFO [Metrics] logloss: 0.496015 - AUC: 0.837490
2020-04-11 20:32:13,718 P40782 INFO Save best model: monitor(max): 0.341475
2020-04-11 20:32:14,058 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:32:14,112 P40782 INFO Train loss: 0.559953
2020-04-11 20:32:14,112 P40782 INFO ************ Epoch=28 end ************
2020-04-11 20:37:40,674 P40782 INFO [Metrics] logloss: 0.495820 - AUC: 0.837573
2020-04-11 20:37:40,675 P40782 INFO Save best model: monitor(max): 0.341753
2020-04-11 20:37:40,997 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:37:41,052 P40782 INFO Train loss: 0.559782
2020-04-11 20:37:41,052 P40782 INFO ************ Epoch=29 end ************
2020-04-11 20:44:20,108 P40782 INFO [Metrics] logloss: 0.495674 - AUC: 0.837591
2020-04-11 20:44:20,110 P40782 INFO Save best model: monitor(max): 0.341918
2020-04-11 20:44:20,429 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:44:20,482 P40782 INFO Train loss: 0.559437
2020-04-11 20:44:20,482 P40782 INFO ************ Epoch=30 end ************
2020-04-11 20:49:52,041 P40782 INFO [Metrics] logloss: 0.495348 - AUC: 0.838052
2020-04-11 20:49:52,042 P40782 INFO Save best model: monitor(max): 0.342704
2020-04-11 20:49:52,360 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:49:52,411 P40782 INFO Train loss: 0.559124
2020-04-11 20:49:52,411 P40782 INFO ************ Epoch=31 end ************
2020-04-11 20:56:56,693 P40782 INFO [Metrics] logloss: 0.495240 - AUC: 0.838354
2020-04-11 20:56:56,694 P40782 INFO Save best model: monitor(max): 0.343114
2020-04-11 20:56:57,031 P40782 INFO --- 591/591 batches finished ---
2020-04-11 20:56:57,083 P40782 INFO Train loss: 0.558825
2020-04-11 20:56:57,084 P40782 INFO ************ Epoch=32 end ************
2020-04-11 21:03:50,749 P40782 INFO [Metrics] logloss: 0.494493 - AUC: 0.838630
2020-04-11 21:03:50,751 P40782 INFO Save best model: monitor(max): 0.344136
2020-04-11 21:03:51,063 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:03:51,115 P40782 INFO Train loss: 0.558561
2020-04-11 21:03:51,115 P40782 INFO ************ Epoch=33 end ************
2020-04-11 21:09:13,899 P40782 INFO [Metrics] logloss: 0.494701 - AUC: 0.838799
2020-04-11 21:09:13,900 P40782 INFO Monitor(max) STOP: 0.344098 !
2020-04-11 21:09:13,900 P40782 INFO Reduce learning rate on plateau: 0.000100
2020-04-11 21:09:13,900 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:09:13,999 P40782 INFO Train loss: 0.558098
2020-04-11 21:09:13,999 P40782 INFO ************ Epoch=34 end ************
2020-04-11 21:15:54,184 P40782 INFO [Metrics] logloss: 0.480304 - AUC: 0.849422
2020-04-11 21:15:54,185 P40782 INFO Save best model: monitor(max): 0.369118
2020-04-11 21:15:54,498 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:15:54,548 P40782 INFO Train loss: 0.499121
2020-04-11 21:15:54,549 P40782 INFO ************ Epoch=35 end ************
2020-04-11 21:22:23,729 P40782 INFO [Metrics] logloss: 0.477396 - AUC: 0.851599
2020-04-11 21:22:23,730 P40782 INFO Save best model: monitor(max): 0.374203
2020-04-11 21:22:24,041 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:22:24,094 P40782 INFO Train loss: 0.470523
2020-04-11 21:22:24,094 P40782 INFO ************ Epoch=36 end ************
2020-04-11 21:28:17,169 P40782 INFO [Metrics] logloss: 0.476635 - AUC: 0.852635
2020-04-11 21:28:17,170 P40782 INFO Save best model: monitor(max): 0.376000
2020-04-11 21:28:17,482 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:28:17,546 P40782 INFO Train loss: 0.459874
2020-04-11 21:28:17,546 P40782 INFO ************ Epoch=37 end ************
2020-04-11 21:34:43,604 P40782 INFO [Metrics] logloss: 0.476751 - AUC: 0.852886
2020-04-11 21:34:43,605 P40782 INFO Save best model: monitor(max): 0.376135
2020-04-11 21:34:43,947 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:34:44,001 P40782 INFO Train loss: 0.453306
2020-04-11 21:34:44,001 P40782 INFO ************ Epoch=38 end ************
2020-04-11 21:40:48,542 P40782 INFO [Metrics] logloss: 0.477345 - AUC: 0.852883
2020-04-11 21:40:48,543 P40782 INFO Monitor(max) STOP: 0.375537 !
2020-04-11 21:40:48,543 P40782 INFO Reduce learning rate on plateau: 0.000010
2020-04-11 21:40:48,543 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:40:48,597 P40782 INFO Train loss: 0.448218
2020-04-11 21:40:48,598 P40782 INFO ************ Epoch=39 end ************
2020-04-11 21:47:23,301 P40782 INFO [Metrics] logloss: 0.481242 - AUC: 0.852635
2020-04-11 21:47:23,302 P40782 INFO Monitor(max) STOP: 0.371394 !
2020-04-11 21:47:23,302 P40782 INFO Reduce learning rate on plateau: 0.000001
2020-04-11 21:47:23,302 P40782 INFO Early stopping at epoch=40
2020-04-11 21:47:23,302 P40782 INFO --- 591/591 batches finished ---
2020-04-11 21:47:23,368 P40782 INFO Train loss: 0.430542
2020-04-11 21:47:23,368 P40782 INFO Training finished.
2020-04-11 21:47:23,368 P40782 INFO Load best model: /home/zhujieming/xxx/OpenCTR1030/benchmarks/KKBox/DCN_kkbox/kkbox_x4_001_c5c9c6e3/DCN_kkbox_x4_005_5c63f623_model.ckpt
2020-04-11 21:47:23,669 P40782 INFO ****** Train/validation evaluation ******
2020-04-11 21:49:27,337 P40782 INFO [Metrics] logloss: 0.389470 - AUC: 0.906981
2020-04-11 21:49:42,876 P40782 INFO [Metrics] logloss: 0.476751 - AUC: 0.852886
2020-04-11 21:49:42,944 P40782 INFO ******** Test evaluation ********
2020-04-11 21:49:42,945 P40782 INFO Loading data...
2020-04-11 21:49:42,945 P40782 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-04-11 21:49:43,012 P40782 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-11 21:49:43,012 P40782 INFO Loading test data done.
2020-04-11 21:49:55,767 P40782 INFO [Metrics] logloss: 0.476084 - AUC: 0.853383


```
