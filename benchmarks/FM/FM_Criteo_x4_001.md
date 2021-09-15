## FM_Criteo_x4_001

A notebook to benchmark FM on Criteo_x4_001 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default <OOV> token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless id field nor specially preprocess the timestamp field.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.


### Code




### Results
```python
[Metrics] logloss: 0.443109 - AUC: 0.808607
```


### Logs
```python
2021-09-01 02:44:40,917 P35557 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_9ea3bdfc",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'na_value': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'na_value': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FM",
    "model_id": "FM_criteo_x4_003_3da0082a",
    "model_root": "./Criteo/FM_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2021-09-01 02:44:40,918 P35557 INFO Set up feature encoder...
2021-09-01 02:44:40,918 P35557 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x4_9ea3bdfc/feature_encoder.pkl
2021-09-01 02:44:42,392 P35557 INFO Total number of parameters: 15482037.
2021-09-01 02:44:42,392 P35557 INFO Loading data...
2021-09-01 02:44:42,394 P35557 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/train.h5
2021-09-01 02:44:47,274 P35557 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/valid.h5
2021-09-01 02:44:48,941 P35557 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2021-09-01 02:44:49,074 P35557 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-01 02:44:49,074 P35557 INFO Loading train data done.
2021-09-01 02:44:51,847 P35557 INFO Start training: 3668 batches/epoch
2021-09-01 02:44:51,848 P35557 INFO ************ Epoch=1 start ************
2021-09-01 02:58:39,834 P35557 INFO [Metrics] logloss: 0.451252 - AUC: 0.799586
2021-09-01 02:58:39,836 P35557 INFO Save best model: monitor(max): 0.348335
2021-09-01 02:58:39,922 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 02:58:39,975 P35557 INFO Train loss: 0.466181
2021-09-01 02:58:39,976 P35557 INFO ************ Epoch=1 end ************
2021-09-01 03:12:28,377 P35557 INFO [Metrics] logloss: 0.450115 - AUC: 0.800813
2021-09-01 03:12:28,378 P35557 INFO Save best model: monitor(max): 0.350698
2021-09-01 03:12:28,493 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 03:12:28,547 P35557 INFO Train loss: 0.460374
2021-09-01 03:12:28,547 P35557 INFO ************ Epoch=2 end ************
2021-09-01 03:26:17,613 P35557 INFO [Metrics] logloss: 0.449570 - AUC: 0.801442
2021-09-01 03:26:17,614 P35557 INFO Save best model: monitor(max): 0.351872
2021-09-01 03:26:17,743 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 03:26:17,801 P35557 INFO Train loss: 0.459731
2021-09-01 03:26:17,802 P35557 INFO ************ Epoch=3 end ************
2021-09-01 03:40:05,183 P35557 INFO [Metrics] logloss: 0.449285 - AUC: 0.801780
2021-09-01 03:40:05,184 P35557 INFO Save best model: monitor(max): 0.352495
2021-09-01 03:40:05,270 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 03:40:05,310 P35557 INFO Train loss: 0.459468
2021-09-01 03:40:05,310 P35557 INFO ************ Epoch=4 end ************
2021-09-01 03:53:48,959 P35557 INFO [Metrics] logloss: 0.449177 - AUC: 0.801886
2021-09-01 03:53:48,960 P35557 INFO Save best model: monitor(max): 0.352709
2021-09-01 03:53:49,046 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 03:53:49,104 P35557 INFO Train loss: 0.459320
2021-09-01 03:53:49,104 P35557 INFO ************ Epoch=5 end ************
2021-09-01 04:07:38,067 P35557 INFO [Metrics] logloss: 0.449182 - AUC: 0.801950
2021-09-01 04:07:38,068 P35557 INFO Save best model: monitor(max): 0.352768
2021-09-01 04:07:38,161 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 04:07:38,220 P35557 INFO Train loss: 0.459205
2021-09-01 04:07:38,221 P35557 INFO ************ Epoch=6 end ************
2021-09-01 04:21:27,386 P35557 INFO [Metrics] logloss: 0.449200 - AUC: 0.801831
2021-09-01 04:21:27,388 P35557 INFO Monitor(max) STOP: 0.352631 !
2021-09-01 04:21:27,388 P35557 INFO Reduce learning rate on plateau: 0.000100
2021-09-01 04:21:27,388 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 04:21:27,429 P35557 INFO Train loss: 0.459127
2021-09-01 04:21:27,429 P35557 INFO ************ Epoch=7 end ************
2021-09-01 04:35:16,418 P35557 INFO [Metrics] logloss: 0.445348 - AUC: 0.806075
2021-09-01 04:35:16,419 P35557 INFO Save best model: monitor(max): 0.360727
2021-09-01 04:35:16,511 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 04:35:16,553 P35557 INFO Train loss: 0.450131
2021-09-01 04:35:16,553 P35557 INFO ************ Epoch=8 end ************
2021-09-01 04:49:06,244 P35557 INFO [Metrics] logloss: 0.444825 - AUC: 0.806666
2021-09-01 04:49:06,245 P35557 INFO Save best model: monitor(max): 0.361841
2021-09-01 04:49:06,366 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 04:49:06,413 P35557 INFO Train loss: 0.447164
2021-09-01 04:49:06,413 P35557 INFO ************ Epoch=9 end ************
2021-09-01 05:02:55,940 P35557 INFO [Metrics] logloss: 0.444550 - AUC: 0.806961
2021-09-01 05:02:55,942 P35557 INFO Save best model: monitor(max): 0.362411
2021-09-01 05:02:56,035 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 05:02:56,101 P35557 INFO Train loss: 0.446370
2021-09-01 05:02:56,102 P35557 INFO ************ Epoch=10 end ************
2021-09-01 05:16:44,408 P35557 INFO [Metrics] logloss: 0.444395 - AUC: 0.807154
2021-09-01 05:16:44,410 P35557 INFO Save best model: monitor(max): 0.362759
2021-09-01 05:16:44,515 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 05:16:44,580 P35557 INFO Train loss: 0.445917
2021-09-01 05:16:44,581 P35557 INFO ************ Epoch=11 end ************
2021-09-01 05:30:33,610 P35557 INFO [Metrics] logloss: 0.444260 - AUC: 0.807283
2021-09-01 05:30:33,612 P35557 INFO Save best model: monitor(max): 0.363023
2021-09-01 05:30:33,704 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 05:30:33,749 P35557 INFO Train loss: 0.445582
2021-09-01 05:30:33,749 P35557 INFO ************ Epoch=12 end ************
2021-09-01 05:44:22,557 P35557 INFO [Metrics] logloss: 0.444169 - AUC: 0.807391
2021-09-01 05:44:22,558 P35557 INFO Save best model: monitor(max): 0.363222
2021-09-01 05:44:22,648 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 05:44:22,693 P35557 INFO Train loss: 0.445307
2021-09-01 05:44:22,693 P35557 INFO ************ Epoch=13 end ************
2021-09-01 05:58:12,774 P35557 INFO [Metrics] logloss: 0.444092 - AUC: 0.807472
2021-09-01 05:58:12,775 P35557 INFO Save best model: monitor(max): 0.363380
2021-09-01 05:58:12,868 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 05:58:12,910 P35557 INFO Train loss: 0.445069
2021-09-01 05:58:12,910 P35557 INFO ************ Epoch=14 end ************
2021-09-01 06:12:02,623 P35557 INFO [Metrics] logloss: 0.444006 - AUC: 0.807559
2021-09-01 06:12:02,624 P35557 INFO Save best model: monitor(max): 0.363554
2021-09-01 06:12:02,717 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 06:12:02,760 P35557 INFO Train loss: 0.444872
2021-09-01 06:12:02,760 P35557 INFO ************ Epoch=15 end ************
2021-09-01 06:25:53,156 P35557 INFO [Metrics] logloss: 0.443980 - AUC: 0.807590
2021-09-01 06:25:53,157 P35557 INFO Save best model: monitor(max): 0.363610
2021-09-01 06:25:53,247 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 06:25:53,289 P35557 INFO Train loss: 0.444688
2021-09-01 06:25:53,289 P35557 INFO ************ Epoch=16 end ************
2021-09-01 06:39:42,185 P35557 INFO [Metrics] logloss: 0.443928 - AUC: 0.807652
2021-09-01 06:39:42,186 P35557 INFO Save best model: monitor(max): 0.363724
2021-09-01 06:39:42,274 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 06:39:42,320 P35557 INFO Train loss: 0.444528
2021-09-01 06:39:42,320 P35557 INFO ************ Epoch=17 end ************
2021-09-01 06:53:30,584 P35557 INFO [Metrics] logloss: 0.443878 - AUC: 0.807698
2021-09-01 06:53:30,585 P35557 INFO Save best model: monitor(max): 0.363820
2021-09-01 06:53:30,676 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 06:53:30,722 P35557 INFO Train loss: 0.444380
2021-09-01 06:53:30,722 P35557 INFO ************ Epoch=18 end ************
2021-09-01 07:07:21,872 P35557 INFO [Metrics] logloss: 0.443843 - AUC: 0.807742
2021-09-01 07:07:21,873 P35557 INFO Save best model: monitor(max): 0.363899
2021-09-01 07:07:21,963 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 07:07:22,008 P35557 INFO Train loss: 0.444239
2021-09-01 07:07:22,008 P35557 INFO ************ Epoch=19 end ************
2021-09-01 07:21:10,503 P35557 INFO [Metrics] logloss: 0.443810 - AUC: 0.807783
2021-09-01 07:21:10,505 P35557 INFO Save best model: monitor(max): 0.363973
2021-09-01 07:21:10,599 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 07:21:10,642 P35557 INFO Train loss: 0.444115
2021-09-01 07:21:10,642 P35557 INFO ************ Epoch=20 end ************
2021-09-01 07:35:10,789 P35557 INFO [Metrics] logloss: 0.443776 - AUC: 0.807819
2021-09-01 07:35:10,791 P35557 INFO Save best model: monitor(max): 0.364042
2021-09-01 07:35:10,880 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 07:35:10,946 P35557 INFO Train loss: 0.443998
2021-09-01 07:35:10,946 P35557 INFO ************ Epoch=21 end ************
2021-09-01 07:49:00,939 P35557 INFO [Metrics] logloss: 0.443712 - AUC: 0.807886
2021-09-01 07:49:00,941 P35557 INFO Save best model: monitor(max): 0.364174
2021-09-01 07:49:01,033 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 07:49:01,094 P35557 INFO Train loss: 0.443889
2021-09-01 07:49:01,094 P35557 INFO ************ Epoch=22 end ************
2021-09-01 08:02:50,261 P35557 INFO [Metrics] logloss: 0.443722 - AUC: 0.807884
2021-09-01 08:02:50,262 P35557 INFO Monitor(max) STOP: 0.364162 !
2021-09-01 08:02:50,262 P35557 INFO Reduce learning rate on plateau: 0.000010
2021-09-01 08:02:50,262 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 08:02:50,322 P35557 INFO Train loss: 0.443781
2021-09-01 08:02:50,323 P35557 INFO ************ Epoch=23 end ************
2021-09-01 08:16:39,086 P35557 INFO [Metrics] logloss: 0.443512 - AUC: 0.808103
2021-09-01 08:16:39,087 P35557 INFO Save best model: monitor(max): 0.364591
2021-09-01 08:16:39,193 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 08:16:39,253 P35557 INFO Train loss: 0.441420
2021-09-01 08:16:39,253 P35557 INFO ************ Epoch=24 end ************
2021-09-01 08:30:29,573 P35557 INFO [Metrics] logloss: 0.443474 - AUC: 0.808143
2021-09-01 08:30:29,574 P35557 INFO Save best model: monitor(max): 0.364669
2021-09-01 08:30:29,664 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 08:30:29,725 P35557 INFO Train loss: 0.441287
2021-09-01 08:30:29,725 P35557 INFO ************ Epoch=25 end ************
2021-09-01 08:44:19,427 P35557 INFO [Metrics] logloss: 0.443459 - AUC: 0.808159
2021-09-01 08:44:19,428 P35557 INFO Save best model: monitor(max): 0.364700
2021-09-01 08:44:19,519 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 08:44:19,561 P35557 INFO Train loss: 0.441218
2021-09-01 08:44:19,561 P35557 INFO ************ Epoch=26 end ************
2021-09-01 08:58:09,065 P35557 INFO [Metrics] logloss: 0.443450 - AUC: 0.808170
2021-09-01 08:58:09,067 P35557 INFO Save best model: monitor(max): 0.364721
2021-09-01 08:58:09,159 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 08:58:09,204 P35557 INFO Train loss: 0.441177
2021-09-01 08:58:09,204 P35557 INFO ************ Epoch=27 end ************
2021-09-01 09:12:25,183 P35557 INFO [Metrics] logloss: 0.443443 - AUC: 0.808181
2021-09-01 09:12:25,184 P35557 INFO Save best model: monitor(max): 0.364738
2021-09-01 09:12:25,289 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 09:12:25,354 P35557 INFO Train loss: 0.441148
2021-09-01 09:12:25,354 P35557 INFO ************ Epoch=28 end ************
2021-09-01 09:26:15,054 P35557 INFO [Metrics] logloss: 0.443440 - AUC: 0.808181
2021-09-01 09:26:15,056 P35557 INFO Save best model: monitor(max): 0.364741
2021-09-01 09:26:15,148 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 09:26:15,210 P35557 INFO Train loss: 0.441116
2021-09-01 09:26:15,210 P35557 INFO ************ Epoch=29 end ************
2021-09-01 09:40:04,295 P35557 INFO [Metrics] logloss: 0.443436 - AUC: 0.808186
2021-09-01 09:40:04,296 P35557 INFO Save best model: monitor(max): 0.364750
2021-09-01 09:40:04,402 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 09:40:04,445 P35557 INFO Train loss: 0.441096
2021-09-01 09:40:04,445 P35557 INFO ************ Epoch=30 end ************
2021-09-01 09:53:55,277 P35557 INFO [Metrics] logloss: 0.443432 - AUC: 0.808186
2021-09-01 09:53:55,279 P35557 INFO Save best model: monitor(max): 0.364754
2021-09-01 09:53:55,371 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 09:53:55,432 P35557 INFO Train loss: 0.441072
2021-09-01 09:53:55,432 P35557 INFO ************ Epoch=31 end ************
2021-09-01 10:07:44,201 P35557 INFO [Metrics] logloss: 0.443433 - AUC: 0.808189
2021-09-01 10:07:44,202 P35557 INFO Save best model: monitor(max): 0.364755
2021-09-01 10:07:44,292 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 10:07:44,356 P35557 INFO Train loss: 0.441049
2021-09-01 10:07:44,356 P35557 INFO ************ Epoch=32 end ************
2021-09-01 10:21:32,637 P35557 INFO [Metrics] logloss: 0.443426 - AUC: 0.808194
2021-09-01 10:21:32,638 P35557 INFO Save best model: monitor(max): 0.364768
2021-09-01 10:21:32,735 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 10:21:32,797 P35557 INFO Train loss: 0.441028
2021-09-01 10:21:32,798 P35557 INFO ************ Epoch=33 end ************
2021-09-01 10:35:31,045 P35557 INFO [Metrics] logloss: 0.443425 - AUC: 0.808192
2021-09-01 10:35:31,046 P35557 INFO Monitor(max) STOP: 0.364767 !
2021-09-01 10:35:31,046 P35557 INFO Reduce learning rate on plateau: 0.000001
2021-09-01 10:35:31,046 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 10:35:31,107 P35557 INFO Train loss: 0.441016
2021-09-01 10:35:31,107 P35557 INFO ************ Epoch=34 end ************
2021-09-01 10:49:20,422 P35557 INFO [Metrics] logloss: 0.443422 - AUC: 0.808197
2021-09-01 10:49:20,423 P35557 INFO Save best model: monitor(max): 0.364774
2021-09-01 10:49:20,524 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 10:49:20,567 P35557 INFO Train loss: 0.440692
2021-09-01 10:49:20,567 P35557 INFO ************ Epoch=35 end ************
2021-09-01 11:03:09,839 P35557 INFO [Metrics] logloss: 0.443422 - AUC: 0.808197
2021-09-01 11:03:09,840 P35557 INFO Save best model: monitor(max): 0.364776
2021-09-01 11:03:09,940 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 11:03:10,005 P35557 INFO Train loss: 0.440688
2021-09-01 11:03:10,005 P35557 INFO ************ Epoch=36 end ************
2021-09-01 11:16:58,118 P35557 INFO [Metrics] logloss: 0.443422 - AUC: 0.808198
2021-09-01 11:16:58,119 P35557 INFO Monitor(max) STOP: 0.364776 !
2021-09-01 11:16:58,119 P35557 INFO Reduce learning rate on plateau: 0.000001
2021-09-01 11:16:58,120 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 11:16:58,163 P35557 INFO Train loss: 0.440683
2021-09-01 11:16:58,163 P35557 INFO ************ Epoch=37 end ************
2021-09-01 11:30:45,610 P35557 INFO [Metrics] logloss: 0.443421 - AUC: 0.808198
2021-09-01 11:30:45,612 P35557 INFO Save best model: monitor(max): 0.364777
2021-09-01 11:30:45,703 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 11:30:45,747 P35557 INFO Train loss: 0.440680
2021-09-01 11:30:45,747 P35557 INFO ************ Epoch=38 end ************
2021-09-01 11:44:32,575 P35557 INFO [Metrics] logloss: 0.443421 - AUC: 0.808198
2021-09-01 11:44:32,576 P35557 INFO Monitor(max) STOP: 0.364777 !
2021-09-01 11:44:32,576 P35557 INFO Reduce learning rate on plateau: 0.000001
2021-09-01 11:44:32,576 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 11:44:32,636 P35557 INFO Train loss: 0.440678
2021-09-01 11:44:32,636 P35557 INFO ************ Epoch=39 end ************
2021-09-01 11:58:20,453 P35557 INFO [Metrics] logloss: 0.443421 - AUC: 0.808199
2021-09-01 11:58:20,454 P35557 INFO Monitor(max) STOP: 0.364777 !
2021-09-01 11:58:20,455 P35557 INFO Reduce learning rate on plateau: 0.000001
2021-09-01 11:58:20,455 P35557 INFO Early stopping at epoch=40
2021-09-01 11:58:20,455 P35557 INFO --- 3668/3668 batches finished ---
2021-09-01 11:58:20,515 P35557 INFO Train loss: 0.440674
2021-09-01 11:58:20,515 P35557 INFO Training finished.
2021-09-01 11:58:20,515 P35557 INFO Load best model: /home/xxx/FuxiCTR/benchmarks/Criteo/FM_criteo/min10/criteo_x4_9ea3bdfc/FM_criteo_x4_003_3da0082a_model.ckpt
2021-09-01 11:58:20,612 P35557 INFO ****** Train/validation evaluation ******
2021-09-01 11:58:46,575 P35557 INFO [Metrics] logloss: 0.443421 - AUC: 0.808198
2021-09-01 11:58:46,610 P35557 INFO ******** Test evaluation ********
2021-09-01 11:58:46,611 P35557 INFO Loading data...
2021-09-01 11:58:46,611 P35557 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/test.h5
2021-09-01 11:58:47,445 P35557 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-01 11:58:47,445 P35557 INFO Loading test data done.
2021-09-01 11:59:13,652 P35557 INFO [Metrics] logloss: 0.443109 - AUC: 0.808607

```
