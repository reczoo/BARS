## YoutubeDNN_microvideo1.7m_x0_001

A notebook to benchmark YoutubeDNN on microvideo1.7m_x0_001 dataset.

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


### Code




### Results
```python
[Metrics] AUC: 0.734471 - logloss: 0.411108
```


### Logs
```python
2021-08-14 01:34:15,133 P48405 INFO {
    "batch_norm": "False",
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'max_len': 128, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_hook': 'layers.MaskedAveragePooling()', 'max_len': 128, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_microvideo_1.7m_x0_003_1a09d614",
    "model_root": "./MicroVideo/DNN_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "train_data": "../data/MicroVideo/MicroVideo_1.7M_x0/train.csv",
    "valid_data": "../data/MicroVideo/MicroVideo_1.7M_x0/test.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-08-14 01:34:15,134 P48405 INFO Set up feature encoder...
2021-08-14 01:34:15,134 P48405 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-08-14 01:34:29,375 P48405 INFO Total number of parameters: 1729345.
2021-08-14 01:34:29,375 P48405 INFO Loading data...
2021-08-14 01:34:29,378 P48405 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-08-14 01:34:44,019 P48405 INFO Train samples: total/8970309, blocks/1
2021-08-14 01:34:44,019 P48405 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-08-14 01:34:50,223 P48405 INFO Validation samples: total/3767308, blocks/1%
2021-08-14 01:34:50,223 P48405 INFO Loading train data done.
2021-08-14 01:34:50,223 P48405 INFO Start training: 4381 batches/epoch
2021-08-14 01:34:50,223 P48405 INFO ************ Epoch=1 start ************
2021-08-14 01:37:21,967 P48405 INFO [Metrics] AUC: 0.716196 - logloss: 0.417633
2021-08-14 01:37:21,971 P48405 INFO Save best model: monitor(max): 0.716196
2021-08-14 01:37:23,171 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:37:23,301 P48405 INFO Train loss: 0.462369
2021-08-14 01:37:23,301 P48405 INFO ************ Epoch=1 end ************
2021-08-14 01:39:57,284 P48405 INFO [Metrics] AUC: 0.720793 - logloss: 0.415678
2021-08-14 01:39:57,288 P48405 INFO Save best model: monitor(max): 0.720793
2021-08-14 01:40:00,491 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:40:00,680 P48405 INFO Train loss: 0.445457
2021-08-14 01:40:00,680 P48405 INFO ************ Epoch=2 end ************
2021-08-14 01:42:31,958 P48405 INFO [Metrics] AUC: 0.723026 - logloss: 0.414577
2021-08-14 01:42:31,961 P48405 INFO Save best model: monitor(max): 0.723026
2021-08-14 01:42:33,976 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:42:34,079 P48405 INFO Train loss: 0.442008
2021-08-14 01:42:34,079 P48405 INFO ************ Epoch=3 end ************
2021-08-14 01:45:06,061 P48405 INFO [Metrics] AUC: 0.722908 - logloss: 0.414798
2021-08-14 01:45:06,065 P48405 INFO Monitor(max) STOP: 0.722908 !
2021-08-14 01:45:06,066 P48405 INFO Reduce learning rate on plateau: 0.000050
2021-08-14 01:45:06,066 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:45:06,225 P48405 INFO Train loss: 0.440981
2021-08-14 01:45:06,225 P48405 INFO ************ Epoch=4 end ************
2021-08-14 01:47:35,127 P48405 INFO [Metrics] AUC: 0.731218 - logloss: 0.410184
2021-08-14 01:47:35,130 P48405 INFO Save best model: monitor(max): 0.731218
2021-08-14 01:47:37,135 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:47:37,231 P48405 INFO Train loss: 0.430430
2021-08-14 01:47:37,231 P48405 INFO ************ Epoch=5 end ************
2021-08-14 01:50:06,043 P48405 INFO [Metrics] AUC: 0.732409 - logloss: 0.409899
2021-08-14 01:50:06,046 P48405 INFO Save best model: monitor(max): 0.732409
2021-08-14 01:50:08,027 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:50:08,134 P48405 INFO Train loss: 0.426491
2021-08-14 01:50:08,135 P48405 INFO ************ Epoch=6 end ************
2021-08-14 01:52:35,398 P48405 INFO [Metrics] AUC: 0.732980 - logloss: 0.409914
2021-08-14 01:52:35,401 P48405 INFO Save best model: monitor(max): 0.732980
2021-08-14 01:52:37,388 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:52:37,489 P48405 INFO Train loss: 0.424760
2021-08-14 01:52:37,489 P48405 INFO ************ Epoch=7 end ************
2021-08-14 01:55:04,452 P48405 INFO [Metrics] AUC: 0.733458 - logloss: 0.409894
2021-08-14 01:55:04,453 P48405 INFO Save best model: monitor(max): 0.733458
2021-08-14 01:55:06,479 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:55:06,583 P48405 INFO Train loss: 0.423448
2021-08-14 01:55:06,584 P48405 INFO ************ Epoch=8 end ************
2021-08-14 01:57:33,403 P48405 INFO [Metrics] AUC: 0.733638 - logloss: 0.410155
2021-08-14 01:57:33,406 P48405 INFO Save best model: monitor(max): 0.733638
2021-08-14 01:57:35,413 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 01:57:35,511 P48405 INFO Train loss: 0.422328
2021-08-14 01:57:35,511 P48405 INFO ************ Epoch=9 end ************
2021-08-14 02:00:01,897 P48405 INFO [Metrics] AUC: 0.733822 - logloss: 0.410096
2021-08-14 02:00:01,899 P48405 INFO Save best model: monitor(max): 0.733822
2021-08-14 02:00:03,892 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 02:00:03,993 P48405 INFO Train loss: 0.421348
2021-08-14 02:00:03,993 P48405 INFO ************ Epoch=10 end ************
2021-08-14 02:02:31,092 P48405 INFO [Metrics] AUC: 0.734049 - logloss: 0.410091
2021-08-14 02:02:31,094 P48405 INFO Save best model: monitor(max): 0.734049
2021-08-14 02:02:33,108 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 02:02:33,244 P48405 INFO Train loss: 0.420513
2021-08-14 02:02:33,244 P48405 INFO ************ Epoch=11 end ************
2021-08-14 02:04:59,364 P48405 INFO [Metrics] AUC: 0.733905 - logloss: 0.410497
2021-08-14 02:04:59,367 P48405 INFO Monitor(max) STOP: 0.733905 !
2021-08-14 02:04:59,367 P48405 INFO Reduce learning rate on plateau: 0.000005
2021-08-14 02:04:59,367 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 02:04:59,500 P48405 INFO Train loss: 0.419760
2021-08-14 02:04:59,500 P48405 INFO ************ Epoch=12 end ************
2021-08-14 02:07:24,479 P48405 INFO [Metrics] AUC: 0.734471 - logloss: 0.411108
2021-08-14 02:07:24,482 P48405 INFO Save best model: monitor(max): 0.734471
2021-08-14 02:07:26,475 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 02:07:26,576 P48405 INFO Train loss: 0.414679
2021-08-14 02:07:26,576 P48405 INFO ************ Epoch=13 end ************
2021-08-14 02:09:54,002 P48405 INFO [Metrics] AUC: 0.734339 - logloss: 0.411239
2021-08-14 02:09:54,003 P48405 INFO Monitor(max) STOP: 0.734339 !
2021-08-14 02:09:54,003 P48405 INFO Reduce learning rate on plateau: 0.000001
2021-08-14 02:09:54,004 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 02:09:54,116 P48405 INFO Train loss: 0.413916
2021-08-14 02:09:54,117 P48405 INFO ************ Epoch=14 end ************
2021-08-14 02:12:17,804 P48405 INFO [Metrics] AUC: 0.734366 - logloss: 0.411510
2021-08-14 02:12:17,806 P48405 INFO Monitor(max) STOP: 0.734366 !
2021-08-14 02:12:17,806 P48405 INFO Reduce learning rate on plateau: 0.000001
2021-08-14 02:12:17,806 P48405 INFO Early stopping at epoch=15
2021-08-14 02:12:17,807 P48405 INFO --- 4381/4381 batches finished ---
2021-08-14 02:12:17,934 P48405 INFO Train loss: 0.412943
2021-08-14 02:12:17,934 P48405 INFO Training finished.
2021-08-14 02:12:17,934 P48405 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/DNN_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/DNN_microvideo_1.7m_x0_003_1a09d614.model
2021-08-14 02:12:21,143 P48405 INFO ****** Train/validation evaluation ******
2021-08-14 02:12:41,381 P48405 INFO [Metrics] AUC: 0.734471 - logloss: 0.411108
2021-08-14 02:12:41,504 P48405 INFO ******** Test evaluation ********
2021-08-14 02:12:41,505 P48405 INFO Loading data...
2021-08-14 02:12:41,505 P48405 INFO Loading test data done.


```
