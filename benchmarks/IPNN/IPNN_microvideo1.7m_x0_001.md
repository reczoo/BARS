## IPNN_microvideo1.7m_x0_001

A notebook to benchmark IPNN on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.736358 - logloss: 0.411003
```


### Logs
```python
2021-09-02 14:22:50,374 P44352 INFO {
    "batch_norm": "True",
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
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "PNN",
    "model_id": "PNN_microvideo_1.7m_x0_004_13460e56",
    "model_root": "./Microvideo/IPNN_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
2021-09-02 14:22:50,375 P44352 INFO Set up feature encoder...
2021-09-02 14:22:50,376 P44352 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-09-02 14:23:02,847 P44352 INFO Total number of parameters: 1743169.
2021-09-02 14:23:02,847 P44352 INFO Loading data...
2021-09-02 14:23:02,851 P44352 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-09-02 14:23:22,701 P44352 INFO Train samples: total/8970309, blocks/1
2021-09-02 14:23:22,702 P44352 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-09-02 14:23:40,160 P44352 INFO Validation samples: total/3767308, blocks/1%
2021-09-02 14:23:40,160 P44352 INFO Loading train data done.
2021-09-02 14:23:40,161 P44352 INFO Start training: 4381 batches/epoch
2021-09-02 14:23:40,161 P44352 INFO ************ Epoch=1 start ************
2021-09-02 14:25:57,435 P44352 INFO [Metrics] AUC: 0.720096 - logloss: 0.415901
2021-09-02 14:25:57,440 P44352 INFO Save best model: monitor(max): 0.720096
2021-09-02 14:26:00,158 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:26:00,464 P44352 INFO Train loss: 0.466890
2021-09-02 14:26:00,464 P44352 INFO ************ Epoch=1 end ************
2021-09-02 14:29:08,673 P44352 INFO [Metrics] AUC: 0.722293 - logloss: 0.415299
2021-09-02 14:29:08,680 P44352 INFO Save best model: monitor(max): 0.722293
2021-09-02 14:29:11,499 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:29:11,808 P44352 INFO Train loss: 0.448357
2021-09-02 14:29:11,809 P44352 INFO ************ Epoch=2 end ************
2021-09-02 14:32:17,801 P44352 INFO [Metrics] AUC: 0.724388 - logloss: 0.413539
2021-09-02 14:32:17,807 P44352 INFO Save best model: monitor(max): 0.724388
2021-09-02 14:32:21,998 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:32:22,341 P44352 INFO Train loss: 0.446341
2021-09-02 14:32:22,341 P44352 INFO ************ Epoch=3 end ************
2021-09-02 14:35:26,315 P44352 INFO [Metrics] AUC: 0.725121 - logloss: 0.413240
2021-09-02 14:35:26,320 P44352 INFO Save best model: monitor(max): 0.725121
2021-09-02 14:35:29,082 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:35:29,422 P44352 INFO Train loss: 0.445200
2021-09-02 14:35:29,422 P44352 INFO ************ Epoch=4 end ************
2021-09-02 14:38:34,857 P44352 INFO [Metrics] AUC: 0.725385 - logloss: 0.412549
2021-09-02 14:38:34,859 P44352 INFO Save best model: monitor(max): 0.725385
2021-09-02 14:38:37,412 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:38:37,764 P44352 INFO Train loss: 0.444394
2021-09-02 14:38:37,764 P44352 INFO ************ Epoch=5 end ************
2021-09-02 14:41:43,955 P44352 INFO [Metrics] AUC: 0.727150 - logloss: 0.412625
2021-09-02 14:41:43,960 P44352 INFO Save best model: monitor(max): 0.727150
2021-09-02 14:41:46,763 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:41:47,066 P44352 INFO Train loss: 0.443592
2021-09-02 14:41:47,066 P44352 INFO ************ Epoch=6 end ************
2021-09-02 14:44:52,469 P44352 INFO [Metrics] AUC: 0.727240 - logloss: 0.413300
2021-09-02 14:44:52,471 P44352 INFO Save best model: monitor(max): 0.727240
2021-09-02 14:44:55,191 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:44:55,512 P44352 INFO Train loss: 0.442989
2021-09-02 14:44:55,512 P44352 INFO ************ Epoch=7 end ************
2021-09-02 14:48:00,683 P44352 INFO [Metrics] AUC: 0.727718 - logloss: 0.412884
2021-09-02 14:48:00,685 P44352 INFO Save best model: monitor(max): 0.727718
2021-09-02 14:48:03,287 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:48:03,649 P44352 INFO Train loss: 0.442393
2021-09-02 14:48:03,649 P44352 INFO ************ Epoch=8 end ************
2021-09-02 14:51:08,909 P44352 INFO [Metrics] AUC: 0.728475 - logloss: 0.412448
2021-09-02 14:51:08,918 P44352 INFO Save best model: monitor(max): 0.728475
2021-09-02 14:51:11,610 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:51:11,956 P44352 INFO Train loss: 0.441873
2021-09-02 14:51:11,957 P44352 INFO ************ Epoch=9 end ************
2021-09-02 14:53:24,495 P44352 INFO [Metrics] AUC: 0.727596 - logloss: 0.412637
2021-09-02 14:53:24,498 P44352 INFO Monitor(max) STOP: 0.727596 !
2021-09-02 14:53:24,498 P44352 INFO Reduce learning rate on plateau: 0.000050
2021-09-02 14:53:24,498 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:53:24,927 P44352 INFO Train loss: 0.441222
2021-09-02 14:53:24,927 P44352 INFO ************ Epoch=10 end ************
2021-09-02 14:56:35,488 P44352 INFO [Metrics] AUC: 0.735363 - logloss: 0.409881
2021-09-02 14:56:35,495 P44352 INFO Save best model: monitor(max): 0.735363
2021-09-02 14:56:38,213 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:56:38,568 P44352 INFO Train loss: 0.425302
2021-09-02 14:56:38,568 P44352 INFO ************ Epoch=11 end ************
2021-09-02 14:59:44,943 P44352 INFO [Metrics] AUC: 0.736352 - logloss: 0.410919
2021-09-02 14:59:44,946 P44352 INFO Save best model: monitor(max): 0.736352
2021-09-02 14:59:48,638 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 14:59:49,007 P44352 INFO Train loss: 0.417439
2021-09-02 14:59:49,007 P44352 INFO ************ Epoch=12 end ************
2021-09-02 15:02:54,621 P44352 INFO [Metrics] AUC: 0.736358 - logloss: 0.411003
2021-09-02 15:02:54,626 P44352 INFO Save best model: monitor(max): 0.736358
2021-09-02 15:02:57,142 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 15:02:57,447 P44352 INFO Train loss: 0.413165
2021-09-02 15:02:57,448 P44352 INFO ************ Epoch=13 end ************
2021-09-02 15:06:06,176 P44352 INFO [Metrics] AUC: 0.735554 - logloss: 0.412800
2021-09-02 15:06:06,178 P44352 INFO Monitor(max) STOP: 0.735554 !
2021-09-02 15:06:06,179 P44352 INFO Reduce learning rate on plateau: 0.000005
2021-09-02 15:06:06,179 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 15:06:06,521 P44352 INFO Train loss: 0.409295
2021-09-02 15:06:06,521 P44352 INFO ************ Epoch=14 end ************
2021-09-02 15:09:11,494 P44352 INFO [Metrics] AUC: 0.733991 - logloss: 0.421472
2021-09-02 15:09:11,507 P44352 INFO Monitor(max) STOP: 0.733991 !
2021-09-02 15:09:11,507 P44352 INFO Reduce learning rate on plateau: 0.000001
2021-09-02 15:09:11,507 P44352 INFO Early stopping at epoch=15
2021-09-02 15:09:11,507 P44352 INFO --- 4381/4381 batches finished ---
2021-09-02 15:09:11,894 P44352 INFO Train loss: 0.396773
2021-09-02 15:09:11,895 P44352 INFO Training finished.
2021-09-02 15:09:11,895 P44352 INFO Load best model: /home/ma-user/work/GroupCTR/benchmark/Microvideo/IPNN_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/PNN_microvideo_1.7m_x0_004_13460e56.model
2021-09-02 15:09:15,137 P44352 INFO ****** Train/validation evaluation ******
2021-09-02 15:09:41,115 P44352 INFO [Metrics] AUC: 0.736358 - logloss: 0.411003
2021-09-02 15:09:41,176 P44352 INFO ******** Test evaluation ********
2021-09-02 15:09:41,177 P44352 INFO Loading data...
2021-09-02 15:09:41,177 P44352 INFO Loading test data done.


```
