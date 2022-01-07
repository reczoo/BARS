## DCN_microvideo1.7m_x0_001

A notebook to benchmark DCN on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.734595 - logloss: 0.411168
```


### Logs
```python
2021-08-13 20:31:19,195 P1493 INFO {
    "batch_norm": "False",
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'max_len': 128, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_hook': 'layers.MaskedAveragePooling()', 'max_len': 128, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_microvideo_1.7m_x0_014_9d3f412e",
    "model_root": "./MicroVideo/DCN_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_cross_layers": "2",
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
2021-08-13 20:31:19,196 P1493 INFO Set up feature encoder...
2021-08-13 20:31:19,196 P1493 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-08-13 20:31:33,276 P1493 INFO Total number of parameters: 1730945.
2021-08-13 20:31:33,276 P1493 INFO Loading data...
2021-08-13 20:31:33,280 P1493 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-08-13 20:31:46,566 P1493 INFO Train samples: total/8970309, blocks/1
2021-08-13 20:31:46,566 P1493 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-08-13 20:31:51,952 P1493 INFO Validation samples: total/3767308, blocks/1%
2021-08-13 20:31:51,953 P1493 INFO Loading train data done.
2021-08-13 20:31:51,953 P1493 INFO Start training: 4381 batches/epoch
2021-08-13 20:31:51,953 P1493 INFO ************ Epoch=1 start ************
2021-08-13 20:36:19,189 P1493 INFO [Metrics] AUC: 0.715385 - logloss: 0.416944
2021-08-13 20:36:19,190 P1493 INFO Save best model: monitor(max): 0.715385
2021-08-13 20:36:20,062 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 20:36:20,187 P1493 INFO Train loss: 0.464376
2021-08-13 20:36:20,187 P1493 INFO ************ Epoch=1 end ************
2021-08-13 20:40:59,972 P1493 INFO [Metrics] AUC: 0.720317 - logloss: 0.414769
2021-08-13 20:40:59,973 P1493 INFO Save best model: monitor(max): 0.720317
2021-08-13 20:41:01,084 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 20:41:01,163 P1493 INFO Train loss: 0.443378
2021-08-13 20:41:01,163 P1493 INFO ************ Epoch=2 end ************
2021-08-13 20:45:25,167 P1493 INFO [Metrics] AUC: 0.723148 - logloss: 0.416029
2021-08-13 20:45:25,171 P1493 INFO Save best model: monitor(max): 0.723148
2021-08-13 20:45:27,121 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 20:45:27,216 P1493 INFO Train loss: 0.441301
2021-08-13 20:45:27,216 P1493 INFO ************ Epoch=3 end ************
2021-08-13 20:49:50,122 P1493 INFO [Metrics] AUC: 0.724072 - logloss: 0.415302
2021-08-13 20:49:50,123 P1493 INFO Save best model: monitor(max): 0.724072
2021-08-13 20:49:52,068 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 20:49:52,173 P1493 INFO Train loss: 0.440569
2021-08-13 20:49:52,173 P1493 INFO ************ Epoch=4 end ************
2021-08-13 20:54:34,929 P1493 INFO [Metrics] AUC: 0.724887 - logloss: 0.411855
2021-08-13 20:54:34,932 P1493 INFO Save best model: monitor(max): 0.724887
2021-08-13 20:54:36,917 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 20:54:37,032 P1493 INFO Train loss: 0.439905
2021-08-13 20:54:37,032 P1493 INFO ************ Epoch=5 end ************
2021-08-13 20:59:16,519 P1493 INFO [Metrics] AUC: 0.725351 - logloss: 0.414734
2021-08-13 20:59:16,520 P1493 INFO Save best model: monitor(max): 0.725351
2021-08-13 20:59:17,616 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 20:59:17,734 P1493 INFO Train loss: 0.439465
2021-08-13 20:59:17,734 P1493 INFO ************ Epoch=6 end ************
2021-08-13 21:03:43,810 P1493 INFO [Metrics] AUC: 0.725629 - logloss: 0.413237
2021-08-13 21:03:43,811 P1493 INFO Save best model: monitor(max): 0.725629
2021-08-13 21:03:45,722 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:03:45,835 P1493 INFO Train loss: 0.438957
2021-08-13 21:03:45,835 P1493 INFO ************ Epoch=7 end ************
2021-08-13 21:07:57,791 P1493 INFO [Metrics] AUC: 0.726077 - logloss: 0.412300
2021-08-13 21:07:57,795 P1493 INFO Save best model: monitor(max): 0.726077
2021-08-13 21:07:59,738 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:07:59,843 P1493 INFO Train loss: 0.438534
2021-08-13 21:07:59,843 P1493 INFO ************ Epoch=8 end ************
2021-08-13 21:12:25,938 P1493 INFO [Metrics] AUC: 0.726182 - logloss: 0.412512
2021-08-13 21:12:25,941 P1493 INFO Save best model: monitor(max): 0.726182
2021-08-13 21:12:27,830 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:12:27,920 P1493 INFO Train loss: 0.438177
2021-08-13 21:12:27,921 P1493 INFO ************ Epoch=9 end ************
2021-08-13 21:16:35,110 P1493 INFO [Metrics] AUC: 0.726926 - logloss: 0.412193
2021-08-13 21:16:35,112 P1493 INFO Save best model: monitor(max): 0.726926
2021-08-13 21:16:36,997 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:16:37,127 P1493 INFO Train loss: 0.437952
2021-08-13 21:16:37,127 P1493 INFO ************ Epoch=10 end ************
2021-08-13 21:20:23,631 P1493 INFO [Metrics] AUC: 0.725981 - logloss: 0.413451
2021-08-13 21:20:23,633 P1493 INFO Monitor(max) STOP: 0.725981 !
2021-08-13 21:20:23,633 P1493 INFO Reduce learning rate on plateau: 0.000050
2021-08-13 21:20:23,633 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:20:23,743 P1493 INFO Train loss: 0.437522
2021-08-13 21:20:23,743 P1493 INFO ************ Epoch=11 end ************
2021-08-13 21:24:05,696 P1493 INFO [Metrics] AUC: 0.732687 - logloss: 0.410334
2021-08-13 21:24:05,699 P1493 INFO Save best model: monitor(max): 0.732687
2021-08-13 21:24:07,624 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:24:07,710 P1493 INFO Train loss: 0.426336
2021-08-13 21:24:07,710 P1493 INFO ************ Epoch=12 end ************
2021-08-13 21:27:49,532 P1493 INFO [Metrics] AUC: 0.733566 - logloss: 0.409982
2021-08-13 21:27:49,535 P1493 INFO Save best model: monitor(max): 0.733566
2021-08-13 21:27:51,467 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:27:51,570 P1493 INFO Train loss: 0.421996
2021-08-13 21:27:51,570 P1493 INFO ************ Epoch=13 end ************
2021-08-13 21:31:40,564 P1493 INFO [Metrics] AUC: 0.733942 - logloss: 0.410315
2021-08-13 21:31:40,565 P1493 INFO Save best model: monitor(max): 0.733942
2021-08-13 21:31:42,492 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:31:42,609 P1493 INFO Train loss: 0.420561
2021-08-13 21:31:42,609 P1493 INFO ************ Epoch=14 end ************
2021-08-13 21:35:43,155 P1493 INFO [Metrics] AUC: 0.734144 - logloss: 0.410237
2021-08-13 21:35:43,156 P1493 INFO Save best model: monitor(max): 0.734144
2021-08-13 21:35:45,177 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:35:45,287 P1493 INFO Train loss: 0.419510
2021-08-13 21:35:45,287 P1493 INFO ************ Epoch=15 end ************
2021-08-13 21:39:45,678 P1493 INFO [Metrics] AUC: 0.734389 - logloss: 0.410315
2021-08-13 21:39:45,681 P1493 INFO Save best model: monitor(max): 0.734389
2021-08-13 21:39:46,796 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:39:46,907 P1493 INFO Train loss: 0.418596
2021-08-13 21:39:46,907 P1493 INFO ************ Epoch=16 end ************
2021-08-13 21:43:45,708 P1493 INFO [Metrics] AUC: 0.734062 - logloss: 0.410191
2021-08-13 21:43:45,710 P1493 INFO Monitor(max) STOP: 0.734062 !
2021-08-13 21:43:45,710 P1493 INFO Reduce learning rate on plateau: 0.000005
2021-08-13 21:43:45,710 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:43:45,815 P1493 INFO Train loss: 0.417813
2021-08-13 21:43:45,815 P1493 INFO ************ Epoch=17 end ************
2021-08-13 21:47:44,050 P1493 INFO [Metrics] AUC: 0.734500 - logloss: 0.410710
2021-08-13 21:47:44,051 P1493 INFO Save best model: monitor(max): 0.734500
2021-08-13 21:47:46,028 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:47:46,169 P1493 INFO Train loss: 0.414009
2021-08-13 21:47:46,170 P1493 INFO ************ Epoch=18 end ************
2021-08-13 21:51:44,131 P1493 INFO [Metrics] AUC: 0.734563 - logloss: 0.410864
2021-08-13 21:51:44,132 P1493 INFO Save best model: monitor(max): 0.734563
2021-08-13 21:51:46,121 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:51:46,242 P1493 INFO Train loss: 0.413474
2021-08-13 21:51:46,242 P1493 INFO ************ Epoch=19 end ************
2021-08-13 21:55:43,655 P1493 INFO [Metrics] AUC: 0.734588 - logloss: 0.410996
2021-08-13 21:55:43,656 P1493 INFO Save best model: monitor(max): 0.734588
2021-08-13 21:55:45,638 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:55:45,746 P1493 INFO Train loss: 0.413168
2021-08-13 21:55:45,746 P1493 INFO ************ Epoch=20 end ************
2021-08-13 21:59:42,529 P1493 INFO [Metrics] AUC: 0.734559 - logloss: 0.410945
2021-08-13 21:59:42,533 P1493 INFO Monitor(max) STOP: 0.734559 !
2021-08-13 21:59:42,533 P1493 INFO Reduce learning rate on plateau: 0.000001
2021-08-13 21:59:42,534 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 21:59:42,633 P1493 INFO Train loss: 0.412799
2021-08-13 21:59:42,633 P1493 INFO ************ Epoch=21 end ************
2021-08-13 22:03:38,240 P1493 INFO [Metrics] AUC: 0.734592 - logloss: 0.411136
2021-08-13 22:03:38,244 P1493 INFO Save best model: monitor(max): 0.734592
2021-08-13 22:03:40,224 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 22:03:40,322 P1493 INFO Train loss: 0.412119
2021-08-13 22:03:40,322 P1493 INFO ************ Epoch=22 end ************
2021-08-13 22:07:39,813 P1493 INFO [Metrics] AUC: 0.734595 - logloss: 0.411168
2021-08-13 22:07:39,814 P1493 INFO Save best model: monitor(max): 0.734595
2021-08-13 22:07:41,838 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 22:07:41,951 P1493 INFO Train loss: 0.412170
2021-08-13 22:07:41,952 P1493 INFO ************ Epoch=23 end ************
2021-08-13 22:11:38,503 P1493 INFO [Metrics] AUC: 0.734586 - logloss: 0.411163
2021-08-13 22:11:38,506 P1493 INFO Monitor(max) STOP: 0.734586 !
2021-08-13 22:11:38,507 P1493 INFO Reduce learning rate on plateau: 0.000001
2021-08-13 22:11:38,507 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 22:11:38,620 P1493 INFO Train loss: 0.412116
2021-08-13 22:11:38,621 P1493 INFO ************ Epoch=24 end ************
2021-08-13 22:15:35,293 P1493 INFO [Metrics] AUC: 0.734595 - logloss: 0.411163
2021-08-13 22:15:35,294 P1493 INFO Monitor(max) STOP: 0.734595 !
2021-08-13 22:15:35,294 P1493 INFO Reduce learning rate on plateau: 0.000001
2021-08-13 22:15:35,294 P1493 INFO Early stopping at epoch=25
2021-08-13 22:15:35,294 P1493 INFO --- 4381/4381 batches finished ---
2021-08-13 22:15:35,404 P1493 INFO Train loss: 0.412059
2021-08-13 22:15:35,405 P1493 INFO Training finished.
2021-08-13 22:15:35,405 P1493 INFO Load best model: /home/zhujieming/zhujieming/GroupCTR/benchmark/MicroVideo/DCN_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/DCN_microvideo_1.7m_x0_014_9d3f412e.model
2021-08-13 22:15:40,030 P1493 INFO ****** Train/validation evaluation ******
2021-08-13 22:16:12,923 P1493 INFO [Metrics] AUC: 0.734595 - logloss: 0.411168
2021-08-13 22:16:12,979 P1493 INFO ******** Test evaluation ********
2021-08-13 22:16:12,979 P1493 INFO Loading data...
2021-08-13 22:16:12,980 P1493 INFO Loading test data done.


```
