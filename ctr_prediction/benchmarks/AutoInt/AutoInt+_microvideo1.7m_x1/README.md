## AutoInt+_microvideo1.7m_x0_001

A notebook to benchmark AutoInt+ on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.734608 - logloss: 0.409701
```


### Logs
```python
2021-09-07 03:20:34,468 P32989 INFO {
    "attention_dim": "64",
    "attention_layers": "3",
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
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "layer_norm": "True",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_microvideo_1.7m_x0_008_411cc866",
    "model_root": "./MicroVideo/AutoInt_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_heads": "2",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "train_data": "../data/MicroVideo/MicroVideo_1.7M_x0/train.csv",
    "use_residual": "False",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/MicroVideo/MicroVideo_1.7M_x0/test.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-09-07 03:20:34,469 P32989 INFO Set up feature encoder...
2021-09-07 03:20:34,469 P32989 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-09-07 03:20:48,074 P32989 INFO Total number of parameters: 5283604.
2021-09-07 03:20:48,074 P32989 INFO Loading data...
2021-09-07 03:20:48,078 P32989 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-09-07 03:21:32,510 P32989 INFO Train samples: total/8970309, blocks/1
2021-09-07 03:21:32,510 P32989 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-09-07 03:21:40,680 P32989 INFO Validation samples: total/3767308, blocks/1%
2021-09-07 03:21:40,681 P32989 INFO Loading train data done.
2021-09-07 03:21:40,681 P32989 INFO Start training: 4381 batches/epoch
2021-09-07 03:21:40,681 P32989 INFO ************ Epoch=1 start ************
2021-09-07 03:27:05,135 P32989 INFO [Metrics] AUC: 0.716367 - logloss: 0.417249
2021-09-07 03:27:05,138 P32989 INFO Save best model: monitor(max): 0.716367
2021-09-07 03:27:06,061 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 03:27:06,503 P32989 INFO Train loss: 0.471867
2021-09-07 03:27:06,503 P32989 INFO ************ Epoch=1 end ************
2021-09-07 03:32:38,575 P32989 INFO [Metrics] AUC: 0.720469 - logloss: 0.415672
2021-09-07 03:32:38,578 P32989 INFO Save best model: monitor(max): 0.720469
2021-09-07 03:32:41,450 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 03:32:41,841 P32989 INFO Train loss: 0.444593
2021-09-07 03:32:41,842 P32989 INFO ************ Epoch=2 end ************
2021-09-07 03:38:07,581 P32989 INFO [Metrics] AUC: 0.720973 - logloss: 0.414239
2021-09-07 03:38:07,585 P32989 INFO Save best model: monitor(max): 0.720973
2021-09-07 03:38:10,599 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 03:38:10,980 P32989 INFO Train loss: 0.442369
2021-09-07 03:38:10,980 P32989 INFO ************ Epoch=3 end ************
2021-09-07 03:43:38,757 P32989 INFO [Metrics] AUC: 0.722307 - logloss: 0.414298
2021-09-07 03:43:38,760 P32989 INFO Save best model: monitor(max): 0.722307
2021-09-07 03:43:41,669 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 03:43:42,079 P32989 INFO Train loss: 0.441467
2021-09-07 03:43:42,080 P32989 INFO ************ Epoch=4 end ************
2021-09-07 03:49:07,177 P32989 INFO [Metrics] AUC: 0.722490 - logloss: 0.413857
2021-09-07 03:49:07,180 P32989 INFO Save best model: monitor(max): 0.722490
2021-09-07 03:49:10,204 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 03:49:10,564 P32989 INFO Train loss: 0.440904
2021-09-07 03:49:10,564 P32989 INFO ************ Epoch=5 end ************
2021-09-07 03:54:32,001 P32989 INFO [Metrics] AUC: 0.723190 - logloss: 0.413645
2021-09-07 03:54:32,005 P32989 INFO Save best model: monitor(max): 0.723190
2021-09-07 03:54:34,891 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 03:54:35,357 P32989 INFO Train loss: 0.440334
2021-09-07 03:54:35,357 P32989 INFO ************ Epoch=6 end ************
2021-09-07 04:00:00,064 P32989 INFO [Metrics] AUC: 0.723859 - logloss: 0.413238
2021-09-07 04:00:00,067 P32989 INFO Save best model: monitor(max): 0.723859
2021-09-07 04:00:02,969 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:00:03,324 P32989 INFO Train loss: 0.440041
2021-09-07 04:00:03,324 P32989 INFO ************ Epoch=7 end ************
2021-09-07 04:05:26,069 P32989 INFO [Metrics] AUC: 0.724187 - logloss: 0.413750
2021-09-07 04:05:26,073 P32989 INFO Save best model: monitor(max): 0.724187
2021-09-07 04:05:29,121 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:05:29,618 P32989 INFO Train loss: 0.439824
2021-09-07 04:05:29,618 P32989 INFO ************ Epoch=8 end ************
2021-09-07 04:10:50,383 P32989 INFO [Metrics] AUC: 0.724763 - logloss: 0.412366
2021-09-07 04:10:50,386 P32989 INFO Save best model: monitor(max): 0.724763
2021-09-07 04:10:53,294 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:10:53,658 P32989 INFO Train loss: 0.439509
2021-09-07 04:10:53,659 P32989 INFO ************ Epoch=9 end ************
2021-09-07 04:16:15,208 P32989 INFO [Metrics] AUC: 0.724270 - logloss: 0.412309
2021-09-07 04:16:15,211 P32989 INFO Monitor(max) STOP: 0.724270 !
2021-09-07 04:16:15,211 P32989 INFO Reduce learning rate on plateau: 0.000050
2021-09-07 04:16:15,212 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:16:15,707 P32989 INFO Train loss: 0.439176
2021-09-07 04:16:15,708 P32989 INFO ************ Epoch=10 end ************
2021-09-07 04:21:38,535 P32989 INFO [Metrics] AUC: 0.732650 - logloss: 0.409703
2021-09-07 04:21:38,539 P32989 INFO Save best model: monitor(max): 0.732650
2021-09-07 04:21:41,479 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:21:41,970 P32989 INFO Train loss: 0.426103
2021-09-07 04:21:41,970 P32989 INFO ************ Epoch=11 end ************
2021-09-07 04:27:04,474 P32989 INFO [Metrics] AUC: 0.733537 - logloss: 0.409347
2021-09-07 04:27:04,477 P32989 INFO Save best model: monitor(max): 0.733537
2021-09-07 04:27:07,404 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:27:07,795 P32989 INFO Train loss: 0.421287
2021-09-07 04:27:07,795 P32989 INFO ************ Epoch=12 end ************
2021-09-07 04:32:30,381 P32989 INFO [Metrics] AUC: 0.734011 - logloss: 0.409351
2021-09-07 04:32:30,384 P32989 INFO Save best model: monitor(max): 0.734011
2021-09-07 04:32:33,409 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:32:33,887 P32989 INFO Train loss: 0.419691
2021-09-07 04:32:33,887 P32989 INFO ************ Epoch=13 end ************
2021-09-07 04:37:49,965 P32989 INFO [Metrics] AUC: 0.733968 - logloss: 0.409090
2021-09-07 04:37:49,971 P32989 INFO Monitor(max) STOP: 0.733968 !
2021-09-07 04:37:49,971 P32989 INFO Reduce learning rate on plateau: 0.000005
2021-09-07 04:37:49,971 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:37:50,374 P32989 INFO Train loss: 0.418611
2021-09-07 04:37:50,374 P32989 INFO ************ Epoch=14 end ************
2021-09-07 04:42:38,457 P32989 INFO [Metrics] AUC: 0.734462 - logloss: 0.409626
2021-09-07 04:42:38,461 P32989 INFO Save best model: monitor(max): 0.734462
2021-09-07 04:42:41,432 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:42:41,880 P32989 INFO Train loss: 0.415022
2021-09-07 04:42:41,881 P32989 INFO ************ Epoch=15 end ************
2021-09-07 04:48:00,489 P32989 INFO [Metrics] AUC: 0.734506 - logloss: 0.409605
2021-09-07 04:48:00,492 P32989 INFO Save best model: monitor(max): 0.734506
2021-09-07 04:48:01,751 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:48:02,184 P32989 INFO Train loss: 0.414428
2021-09-07 04:48:02,185 P32989 INFO ************ Epoch=16 end ************
2021-09-07 04:53:21,449 P32989 INFO [Metrics] AUC: 0.734542 - logloss: 0.409629
2021-09-07 04:53:21,453 P32989 INFO Save best model: monitor(max): 0.734542
2021-09-07 04:53:24,520 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:53:24,957 P32989 INFO Train loss: 0.414044
2021-09-07 04:53:24,957 P32989 INFO ************ Epoch=17 end ************
2021-09-07 04:58:44,706 P32989 INFO [Metrics] AUC: 0.734608 - logloss: 0.409701
2021-09-07 04:58:44,711 P32989 INFO Save best model: monitor(max): 0.734608
2021-09-07 04:58:47,774 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 04:58:48,210 P32989 INFO Train loss: 0.413834
2021-09-07 04:58:48,210 P32989 INFO ************ Epoch=18 end ************
2021-09-07 05:04:07,346 P32989 INFO [Metrics] AUC: 0.734562 - logloss: 0.409634
2021-09-07 05:04:07,350 P32989 INFO Monitor(max) STOP: 0.734562 !
2021-09-07 05:04:07,350 P32989 INFO Reduce learning rate on plateau: 0.000001
2021-09-07 05:04:07,350 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 05:04:07,687 P32989 INFO Train loss: 0.413514
2021-09-07 05:04:07,698 P32989 INFO ************ Epoch=19 end ************
2021-09-07 05:09:26,097 P32989 INFO [Metrics] AUC: 0.734567 - logloss: 0.409883
2021-09-07 05:09:26,100 P32989 INFO Monitor(max) STOP: 0.734567 !
2021-09-07 05:09:26,100 P32989 INFO Reduce learning rate on plateau: 0.000001
2021-09-07 05:09:26,100 P32989 INFO Early stopping at epoch=20
2021-09-07 05:09:26,100 P32989 INFO --- 4381/4381 batches finished ---
2021-09-07 05:09:26,539 P32989 INFO Train loss: 0.412976
2021-09-07 05:09:26,539 P32989 INFO Training finished.
2021-09-07 05:09:26,539 P32989 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/AutoInt_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/AutoInt_microvideo_1.7m_x0_008_411cc866.model
2021-09-07 05:09:27,215 P32989 INFO ****** Train/validation evaluation ******
2021-09-07 05:10:06,488 P32989 INFO [Metrics] AUC: 0.734608 - logloss: 0.409701
2021-09-07 05:10:06,549 P32989 INFO ******** Test evaluation ********
2021-09-07 05:10:06,550 P32989 INFO Loading data...
2021-09-07 05:10:06,550 P32989 INFO Loading test data done.


```
