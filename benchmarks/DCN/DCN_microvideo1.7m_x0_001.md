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
[Metrics] AUC: 0.734937 - logloss: 0.411225
```


### Logs
```python
2021-08-13 20:09:41,776 P37989 INFO {
    "batch_norm": "False",
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "dnn_activations": "relu",
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
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN_v2",
    "model_id": "DCN_v2_microvideo_1.7m_x0_010_af571c0e",
    "model_root": "./MicroVideo/DCN_v2_microvideo_1.7m_x0/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_cross_layers": "2",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[1024, 512, 256]",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "train_data": "../data/MicroVideo/MicroVideo_1.7M_x0/train.csv",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/MicroVideo/MicroVideo_1.7M_x0/test.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-08-13 20:09:41,776 P37989 INFO Set up feature encoder...
2021-08-13 20:09:41,777 P37989 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-08-13 20:09:55,632 P37989 INFO Total number of parameters: 1935105.
2021-08-13 20:09:55,632 P37989 INFO Loading data...
2021-08-13 20:09:55,636 P37989 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-08-13 20:10:09,068 P37989 INFO Train samples: total/8970309, blocks/1
2021-08-13 20:10:09,068 P37989 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-08-13 20:10:15,176 P37989 INFO Validation samples: total/3767308, blocks/1%
2021-08-13 20:10:15,177 P37989 INFO Loading train data done.
2021-08-13 20:10:15,177 P37989 INFO Start training: 4381 batches/epoch
2021-08-13 20:10:15,177 P37989 INFO ************ Epoch=1 start ************
2021-08-13 20:13:47,360 P37989 INFO [Metrics] AUC: 0.719425 - logloss: 0.416599
2021-08-13 20:13:47,363 P37989 INFO Save best model: monitor(max): 0.719425
2021-08-13 20:13:48,607 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:13:48,794 P37989 INFO Train loss: 0.458526
2021-08-13 20:13:48,794 P37989 INFO ************ Epoch=1 end ************
2021-08-13 20:17:35,223 P37989 INFO [Metrics] AUC: 0.721559 - logloss: 0.416963
2021-08-13 20:17:35,226 P37989 INFO Save best model: monitor(max): 0.721559
2021-08-13 20:17:37,231 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:17:37,334 P37989 INFO Train loss: 0.442605
2021-08-13 20:17:37,334 P37989 INFO ************ Epoch=2 end ************
2021-08-13 20:21:25,003 P37989 INFO [Metrics] AUC: 0.723947 - logloss: 0.414681
2021-08-13 20:21:25,008 P37989 INFO Save best model: monitor(max): 0.723947
2021-08-13 20:21:26,987 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:21:27,093 P37989 INFO Train loss: 0.440546
2021-08-13 20:21:27,093 P37989 INFO ************ Epoch=3 end ************
2021-08-13 20:25:13,217 P37989 INFO [Metrics] AUC: 0.724626 - logloss: 0.414490
2021-08-13 20:25:13,221 P37989 INFO Save best model: monitor(max): 0.724626
2021-08-13 20:25:15,132 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:25:15,234 P37989 INFO Train loss: 0.439284
2021-08-13 20:25:15,234 P37989 INFO ************ Epoch=4 end ************
2021-08-13 20:29:00,626 P37989 INFO [Metrics] AUC: 0.724971 - logloss: 0.413332
2021-08-13 20:29:00,631 P37989 INFO Save best model: monitor(max): 0.724971
2021-08-13 20:29:02,581 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:29:02,687 P37989 INFO Train loss: 0.438604
2021-08-13 20:29:02,688 P37989 INFO ************ Epoch=5 end ************
2021-08-13 20:32:50,093 P37989 INFO [Metrics] AUC: 0.725481 - logloss: 0.415153
2021-08-13 20:32:50,094 P37989 INFO Save best model: monitor(max): 0.725481
2021-08-13 20:32:52,001 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:32:52,112 P37989 INFO Train loss: 0.438049
2021-08-13 20:32:52,112 P37989 INFO ************ Epoch=6 end ************
2021-08-13 20:36:39,171 P37989 INFO [Metrics] AUC: 0.726270 - logloss: 0.413682
2021-08-13 20:36:39,172 P37989 INFO Save best model: monitor(max): 0.726270
2021-08-13 20:36:41,101 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:36:41,242 P37989 INFO Train loss: 0.437711
2021-08-13 20:36:41,242 P37989 INFO ************ Epoch=7 end ************
2021-08-13 20:40:28,103 P37989 INFO [Metrics] AUC: 0.727051 - logloss: 0.412382
2021-08-13 20:40:28,107 P37989 INFO Save best model: monitor(max): 0.727051
2021-08-13 20:40:30,099 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:40:30,235 P37989 INFO Train loss: 0.437378
2021-08-13 20:40:30,236 P37989 INFO ************ Epoch=8 end ************
2021-08-13 20:44:16,485 P37989 INFO [Metrics] AUC: 0.726992 - logloss: 0.412848
2021-08-13 20:44:16,486 P37989 INFO Monitor(max) STOP: 0.726992 !
2021-08-13 20:44:16,487 P37989 INFO Reduce learning rate on plateau: 0.000050
2021-08-13 20:44:16,487 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:44:16,629 P37989 INFO Train loss: 0.437072
2021-08-13 20:44:16,629 P37989 INFO ************ Epoch=9 end ************
2021-08-13 20:48:02,143 P37989 INFO [Metrics] AUC: 0.733538 - logloss: 0.409670
2021-08-13 20:48:02,148 P37989 INFO Save best model: monitor(max): 0.733538
2021-08-13 20:48:04,093 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:48:04,219 P37989 INFO Train loss: 0.425541
2021-08-13 20:48:04,219 P37989 INFO ************ Epoch=10 end ************
2021-08-13 20:51:50,377 P37989 INFO [Metrics] AUC: 0.734279 - logloss: 0.409596
2021-08-13 20:51:50,380 P37989 INFO Save best model: monitor(max): 0.734279
2021-08-13 20:51:52,326 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:51:52,432 P37989 INFO Train loss: 0.420866
2021-08-13 20:51:52,432 P37989 INFO ************ Epoch=11 end ************
2021-08-13 20:55:27,216 P37989 INFO [Metrics] AUC: 0.734561 - logloss: 0.410078
2021-08-13 20:55:27,219 P37989 INFO Save best model: monitor(max): 0.734561
2021-08-13 20:55:29,097 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:55:29,201 P37989 INFO Train loss: 0.419076
2021-08-13 20:55:29,201 P37989 INFO ************ Epoch=12 end ************
2021-08-13 20:59:44,930 P37989 INFO [Metrics] AUC: 0.734602 - logloss: 0.410261
2021-08-13 20:59:44,932 P37989 INFO Save best model: monitor(max): 0.734602
2021-08-13 20:59:46,823 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 20:59:46,958 P37989 INFO Train loss: 0.417827
2021-08-13 20:59:46,958 P37989 INFO ************ Epoch=13 end ************
2021-08-13 21:04:05,918 P37989 INFO [Metrics] AUC: 0.734771 - logloss: 0.410488
2021-08-13 21:04:05,921 P37989 INFO Save best model: monitor(max): 0.734771
2021-08-13 21:04:07,788 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 21:04:07,896 P37989 INFO Train loss: 0.416752
2021-08-13 21:04:07,896 P37989 INFO ************ Epoch=14 end ************
2021-08-13 21:08:14,299 P37989 INFO [Metrics] AUC: 0.734919 - logloss: 0.410886
2021-08-13 21:08:14,301 P37989 INFO Save best model: monitor(max): 0.734919
2021-08-13 21:08:16,171 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 21:08:16,284 P37989 INFO Train loss: 0.415793
2021-08-13 21:08:16,284 P37989 INFO ************ Epoch=15 end ************
2021-08-13 21:12:38,803 P37989 INFO [Metrics] AUC: 0.734746 - logloss: 0.410863
2021-08-13 21:12:38,804 P37989 INFO Monitor(max) STOP: 0.734746 !
2021-08-13 21:12:38,805 P37989 INFO Reduce learning rate on plateau: 0.000005
2021-08-13 21:12:38,805 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 21:12:38,930 P37989 INFO Train loss: 0.414973
2021-08-13 21:12:38,930 P37989 INFO ************ Epoch=16 end ************
2021-08-13 21:16:54,704 P37989 INFO [Metrics] AUC: 0.734937 - logloss: 0.411225
2021-08-13 21:16:54,709 P37989 INFO Save best model: monitor(max): 0.734937
2021-08-13 21:16:55,775 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 21:16:55,879 P37989 INFO Train loss: 0.410373
2021-08-13 21:16:55,879 P37989 INFO ************ Epoch=17 end ************
2021-08-13 21:21:13,431 P37989 INFO [Metrics] AUC: 0.734900 - logloss: 0.411452
2021-08-13 21:21:13,433 P37989 INFO Monitor(max) STOP: 0.734900 !
2021-08-13 21:21:13,433 P37989 INFO Reduce learning rate on plateau: 0.000001
2021-08-13 21:21:13,433 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 21:21:13,556 P37989 INFO Train loss: 0.409888
2021-08-13 21:21:13,556 P37989 INFO ************ Epoch=18 end ************
2021-08-13 21:25:37,333 P37989 INFO [Metrics] AUC: 0.734915 - logloss: 0.411577
2021-08-13 21:25:37,336 P37989 INFO Monitor(max) STOP: 0.734915 !
2021-08-13 21:25:37,336 P37989 INFO Reduce learning rate on plateau: 0.000001
2021-08-13 21:25:37,336 P37989 INFO Early stopping at epoch=19
2021-08-13 21:25:37,337 P37989 INFO --- 4381/4381 batches finished ---
2021-08-13 21:25:37,458 P37989 INFO Train loss: 0.409173
2021-08-13 21:25:37,458 P37989 INFO Training finished.
2021-08-13 21:25:37,458 P37989 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/DCN_v2_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/DCN_v2_microvideo_1.7m_x0_010_af571c0e.model
2021-08-13 21:25:41,664 P37989 INFO ****** Train/validation evaluation ******
2021-08-13 21:26:15,255 P37989 INFO [Metrics] AUC: 0.734937 - logloss: 0.411225
2021-08-13 21:26:15,310 P37989 INFO ******** Test evaluation ********
2021-08-13 21:26:15,310 P37989 INFO Loading data...
2021-08-13 21:26:15,311 P37989 INFO Loading test data done.

```
