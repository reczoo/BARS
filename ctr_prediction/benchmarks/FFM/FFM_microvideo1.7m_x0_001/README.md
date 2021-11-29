## FFM_microvideo1.7m_x0_001

A notebook to benchmark FFM on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.723667 - logloss: 0.414184
```


### Logs
```python
2021-09-09 22:43:13,595 P110388 INFO {
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_e31a83b2",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': 'nn.Linear(64, 256, bias=False)', 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 256, bias=False)'], 'max_len': 128, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_hook': 'layers.MaskedAveragePooling()', 'max_len': 128, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FFMv2",
    "model_id": "FFMv2_microvideo_1.7m_x0_002_fbdc2ae3",
    "model_root": "./MicroVideo/FFM_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "regularizer": "0.0001",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "train_data": "../data/MicroVideo/MicroVideo_1.7M_x0/train.csv",
    "valid_data": "../data/MicroVideo/MicroVideo_1.7M_x0/test.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-09-09 22:43:13,595 P110388 INFO Set up feature encoder...
2021-09-09 22:43:13,595 P110388 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_e31a83b2/feature_encoder.pkl
2021-09-09 22:43:24,217 P110388 INFO Total number of parameters: 6398803.
2021-09-09 22:43:24,217 P110388 INFO Loading data...
2021-09-09 22:43:24,219 P110388 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_e31a83b2/train.h5
2021-09-09 22:43:37,930 P110388 INFO Train samples: total/8970309, blocks/1
2021-09-09 22:43:37,931 P110388 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_e31a83b2/valid.h5
2021-09-09 22:43:43,760 P110388 INFO Validation samples: total/3767308, blocks/1%
2021-09-09 22:43:43,760 P110388 INFO Loading train data done.
2021-09-09 22:43:43,760 P110388 INFO Start training: 4381 batches/epoch
2021-09-09 22:43:43,760 P110388 INFO ************ Epoch=1 start ************
2021-09-09 22:46:20,912 P110388 INFO [Metrics] AUC: 0.711782 - logloss: 0.419359
2021-09-09 22:46:20,915 P110388 INFO Save best model: monitor(max): 0.711782
2021-09-09 22:46:34,218 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 22:46:34,314 P110388 INFO Train loss: 0.453391
2021-09-09 22:46:34,314 P110388 INFO ************ Epoch=1 end ************
2021-09-09 22:49:12,238 P110388 INFO [Metrics] AUC: 0.713232 - logloss: 0.419029
2021-09-09 22:49:12,241 P110388 INFO Save best model: monitor(max): 0.713232
2021-09-09 22:49:26,579 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 22:49:26,689 P110388 INFO Train loss: 0.442076
2021-09-09 22:49:26,689 P110388 INFO ************ Epoch=2 end ************
2021-09-09 22:52:04,197 P110388 INFO [Metrics] AUC: 0.714909 - logloss: 0.418076
2021-09-09 22:52:04,200 P110388 INFO Save best model: monitor(max): 0.714909
2021-09-09 22:52:17,872 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 22:52:17,983 P110388 INFO Train loss: 0.439168
2021-09-09 22:52:17,983 P110388 INFO ************ Epoch=3 end ************
2021-09-09 22:54:54,256 P110388 INFO [Metrics] AUC: 0.715099 - logloss: 0.418288
2021-09-09 22:54:54,258 P110388 INFO Save best model: monitor(max): 0.715099
2021-09-09 22:55:08,219 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 22:55:08,331 P110388 INFO Train loss: 0.437805
2021-09-09 22:55:08,331 P110388 INFO ************ Epoch=4 end ************
2021-09-09 22:57:43,950 P110388 INFO [Metrics] AUC: 0.715715 - logloss: 0.417934
2021-09-09 22:57:43,953 P110388 INFO Save best model: monitor(max): 0.715715
2021-09-09 22:57:57,640 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 22:57:57,743 P110388 INFO Train loss: 0.437031
2021-09-09 22:57:57,744 P110388 INFO ************ Epoch=5 end ************
2021-09-09 23:00:31,286 P110388 INFO [Metrics] AUC: 0.715474 - logloss: 0.417889
2021-09-09 23:00:31,289 P110388 INFO Monitor(max) STOP: 0.715474 !
2021-09-09 23:00:31,289 P110388 INFO Reduce learning rate on plateau: 0.000050
2021-09-09 23:00:31,289 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:00:31,388 P110388 INFO Train loss: 0.436556
2021-09-09 23:00:31,388 P110388 INFO ************ Epoch=6 end ************
2021-09-09 23:03:07,136 P110388 INFO [Metrics] AUC: 0.721229 - logloss: 0.414748
2021-09-09 23:03:07,139 P110388 INFO Save best model: monitor(max): 0.721229
2021-09-09 23:03:24,222 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:03:24,335 P110388 INFO Train loss: 0.427989
2021-09-09 23:03:24,335 P110388 INFO ************ Epoch=7 end ************
2021-09-09 23:05:59,148 P110388 INFO [Metrics] AUC: 0.722257 - logloss: 0.414246
2021-09-09 23:05:59,151 P110388 INFO Save best model: monitor(max): 0.722257
2021-09-09 23:06:13,582 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:06:13,704 P110388 INFO Train loss: 0.425722
2021-09-09 23:06:13,705 P110388 INFO ************ Epoch=8 end ************
2021-09-09 23:08:49,318 P110388 INFO [Metrics] AUC: 0.722598 - logloss: 0.414446
2021-09-09 23:08:49,321 P110388 INFO Save best model: monitor(max): 0.722598
2021-09-09 23:09:02,910 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:09:03,025 P110388 INFO Train loss: 0.424888
2021-09-09 23:09:03,026 P110388 INFO ************ Epoch=9 end ************
2021-09-09 23:11:38,957 P110388 INFO [Metrics] AUC: 0.722657 - logloss: 0.414307
2021-09-09 23:11:38,960 P110388 INFO Save best model: monitor(max): 0.722657
2021-09-09 23:11:52,247 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:11:52,360 P110388 INFO Train loss: 0.424350
2021-09-09 23:11:52,360 P110388 INFO ************ Epoch=10 end ************
2021-09-09 23:14:26,747 P110388 INFO [Metrics] AUC: 0.723009 - logloss: 0.414114
2021-09-09 23:14:26,750 P110388 INFO Save best model: monitor(max): 0.723009
2021-09-09 23:14:40,548 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:14:40,656 P110388 INFO Train loss: 0.424042
2021-09-09 23:14:40,657 P110388 INFO ************ Epoch=11 end ************
2021-09-09 23:17:14,658 P110388 INFO [Metrics] AUC: 0.723189 - logloss: 0.414237
2021-09-09 23:17:14,661 P110388 INFO Save best model: monitor(max): 0.723189
2021-09-09 23:17:28,954 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:17:29,068 P110388 INFO Train loss: 0.423751
2021-09-09 23:17:29,068 P110388 INFO ************ Epoch=12 end ************
2021-09-09 23:20:04,395 P110388 INFO [Metrics] AUC: 0.723208 - logloss: 0.414263
2021-09-09 23:20:04,397 P110388 INFO Save best model: monitor(max): 0.723208
2021-09-09 23:20:17,358 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:20:17,471 P110388 INFO Train loss: 0.423554
2021-09-09 23:20:17,472 P110388 INFO ************ Epoch=13 end ************
2021-09-09 23:22:52,832 P110388 INFO [Metrics] AUC: 0.723300 - logloss: 0.414184
2021-09-09 23:22:52,834 P110388 INFO Save best model: monitor(max): 0.723300
2021-09-09 23:23:06,677 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:23:06,784 P110388 INFO Train loss: 0.423399
2021-09-09 23:23:06,785 P110388 INFO ************ Epoch=14 end ************
2021-09-09 23:25:41,124 P110388 INFO [Metrics] AUC: 0.723454 - logloss: 0.414332
2021-09-09 23:25:41,126 P110388 INFO Save best model: monitor(max): 0.723454
2021-09-09 23:25:54,996 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:25:55,101 P110388 INFO Train loss: 0.423293
2021-09-09 23:25:55,101 P110388 INFO ************ Epoch=15 end ************
2021-09-09 23:28:31,830 P110388 INFO [Metrics] AUC: 0.723430 - logloss: 0.414218
2021-09-09 23:28:31,833 P110388 INFO Monitor(max) STOP: 0.723430 !
2021-09-09 23:28:31,833 P110388 INFO Reduce learning rate on plateau: 0.000005
2021-09-09 23:28:31,833 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:28:31,934 P110388 INFO Train loss: 0.423197
2021-09-09 23:28:31,935 P110388 INFO ************ Epoch=16 end ************
2021-09-09 23:31:06,950 P110388 INFO [Metrics] AUC: 0.723564 - logloss: 0.414192
2021-09-09 23:31:06,952 P110388 INFO Save best model: monitor(max): 0.723564
2021-09-09 23:31:20,658 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:31:20,763 P110388 INFO Train loss: 0.421269
2021-09-09 23:31:20,763 P110388 INFO ************ Epoch=17 end ************
2021-09-09 23:33:59,978 P110388 INFO [Metrics] AUC: 0.723620 - logloss: 0.414196
2021-09-09 23:33:59,981 P110388 INFO Save best model: monitor(max): 0.723620
2021-09-09 23:34:13,919 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:34:14,017 P110388 INFO Train loss: 0.421234
2021-09-09 23:34:14,017 P110388 INFO ************ Epoch=18 end ************
2021-09-09 23:36:52,120 P110388 INFO [Metrics] AUC: 0.723626 - logloss: 0.414201
2021-09-09 23:36:52,122 P110388 INFO Save best model: monitor(max): 0.723626
2021-09-09 23:37:06,262 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:37:06,375 P110388 INFO Train loss: 0.421173
2021-09-09 23:37:06,376 P110388 INFO ************ Epoch=19 end ************
2021-09-09 23:39:43,503 P110388 INFO [Metrics] AUC: 0.723630 - logloss: 0.414191
2021-09-09 23:39:43,506 P110388 INFO Save best model: monitor(max): 0.723630
2021-09-09 23:39:57,538 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:39:57,639 P110388 INFO Train loss: 0.421168
2021-09-09 23:39:57,640 P110388 INFO ************ Epoch=20 end ************
2021-09-09 23:42:33,815 P110388 INFO [Metrics] AUC: 0.723639 - logloss: 0.414189
2021-09-09 23:42:33,818 P110388 INFO Save best model: monitor(max): 0.723639
2021-09-09 23:42:47,892 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:42:47,990 P110388 INFO Train loss: 0.421146
2021-09-09 23:42:47,990 P110388 INFO ************ Epoch=21 end ************
2021-09-09 23:45:25,618 P110388 INFO [Metrics] AUC: 0.723667 - logloss: 0.414184
2021-09-09 23:45:25,621 P110388 INFO Save best model: monitor(max): 0.723667
2021-09-09 23:45:39,184 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:45:39,291 P110388 INFO Train loss: 0.421123
2021-09-09 23:45:39,291 P110388 INFO ************ Epoch=22 end ************
2021-09-09 23:48:16,463 P110388 INFO [Metrics] AUC: 0.723637 - logloss: 0.414208
2021-09-09 23:48:16,466 P110388 INFO Monitor(max) STOP: 0.723637 !
2021-09-09 23:48:16,466 P110388 INFO Reduce learning rate on plateau: 0.000001
2021-09-09 23:48:16,466 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:48:16,573 P110388 INFO Train loss: 0.421116
2021-09-09 23:48:16,573 P110388 INFO ************ Epoch=23 end ************
2021-09-09 23:50:52,573 P110388 INFO [Metrics] AUC: 0.723650 - logloss: 0.414210
2021-09-09 23:50:52,575 P110388 INFO Monitor(max) STOP: 0.723650 !
2021-09-09 23:50:52,575 P110388 INFO Reduce learning rate on plateau: 0.000001
2021-09-09 23:50:52,576 P110388 INFO Early stopping at epoch=24
2021-09-09 23:50:52,576 P110388 INFO --- 4381/4381 batches finished ---
2021-09-09 23:50:52,678 P110388 INFO Train loss: 0.420875
2021-09-09 23:50:52,678 P110388 INFO Training finished.
2021-09-09 23:50:52,678 P110388 INFO Load best model: /home/ma-user/work/GroupCTR/benchmark/MicroVideo/FFM_microvideo_1.7m_x0/microvideo_1.7m_x0_e31a83b2/FFMv2_microvideo_1.7m_x0_002_fbdc2ae3.model
2021-09-09 23:50:55,854 P110388 INFO ****** Train/validation evaluation ******
2021-09-09 23:51:26,694 P110388 INFO [Metrics] AUC: 0.723667 - logloss: 0.414184
2021-09-09 23:51:26,807 P110388 INFO ******** Test evaluation ********
2021-09-09 23:51:26,807 P110388 INFO Loading data...
2021-09-09 23:51:26,807 P110388 INFO Loading test data done.

```
