## DeepFM_microvideo1.7m_x0_001

A notebook to benchmark DeepFM on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.734005 - logloss: 0.410363
```


### Logs
```python
2021-08-15 00:49:10,356 P22281 INFO {
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
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepFM",
    "model_id": "DeepFM_microvideo_1.7m_x0_008_0c0bb76b",
    "model_root": "./MicroVideo/DeepFM_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.4",
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
2021-08-15 00:49:10,357 P22281 INFO Set up feature encoder...
2021-08-15 00:49:10,357 P22281 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-08-15 00:49:23,929 P22281 INFO Total number of parameters: 5154708.
2021-08-15 00:49:23,930 P22281 INFO Loading data...
2021-08-15 00:49:23,933 P22281 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-08-15 00:49:36,601 P22281 INFO Train samples: total/8970309, blocks/1
2021-08-15 00:49:36,601 P22281 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-08-15 00:49:41,933 P22281 INFO Validation samples: total/3767308, blocks/1%
2021-08-15 00:49:41,933 P22281 INFO Loading train data done.
2021-08-15 00:49:41,933 P22281 INFO Start training: 4381 batches/epoch
2021-08-15 00:49:41,933 P22281 INFO ************ Epoch=1 start ************
2021-08-15 00:54:35,926 P22281 INFO [Metrics] AUC: 0.712629 - logloss: 0.418069
2021-08-15 00:54:35,929 P22281 INFO Save best model: monitor(max): 0.712629
2021-08-15 00:54:36,850 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 00:54:36,931 P22281 INFO Train loss: 0.494888
2021-08-15 00:54:36,931 P22281 INFO ************ Epoch=1 end ************
2021-08-15 00:59:31,139 P22281 INFO [Metrics] AUC: 0.717228 - logloss: 0.415986
2021-08-15 00:59:31,142 P22281 INFO Save best model: monitor(max): 0.717228
2021-08-15 00:59:32,204 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 00:59:32,300 P22281 INFO Train loss: 0.449051
2021-08-15 00:59:32,301 P22281 INFO ************ Epoch=2 end ************
2021-08-15 01:04:24,476 P22281 INFO [Metrics] AUC: 0.720970 - logloss: 0.415151
2021-08-15 01:04:24,479 P22281 INFO Save best model: monitor(max): 0.720970
2021-08-15 01:04:26,350 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:04:26,445 P22281 INFO Train loss: 0.444330
2021-08-15 01:04:26,445 P22281 INFO ************ Epoch=3 end ************
2021-08-15 01:09:19,557 P22281 INFO [Metrics] AUC: 0.721204 - logloss: 0.413688
2021-08-15 01:09:19,560 P22281 INFO Save best model: monitor(max): 0.721204
2021-08-15 01:09:21,408 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:09:21,496 P22281 INFO Train loss: 0.443039
2021-08-15 01:09:21,496 P22281 INFO ************ Epoch=4 end ************
2021-08-15 01:14:13,170 P22281 INFO [Metrics] AUC: 0.721295 - logloss: 0.414073
2021-08-15 01:14:13,173 P22281 INFO Save best model: monitor(max): 0.721295
2021-08-15 01:14:15,048 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:14:15,149 P22281 INFO Train loss: 0.442470
2021-08-15 01:14:15,149 P22281 INFO ************ Epoch=5 end ************
2021-08-15 01:19:07,779 P22281 INFO [Metrics] AUC: 0.722766 - logloss: 0.413958
2021-08-15 01:19:07,782 P22281 INFO Save best model: monitor(max): 0.722766
2021-08-15 01:19:09,680 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:19:09,778 P22281 INFO Train loss: 0.442072
2021-08-15 01:19:09,778 P22281 INFO ************ Epoch=6 end ************
2021-08-15 01:23:34,868 P22281 INFO [Metrics] AUC: 0.723526 - logloss: 0.413233
2021-08-15 01:23:34,871 P22281 INFO Save best model: monitor(max): 0.723526
2021-08-15 01:23:36,748 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:23:36,855 P22281 INFO Train loss: 0.441731
2021-08-15 01:23:36,856 P22281 INFO ************ Epoch=7 end ************
2021-08-15 01:28:16,572 P22281 INFO [Metrics] AUC: 0.723679 - logloss: 0.413596
2021-08-15 01:28:16,575 P22281 INFO Save best model: monitor(max): 0.723679
2021-08-15 01:28:18,492 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:28:18,597 P22281 INFO Train loss: 0.441453
2021-08-15 01:28:18,597 P22281 INFO ************ Epoch=8 end ************
2021-08-15 01:32:58,973 P22281 INFO [Metrics] AUC: 0.723331 - logloss: 0.414080
2021-08-15 01:32:58,976 P22281 INFO Monitor(max) STOP: 0.723331 !
2021-08-15 01:32:58,976 P22281 INFO Reduce learning rate on plateau: 0.000050
2021-08-15 01:32:58,976 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:32:59,075 P22281 INFO Train loss: 0.441207
2021-08-15 01:32:59,076 P22281 INFO ************ Epoch=9 end ************
2021-08-15 01:37:39,196 P22281 INFO [Metrics] AUC: 0.732117 - logloss: 0.410021
2021-08-15 01:37:39,199 P22281 INFO Save best model: monitor(max): 0.732117
2021-08-15 01:37:41,095 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:37:41,226 P22281 INFO Train loss: 0.429137
2021-08-15 01:37:41,226 P22281 INFO ************ Epoch=10 end ************
2021-08-15 01:42:20,981 P22281 INFO [Metrics] AUC: 0.732936 - logloss: 0.409092
2021-08-15 01:42:20,984 P22281 INFO Save best model: monitor(max): 0.732936
2021-08-15 01:42:22,912 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:42:23,013 P22281 INFO Train loss: 0.424926
2021-08-15 01:42:23,013 P22281 INFO ************ Epoch=11 end ************
2021-08-15 01:47:03,317 P22281 INFO [Metrics] AUC: 0.733144 - logloss: 0.409588
2021-08-15 01:47:03,320 P22281 INFO Save best model: monitor(max): 0.733144
2021-08-15 01:47:05,376 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:47:05,474 P22281 INFO Train loss: 0.423555
2021-08-15 01:47:05,475 P22281 INFO ************ Epoch=12 end ************
2021-08-15 01:51:45,574 P22281 INFO [Metrics] AUC: 0.733302 - logloss: 0.409421
2021-08-15 01:51:45,577 P22281 INFO Save best model: monitor(max): 0.733302
2021-08-15 01:51:47,487 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:51:47,588 P22281 INFO Train loss: 0.422487
2021-08-15 01:51:47,588 P22281 INFO ************ Epoch=13 end ************
2021-08-15 01:56:27,753 P22281 INFO [Metrics] AUC: 0.733182 - logloss: 0.409795
2021-08-15 01:56:27,756 P22281 INFO Monitor(max) STOP: 0.733182 !
2021-08-15 01:56:27,756 P22281 INFO Reduce learning rate on plateau: 0.000005
2021-08-15 01:56:27,757 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 01:56:27,858 P22281 INFO Train loss: 0.421599
2021-08-15 01:56:27,858 P22281 INFO ************ Epoch=14 end ************
2021-08-15 02:01:06,887 P22281 INFO [Metrics] AUC: 0.733724 - logloss: 0.410996
2021-08-15 02:01:06,890 P22281 INFO Save best model: monitor(max): 0.733724
2021-08-15 02:01:08,807 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 02:01:08,915 P22281 INFO Train loss: 0.417866
2021-08-15 02:01:08,916 P22281 INFO ************ Epoch=15 end ************
2021-08-15 02:05:46,966 P22281 INFO [Metrics] AUC: 0.733782 - logloss: 0.410848
2021-08-15 02:05:46,969 P22281 INFO Save best model: monitor(max): 0.733782
2021-08-15 02:05:48,107 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 02:05:48,200 P22281 INFO Train loss: 0.417179
2021-08-15 02:05:48,200 P22281 INFO ************ Epoch=16 end ************
2021-08-15 02:10:26,629 P22281 INFO [Metrics] AUC: 0.733913 - logloss: 0.410300
2021-08-15 02:10:26,632 P22281 INFO Save best model: monitor(max): 0.733913
2021-08-15 02:10:28,550 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 02:10:28,649 P22281 INFO Train loss: 0.416783
2021-08-15 02:10:28,649 P22281 INFO ************ Epoch=17 end ************
2021-08-15 02:15:07,641 P22281 INFO [Metrics] AUC: 0.733969 - logloss: 0.410866
2021-08-15 02:15:07,645 P22281 INFO Save best model: monitor(max): 0.733969
2021-08-15 02:15:09,591 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 02:15:09,689 P22281 INFO Train loss: 0.416501
2021-08-15 02:15:09,690 P22281 INFO ************ Epoch=18 end ************
2021-08-15 02:19:47,628 P22281 INFO [Metrics] AUC: 0.734005 - logloss: 0.410363
2021-08-15 02:19:47,631 P22281 INFO Save best model: monitor(max): 0.734005
2021-08-15 02:19:49,473 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 02:19:49,560 P22281 INFO Train loss: 0.416172
2021-08-15 02:19:49,560 P22281 INFO ************ Epoch=19 end ************
2021-08-15 02:24:29,474 P22281 INFO [Metrics] AUC: 0.733930 - logloss: 0.410600
2021-08-15 02:24:29,477 P22281 INFO Monitor(max) STOP: 0.733930 !
2021-08-15 02:24:29,477 P22281 INFO Reduce learning rate on plateau: 0.000001
2021-08-15 02:24:29,477 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 02:24:29,564 P22281 INFO Train loss: 0.415823
2021-08-15 02:24:29,565 P22281 INFO ************ Epoch=20 end ************
2021-08-15 02:28:58,353 P22281 INFO [Metrics] AUC: 0.733904 - logloss: 0.410882
2021-08-15 02:28:58,355 P22281 INFO Monitor(max) STOP: 0.733904 !
2021-08-15 02:28:58,356 P22281 INFO Reduce learning rate on plateau: 0.000001
2021-08-15 02:28:58,356 P22281 INFO Early stopping at epoch=21
2021-08-15 02:28:58,356 P22281 INFO --- 4381/4381 batches finished ---
2021-08-15 02:28:58,476 P22281 INFO Train loss: 0.415228
2021-08-15 02:28:58,476 P22281 INFO Training finished.
2021-08-15 02:28:58,476 P22281 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/DeepFM_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/DeepFM_microvideo_1.7m_x0_008_0c0bb76b.model
2021-08-15 02:29:02,661 P22281 INFO ****** Train/validation evaluation ******
2021-08-15 02:29:32,456 P22281 INFO [Metrics] AUC: 0.734005 - logloss: 0.410363
2021-08-15 02:29:32,501 P22281 INFO ******** Test evaluation ********
2021-08-15 02:29:32,501 P22281 INFO Loading data...
2021-08-15 02:29:32,502 P22281 INFO Loading test data done.


```
