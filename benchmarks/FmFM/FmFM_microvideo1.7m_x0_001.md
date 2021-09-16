## FmFM_microvideo1.7m_x0_001

A notebook to benchmark FmFM on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.722027 - logloss: 0.415730
```


### Logs
```python
2021-09-05 20:06:51,072 P67458 INFO {
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'max_len': 128, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_hook': 'layers.MaskedAveragePooling()', 'max_len': 128, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "field_interaction_type": "matrixed",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_microvideo_1.7m_x0_001_8bfb8d74",
    "model_root": "./MicroVideo/FmFM_microvideo_1.7m_x0/",
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
2021-09-05 20:06:51,074 P67458 INFO Set up feature encoder...
2021-09-05 20:06:51,074 P67458 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-09-05 20:07:37,103 P67458 INFO Total number of parameters: 4206995.
2021-09-05 20:07:37,104 P67458 INFO Loading data...
2021-09-05 20:07:37,106 P67458 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-09-05 20:10:23,518 P67458 INFO Train samples: total/8970309, blocks/1
2021-09-05 20:10:23,519 P67458 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-09-05 20:11:30,037 P67458 INFO Validation samples: total/3767308, blocks/1%
2021-09-05 20:11:30,037 P67458 INFO Loading train data done.
2021-09-05 20:11:30,037 P67458 INFO Start training: 4381 batches/epoch
2021-09-05 20:11:30,038 P67458 INFO ************ Epoch=1 start ************
2021-09-05 20:15:30,238 P67458 INFO [Metrics] AUC: 0.708767 - logloss: 0.420473
2021-09-05 20:15:30,241 P67458 INFO Save best model: monitor(max): 0.708767
2021-09-05 20:15:31,308 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:15:31,515 P67458 INFO Train loss: 0.450666
2021-09-05 20:15:31,515 P67458 INFO ************ Epoch=1 end ************
2021-09-05 20:19:32,037 P67458 INFO [Metrics] AUC: 0.712010 - logloss: 0.419249
2021-09-05 20:19:32,040 P67458 INFO Save best model: monitor(max): 0.712010
2021-09-05 20:19:59,617 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:19:59,811 P67458 INFO Train loss: 0.441987
2021-09-05 20:19:59,811 P67458 INFO ************ Epoch=2 end ************
2021-09-05 20:23:57,388 P67458 INFO [Metrics] AUC: 0.712891 - logloss: 0.419363
2021-09-05 20:23:57,390 P67458 INFO Save best model: monitor(max): 0.712891
2021-09-05 20:24:10,120 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:24:10,313 P67458 INFO Train loss: 0.439367
2021-09-05 20:24:10,313 P67458 INFO ************ Epoch=3 end ************
2021-09-05 20:28:03,450 P67458 INFO [Metrics] AUC: 0.713890 - logloss: 0.418385
2021-09-05 20:28:03,452 P67458 INFO Save best model: monitor(max): 0.713890
2021-09-05 20:28:17,632 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:28:17,828 P67458 INFO Train loss: 0.437909
2021-09-05 20:28:17,829 P67458 INFO ************ Epoch=4 end ************
2021-09-05 20:32:09,799 P67458 INFO [Metrics] AUC: 0.713824 - logloss: 0.419844
2021-09-05 20:32:09,802 P67458 INFO Monitor(max) STOP: 0.713824 !
2021-09-05 20:32:09,802 P67458 INFO Reduce learning rate on plateau: 0.000050
2021-09-05 20:32:09,802 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:32:09,985 P67458 INFO Train loss: 0.436958
2021-09-05 20:32:09,986 P67458 INFO ************ Epoch=5 end ************
2021-09-05 20:36:06,016 P67458 INFO [Metrics] AUC: 0.720236 - logloss: 0.415692
2021-09-05 20:36:06,019 P67458 INFO Save best model: monitor(max): 0.720236
2021-09-05 20:36:18,606 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:36:20,788 P67458 INFO Train loss: 0.429170
2021-09-05 20:36:20,789 P67458 INFO ************ Epoch=6 end ************
2021-09-05 20:38:46,992 P67458 INFO [Metrics] AUC: 0.721109 - logloss: 0.415608
2021-09-05 20:38:46,995 P67458 INFO Save best model: monitor(max): 0.721109
2021-09-05 20:38:59,923 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:39:00,104 P67458 INFO Train loss: 0.426821
2021-09-05 20:39:00,104 P67458 INFO ************ Epoch=7 end ************
2021-09-05 20:41:26,502 P67458 INFO [Metrics] AUC: 0.721135 - logloss: 0.415415
2021-09-05 20:41:26,506 P67458 INFO Save best model: monitor(max): 0.721135
2021-09-05 20:41:39,125 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:41:41,327 P67458 INFO Train loss: 0.425800
2021-09-05 20:41:41,328 P67458 INFO ************ Epoch=8 end ************
2021-09-05 20:44:07,170 P67458 INFO [Metrics] AUC: 0.721261 - logloss: 0.415466
2021-09-05 20:44:07,173 P67458 INFO Save best model: monitor(max): 0.721261
2021-09-05 20:44:19,445 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:44:19,685 P67458 INFO Train loss: 0.425187
2021-09-05 20:44:19,685 P67458 INFO ************ Epoch=9 end ************
2021-09-05 20:46:44,184 P67458 INFO [Metrics] AUC: 0.721606 - logloss: 0.415481
2021-09-05 20:46:44,187 P67458 INFO Save best model: monitor(max): 0.721606
2021-09-05 20:46:56,827 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:46:57,039 P67458 INFO Train loss: 0.424681
2021-09-05 20:46:57,040 P67458 INFO ************ Epoch=10 end ************
2021-09-05 20:49:22,381 P67458 INFO [Metrics] AUC: 0.721745 - logloss: 0.415572
2021-09-05 20:49:22,384 P67458 INFO Save best model: monitor(max): 0.721745
2021-09-05 20:49:35,028 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:49:37,210 P67458 INFO Train loss: 0.424286
2021-09-05 20:49:37,210 P67458 INFO ************ Epoch=11 end ************
2021-09-05 20:52:01,045 P67458 INFO [Metrics] AUC: 0.721724 - logloss: 0.415827
2021-09-05 20:52:01,047 P67458 INFO Monitor(max) STOP: 0.721724 !
2021-09-05 20:52:01,048 P67458 INFO Reduce learning rate on plateau: 0.000005
2021-09-05 20:52:01,048 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:52:01,235 P67458 INFO Train loss: 0.423946
2021-09-05 20:52:01,235 P67458 INFO ************ Epoch=12 end ************
2021-09-05 20:54:27,289 P67458 INFO [Metrics] AUC: 0.721936 - logloss: 0.415664
2021-09-05 20:54:27,291 P67458 INFO Save best model: monitor(max): 0.721936
2021-09-05 20:54:39,549 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:54:39,739 P67458 INFO Train loss: 0.421929
2021-09-05 20:54:39,740 P67458 INFO ************ Epoch=13 end ************
2021-09-05 20:57:03,909 P67458 INFO [Metrics] AUC: 0.721887 - logloss: 0.415704
2021-09-05 20:57:03,911 P67458 INFO Monitor(max) STOP: 0.721887 !
2021-09-05 20:57:03,912 P67458 INFO Reduce learning rate on plateau: 0.000001
2021-09-05 20:57:03,912 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:57:04,095 P67458 INFO Train loss: 0.421842
2021-09-05 20:57:04,096 P67458 INFO ************ Epoch=14 end ************
2021-09-05 20:59:30,278 P67458 INFO [Metrics] AUC: 0.721982 - logloss: 0.415683
2021-09-05 20:59:30,281 P67458 INFO Save best model: monitor(max): 0.721982
2021-09-05 20:59:43,152 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 20:59:45,304 P67458 INFO Train loss: 0.421612
2021-09-05 20:59:45,304 P67458 INFO ************ Epoch=15 end ************
2021-09-05 21:02:11,339 P67458 INFO [Metrics] AUC: 0.721987 - logloss: 0.415689
2021-09-05 21:02:11,342 P67458 INFO Save best model: monitor(max): 0.721987
2021-09-05 21:02:23,445 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:02:25,680 P67458 INFO Train loss: 0.421590
2021-09-05 21:02:25,680 P67458 INFO ************ Epoch=16 end ************
2021-09-05 21:04:51,434 P67458 INFO [Metrics] AUC: 0.721977 - logloss: 0.415692
2021-09-05 21:04:51,437 P67458 INFO Monitor(max) STOP: 0.721977 !
2021-09-05 21:04:51,437 P67458 INFO Reduce learning rate on plateau: 0.000001
2021-09-05 21:04:51,437 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:04:51,680 P67458 INFO Train loss: 0.421578
2021-09-05 21:04:51,680 P67458 INFO ************ Epoch=17 end ************
2021-09-05 21:07:17,331 P67458 INFO [Metrics] AUC: 0.722006 - logloss: 0.415702
2021-09-05 21:07:17,333 P67458 INFO Save best model: monitor(max): 0.722006
2021-09-05 21:07:29,889 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:07:32,106 P67458 INFO Train loss: 0.421597
2021-09-05 21:07:32,107 P67458 INFO ************ Epoch=18 end ************
2021-09-05 21:10:00,201 P67458 INFO [Metrics] AUC: 0.721990 - logloss: 0.415711
2021-09-05 21:10:00,203 P67458 INFO Monitor(max) STOP: 0.721990 !
2021-09-05 21:10:00,203 P67458 INFO Reduce learning rate on plateau: 0.000001
2021-09-05 21:10:00,203 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:10:00,419 P67458 INFO Train loss: 0.421542
2021-09-05 21:10:00,419 P67458 INFO ************ Epoch=19 end ************
2021-09-05 21:12:28,478 P67458 INFO [Metrics] AUC: 0.722011 - logloss: 0.415718
2021-09-05 21:12:28,481 P67458 INFO Save best model: monitor(max): 0.722011
2021-09-05 21:12:40,834 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:12:42,009 P67458 INFO Train loss: 0.421552
2021-09-05 21:12:42,009 P67458 INFO ************ Epoch=20 end ************
2021-09-05 21:15:09,204 P67458 INFO [Metrics] AUC: 0.722013 - logloss: 0.415724
2021-09-05 21:15:09,208 P67458 INFO Save best model: monitor(max): 0.722013
2021-09-05 21:15:22,159 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:15:22,377 P67458 INFO Train loss: 0.421551
2021-09-05 21:15:22,378 P67458 INFO ************ Epoch=21 end ************
2021-09-05 21:17:49,552 P67458 INFO [Metrics] AUC: 0.721998 - logloss: 0.415727
2021-09-05 21:17:49,554 P67458 INFO Monitor(max) STOP: 0.721998 !
2021-09-05 21:17:49,555 P67458 INFO Reduce learning rate on plateau: 0.000001
2021-09-05 21:17:49,555 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:17:49,782 P67458 INFO Train loss: 0.421515
2021-09-05 21:17:49,782 P67458 INFO ************ Epoch=22 end ************
2021-09-05 21:20:17,053 P67458 INFO [Metrics] AUC: 0.722027 - logloss: 0.415730
2021-09-05 21:20:17,056 P67458 INFO Save best model: monitor(max): 0.722027
2021-09-05 21:20:30,814 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:20:31,045 P67458 INFO Train loss: 0.421514
2021-09-05 21:20:31,046 P67458 INFO ************ Epoch=23 end ************
2021-09-05 21:22:57,722 P67458 INFO [Metrics] AUC: 0.722019 - logloss: 0.415737
2021-09-05 21:22:57,725 P67458 INFO Monitor(max) STOP: 0.722019 !
2021-09-05 21:22:57,725 P67458 INFO Reduce learning rate on plateau: 0.000001
2021-09-05 21:22:57,725 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:22:57,941 P67458 INFO Train loss: 0.421499
2021-09-05 21:22:57,941 P67458 INFO ************ Epoch=24 end ************
2021-09-05 21:25:23,677 P67458 INFO [Metrics] AUC: 0.722020 - logloss: 0.415740
2021-09-05 21:25:23,680 P67458 INFO Monitor(max) STOP: 0.722020 !
2021-09-05 21:25:23,680 P67458 INFO Reduce learning rate on plateau: 0.000001
2021-09-05 21:25:23,680 P67458 INFO Early stopping at epoch=25
2021-09-05 21:25:23,680 P67458 INFO --- 4381/4381 batches finished ---
2021-09-05 21:25:23,903 P67458 INFO Train loss: 0.421498
2021-09-05 21:25:23,903 P67458 INFO Training finished.
2021-09-05 21:25:23,903 P67458 INFO Load best model: /home/ma-user/work/GroupCTR/benchmark/MicroVideo/FmFM_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/FmFM_microvideo_1.7m_x0_001_8bfb8d74.model
2021-09-05 21:25:24,513 P67458 INFO ****** Train/validation evaluation ******
2021-09-05 21:25:51,892 P67458 INFO [Metrics] AUC: 0.722027 - logloss: 0.415730
2021-09-05 21:25:52,542 P67458 INFO ******** Test evaluation ********
2021-09-05 21:25:52,543 P67458 INFO Loading data...
2021-09-05 21:25:52,543 P67458 INFO Loading test data done.


```
