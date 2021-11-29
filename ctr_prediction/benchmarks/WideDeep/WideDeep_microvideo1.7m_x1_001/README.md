## WideDeep_microvideo1.7m_x0_001

A notebook to benchmark WideDeep on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.734029 - logloss: 0.411181
```


### Logs
```python
2021-09-09 10:15:01,807 P80767 INFO {
    "batch_norm": "False",
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
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
    "model": "WideDeep",
    "model_id": "WideDeep_microvideo_1.7m_x0_001_b7138ebd",
    "model_root": "./MicroVideo/WideDeep_microvideo1.7m_x0/",
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
2021-09-09 10:15:01,808 P80767 INFO Set up feature encoder...
2021-09-09 10:15:01,808 P80767 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-09-09 10:15:12,597 P80767 INFO Total number of parameters: 5151123.
2021-09-09 10:15:12,597 P80767 INFO Loading data...
2021-09-09 10:15:12,599 P80767 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-09-09 10:15:27,250 P80767 INFO Train samples: total/8970309, blocks/1
2021-09-09 10:15:27,251 P80767 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-09-09 10:15:34,782 P80767 INFO Validation samples: total/3767308, blocks/1%
2021-09-09 10:15:34,782 P80767 INFO Loading train data done.
2021-09-09 10:15:34,783 P80767 INFO Start training: 4381 batches/epoch
2021-09-09 10:15:34,783 P80767 INFO ************ Epoch=1 start ************
2021-09-09 10:17:42,337 P80767 INFO [Metrics] AUC: 0.716897 - logloss: 0.417339
2021-09-09 10:17:42,340 P80767 INFO Save best model: monitor(max): 0.716897
2021-09-09 10:17:43,341 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:17:43,501 P80767 INFO Train loss: 0.457412
2021-09-09 10:17:43,502 P80767 INFO ************ Epoch=1 end ************
2021-09-09 10:19:49,465 P80767 INFO [Metrics] AUC: 0.719195 - logloss: 0.415947
2021-09-09 10:19:49,467 P80767 INFO Save best model: monitor(max): 0.719195
2021-09-09 10:20:15,358 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:20:15,494 P80767 INFO Train loss: 0.443898
2021-09-09 10:20:15,495 P80767 INFO ************ Epoch=2 end ************
2021-09-09 10:22:21,649 P80767 INFO [Metrics] AUC: 0.722679 - logloss: 0.413711
2021-09-09 10:22:21,652 P80767 INFO Save best model: monitor(max): 0.722679
2021-09-09 10:22:40,694 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:22:40,845 P80767 INFO Train loss: 0.441701
2021-09-09 10:22:40,845 P80767 INFO ************ Epoch=3 end ************
2021-09-09 10:24:46,086 P80767 INFO [Metrics] AUC: 0.723552 - logloss: 0.413971
2021-09-09 10:24:46,089 P80767 INFO Save best model: monitor(max): 0.723552
2021-09-09 10:24:57,947 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:24:58,086 P80767 INFO Train loss: 0.440120
2021-09-09 10:24:58,087 P80767 INFO ************ Epoch=4 end ************
2021-09-09 10:27:02,208 P80767 INFO [Metrics] AUC: 0.724448 - logloss: 0.414679
2021-09-09 10:27:02,211 P80767 INFO Save best model: monitor(max): 0.724448
2021-09-09 10:27:15,237 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:27:15,374 P80767 INFO Train loss: 0.439254
2021-09-09 10:27:15,375 P80767 INFO ************ Epoch=5 end ************
2021-09-09 10:29:21,506 P80767 INFO [Metrics] AUC: 0.723350 - logloss: 0.414050
2021-09-09 10:29:21,508 P80767 INFO Monitor(max) STOP: 0.723350 !
2021-09-09 10:29:21,509 P80767 INFO Reduce learning rate on plateau: 0.000050
2021-09-09 10:29:21,509 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:29:21,662 P80767 INFO Train loss: 0.438561
2021-09-09 10:29:21,662 P80767 INFO ************ Epoch=6 end ************
2021-09-09 10:31:28,255 P80767 INFO [Metrics] AUC: 0.732455 - logloss: 0.409819
2021-09-09 10:31:28,258 P80767 INFO Save best model: monitor(max): 0.732455
2021-09-09 10:31:40,638 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:31:40,771 P80767 INFO Train loss: 0.424977
2021-09-09 10:31:40,771 P80767 INFO ************ Epoch=7 end ************
2021-09-09 10:33:46,476 P80767 INFO [Metrics] AUC: 0.733183 - logloss: 0.410158
2021-09-09 10:33:46,479 P80767 INFO Save best model: monitor(max): 0.733183
2021-09-09 10:33:58,911 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:33:59,045 P80767 INFO Train loss: 0.420333
2021-09-09 10:33:59,045 P80767 INFO ************ Epoch=8 end ************
2021-09-09 10:36:05,161 P80767 INFO [Metrics] AUC: 0.733722 - logloss: 0.409804
2021-09-09 10:36:05,163 P80767 INFO Save best model: monitor(max): 0.733722
2021-09-09 10:36:17,779 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:36:17,910 P80767 INFO Train loss: 0.418569
2021-09-09 10:36:17,911 P80767 INFO ************ Epoch=9 end ************
2021-09-09 10:38:23,132 P80767 INFO [Metrics] AUC: 0.733555 - logloss: 0.409947
2021-09-09 10:38:23,134 P80767 INFO Monitor(max) STOP: 0.733555 !
2021-09-09 10:38:23,134 P80767 INFO Reduce learning rate on plateau: 0.000005
2021-09-09 10:38:23,134 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:38:23,286 P80767 INFO Train loss: 0.417294
2021-09-09 10:38:23,286 P80767 INFO ************ Epoch=10 end ************
2021-09-09 10:40:29,224 P80767 INFO [Metrics] AUC: 0.733928 - logloss: 0.410707
2021-09-09 10:40:29,227 P80767 INFO Save best model: monitor(max): 0.733928
2021-09-09 10:40:42,364 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:40:42,499 P80767 INFO Train loss: 0.413294
2021-09-09 10:40:42,499 P80767 INFO ************ Epoch=11 end ************
2021-09-09 10:42:48,734 P80767 INFO [Metrics] AUC: 0.733990 - logloss: 0.410906
2021-09-09 10:42:48,737 P80767 INFO Save best model: monitor(max): 0.733990
2021-09-09 10:43:02,578 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:43:02,745 P80767 INFO Train loss: 0.412652
2021-09-09 10:43:02,745 P80767 INFO ************ Epoch=12 end ************
2021-09-09 10:45:09,616 P80767 INFO [Metrics] AUC: 0.734025 - logloss: 0.411046
2021-09-09 10:45:09,619 P80767 INFO Save best model: monitor(max): 0.734025
2021-09-09 10:45:23,822 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:45:23,984 P80767 INFO Train loss: 0.412211
2021-09-09 10:45:23,984 P80767 INFO ************ Epoch=13 end ************
2021-09-09 10:47:27,978 P80767 INFO [Metrics] AUC: 0.733969 - logloss: 0.411011
2021-09-09 10:47:27,981 P80767 INFO Monitor(max) STOP: 0.733969 !
2021-09-09 10:47:27,981 P80767 INFO Reduce learning rate on plateau: 0.000001
2021-09-09 10:47:27,981 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:47:28,129 P80767 INFO Train loss: 0.411847
2021-09-09 10:47:28,129 P80767 INFO ************ Epoch=14 end ************
2021-09-09 10:49:32,377 P80767 INFO [Metrics] AUC: 0.734029 - logloss: 0.411181
2021-09-09 10:49:32,380 P80767 INFO Save best model: monitor(max): 0.734029
2021-09-09 10:49:44,350 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:49:44,486 P80767 INFO Train loss: 0.411216
2021-09-09 10:49:44,486 P80767 INFO ************ Epoch=15 end ************
2021-09-09 10:51:50,783 P80767 INFO [Metrics] AUC: 0.734020 - logloss: 0.411200
2021-09-09 10:51:50,786 P80767 INFO Monitor(max) STOP: 0.734020 !
2021-09-09 10:51:50,786 P80767 INFO Reduce learning rate on plateau: 0.000001
2021-09-09 10:51:50,786 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:51:50,921 P80767 INFO Train loss: 0.411123
2021-09-09 10:51:50,921 P80767 INFO ************ Epoch=16 end ************
2021-09-09 10:53:57,281 P80767 INFO [Metrics] AUC: 0.734025 - logloss: 0.411204
2021-09-09 10:53:57,284 P80767 INFO Monitor(max) STOP: 0.734025 !
2021-09-09 10:53:57,284 P80767 INFO Reduce learning rate on plateau: 0.000001
2021-09-09 10:53:57,284 P80767 INFO Early stopping at epoch=17
2021-09-09 10:53:57,284 P80767 INFO --- 4381/4381 batches finished ---
2021-09-09 10:53:57,425 P80767 INFO Train loss: 0.411152
2021-09-09 10:53:57,425 P80767 INFO Training finished.
2021-09-09 10:53:57,426 P80767 INFO Load best model: /home/ma-user/work/GroupCTR/benchmark/MicroVideo/WideDeep_microvideo1.7m_x0/microvideo_1.7m_x0_710d1f85/WideDeep_microvideo_1.7m_x0_001_b7138ebd.model
2021-09-09 10:53:58,011 P80767 INFO ****** Train/validation evaluation ******
2021-09-09 10:54:19,855 P80767 INFO [Metrics] AUC: 0.734029 - logloss: 0.411181
2021-09-09 10:54:19,967 P80767 INFO ******** Test evaluation ********
2021-09-09 10:54:19,968 P80767 INFO Loading data...
2021-09-09 10:54:19,968 P80767 INFO Loading test data done.



```
