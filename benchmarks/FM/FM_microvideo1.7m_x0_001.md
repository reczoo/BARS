## FM_microvideo1.7m_x0_001

A notebook to benchmark FM on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.720091 - logloss: 0.414796
```


### Logs
```python
2021-08-14 18:15:58,512 P3904 INFO {
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_dropout": "0",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'max_len': 128, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_hook': 'layers.MaskedAveragePooling()', 'max_len': 128, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_microvideo_1.7m_x0_002_73b6343b",
    "model_root": "./MicroVideo/FM_microvideo_1.7m_x0/",
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
2021-08-14 18:15:58,513 P3904 INFO Set up feature encoder...
2021-08-14 18:15:58,513 P3904 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-08-14 18:16:13,008 P3904 INFO Total number of parameters: 4166035.
2021-08-14 18:16:13,009 P3904 INFO Loading data...
2021-08-14 18:16:13,012 P3904 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-08-14 18:16:30,193 P3904 INFO Train samples: total/8970309, blocks/1
2021-08-14 18:16:30,193 P3904 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-08-14 18:16:36,011 P3904 INFO Validation samples: total/3767308, blocks/1%
2021-08-14 18:16:36,011 P3904 INFO Loading train data done.
2021-08-14 18:16:36,011 P3904 INFO Start training: 4381 batches/epoch
2021-08-14 18:16:36,011 P3904 INFO ************ Epoch=1 start ************
2021-08-14 18:20:13,143 P3904 INFO [Metrics] AUC: 0.707577 - logloss: 0.421310
2021-08-14 18:20:13,145 P3904 INFO Save best model: monitor(max): 0.707577
2021-08-14 18:20:14,086 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:20:14,188 P3904 INFO Train loss: 0.470895
2021-08-14 18:20:14,188 P3904 INFO ************ Epoch=1 end ************
2021-08-14 18:23:49,313 P3904 INFO [Metrics] AUC: 0.708797 - logloss: 0.420597
2021-08-14 18:23:49,314 P3904 INFO Save best model: monitor(max): 0.708797
2021-08-14 18:23:51,146 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:23:51,235 P3904 INFO Train loss: 0.445108
2021-08-14 18:23:51,235 P3904 INFO ************ Epoch=2 end ************
2021-08-14 18:27:27,313 P3904 INFO [Metrics] AUC: 0.710473 - logloss: 0.420193
2021-08-14 18:27:27,315 P3904 INFO Save best model: monitor(max): 0.710473
2021-08-14 18:27:29,107 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:27:29,217 P3904 INFO Train loss: 0.443406
2021-08-14 18:27:29,218 P3904 INFO ************ Epoch=3 end ************
2021-08-14 18:31:01,886 P3904 INFO [Metrics] AUC: 0.711434 - logloss: 0.420261
2021-08-14 18:31:01,889 P3904 INFO Save best model: monitor(max): 0.711434
2021-08-14 18:31:03,911 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:31:04,059 P3904 INFO Train loss: 0.442591
2021-08-14 18:31:04,059 P3904 INFO ************ Epoch=4 end ************
2021-08-14 18:34:39,120 P3904 INFO [Metrics] AUC: 0.710774 - logloss: 0.419824
2021-08-14 18:34:39,122 P3904 INFO Monitor(max) STOP: 0.710774 !
2021-08-14 18:34:39,122 P3904 INFO Reduce learning rate on plateau: 0.000050
2021-08-14 18:34:39,122 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:34:39,240 P3904 INFO Train loss: 0.442011
2021-08-14 18:34:39,240 P3904 INFO ************ Epoch=5 end ************
2021-08-14 18:38:16,027 P3904 INFO [Metrics] AUC: 0.717998 - logloss: 0.415647
2021-08-14 18:38:16,030 P3904 INFO Save best model: monitor(max): 0.717998
2021-08-14 18:38:18,096 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:38:18,255 P3904 INFO Train loss: 0.432709
2021-08-14 18:38:18,255 P3904 INFO ************ Epoch=6 end ************
2021-08-14 18:41:53,088 P3904 INFO [Metrics] AUC: 0.718784 - logloss: 0.415253
2021-08-14 18:41:53,090 P3904 INFO Save best model: monitor(max): 0.718784
2021-08-14 18:41:55,129 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:41:55,245 P3904 INFO Train loss: 0.430317
2021-08-14 18:41:55,245 P3904 INFO ************ Epoch=7 end ************
2021-08-14 18:45:29,495 P3904 INFO [Metrics] AUC: 0.718918 - logloss: 0.415146
2021-08-14 18:45:29,498 P3904 INFO Save best model: monitor(max): 0.718918
2021-08-14 18:45:31,524 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:45:31,648 P3904 INFO Train loss: 0.429509
2021-08-14 18:45:31,649 P3904 INFO ************ Epoch=8 end ************
2021-08-14 18:49:06,326 P3904 INFO [Metrics] AUC: 0.719035 - logloss: 0.415108
2021-08-14 18:49:06,328 P3904 INFO Save best model: monitor(max): 0.719035
2021-08-14 18:49:08,409 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:49:08,515 P3904 INFO Train loss: 0.429021
2021-08-14 18:49:08,515 P3904 INFO ************ Epoch=9 end ************
2021-08-14 18:52:43,438 P3904 INFO [Metrics] AUC: 0.719091 - logloss: 0.415172
2021-08-14 18:52:43,441 P3904 INFO Save best model: monitor(max): 0.719091
2021-08-14 18:52:44,649 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:52:44,841 P3904 INFO Train loss: 0.428646
2021-08-14 18:52:44,842 P3904 INFO ************ Epoch=10 end ************
2021-08-14 18:56:20,463 P3904 INFO [Metrics] AUC: 0.719377 - logloss: 0.415070
2021-08-14 18:56:20,466 P3904 INFO Save best model: monitor(max): 0.719377
2021-08-14 18:56:22,307 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:56:22,489 P3904 INFO Train loss: 0.428340
2021-08-14 18:56:22,489 P3904 INFO ************ Epoch=11 end ************
2021-08-14 18:59:57,778 P3904 INFO [Metrics] AUC: 0.719564 - logloss: 0.415138
2021-08-14 18:59:57,781 P3904 INFO Save best model: monitor(max): 0.719564
2021-08-14 18:59:59,607 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 18:59:59,721 P3904 INFO Train loss: 0.428071
2021-08-14 18:59:59,721 P3904 INFO ************ Epoch=12 end ************
2021-08-14 19:03:34,480 P3904 INFO [Metrics] AUC: 0.719430 - logloss: 0.415026
2021-08-14 19:03:34,482 P3904 INFO Monitor(max) STOP: 0.719430 !
2021-08-14 19:03:34,482 P3904 INFO Reduce learning rate on plateau: 0.000005
2021-08-14 19:03:34,482 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 19:03:34,652 P3904 INFO Train loss: 0.427840
2021-08-14 19:03:34,652 P3904 INFO ************ Epoch=13 end ************
2021-08-14 19:07:07,923 P3904 INFO [Metrics] AUC: 0.720033 - logloss: 0.414837
2021-08-14 19:07:07,925 P3904 INFO Save best model: monitor(max): 0.720033
2021-08-14 19:07:09,758 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 19:07:09,891 P3904 INFO Train loss: 0.425900
2021-08-14 19:07:09,892 P3904 INFO ************ Epoch=14 end ************
2021-08-14 19:10:43,525 P3904 INFO [Metrics] AUC: 0.720091 - logloss: 0.414796
2021-08-14 19:10:43,528 P3904 INFO Save best model: monitor(max): 0.720091
2021-08-14 19:10:45,332 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 19:10:45,454 P3904 INFO Train loss: 0.425843
2021-08-14 19:10:45,454 P3904 INFO ************ Epoch=15 end ************
2021-08-14 19:14:19,732 P3904 INFO [Metrics] AUC: 0.720090 - logloss: 0.414860
2021-08-14 19:14:19,735 P3904 INFO Monitor(max) STOP: 0.720090 !
2021-08-14 19:14:19,735 P3904 INFO Reduce learning rate on plateau: 0.000001
2021-08-14 19:14:19,735 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 19:14:19,860 P3904 INFO Train loss: 0.425799
2021-08-14 19:14:19,860 P3904 INFO ************ Epoch=16 end ************
2021-08-14 19:17:54,355 P3904 INFO [Metrics] AUC: 0.720074 - logloss: 0.414805
2021-08-14 19:17:54,357 P3904 INFO Monitor(max) STOP: 0.720074 !
2021-08-14 19:17:54,357 P3904 INFO Reduce learning rate on plateau: 0.000001
2021-08-14 19:17:54,358 P3904 INFO Early stopping at epoch=17
2021-08-14 19:17:54,358 P3904 INFO --- 4381/4381 batches finished ---
2021-08-14 19:17:54,491 P3904 INFO Train loss: 0.425562
2021-08-14 19:17:54,491 P3904 INFO Training finished.
2021-08-14 19:17:54,491 P3904 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/FM_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/FM_microvideo_1.7m_x0_002_73b6343b.model
2021-08-14 19:17:55,074 P3904 INFO ****** Train/validation evaluation ******
2021-08-14 19:18:20,444 P3904 INFO [Metrics] AUC: 0.720091 - logloss: 0.414796
2021-08-14 19:18:20,492 P3904 INFO ******** Test evaluation ********
2021-08-14 19:18:20,492 P3904 INFO Loading data...
2021-08-14 19:18:20,493 P3904 INFO Loading test data done.


```
