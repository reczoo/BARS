## LR_microvideo1.7m_x0_001

A notebook to benchmark LR on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.661868 - logloss: 0.438600
```


### Logs
```python
2021-08-14 18:45:46,098 P47836 INFO {
    "batch_size": "2048",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'max_len': 128, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_hook': 'layers.MaskedAveragePooling()', 'max_len': 128, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "LR",
    "model_id": "LR_microvideo_1.7m_x0_005_1c991629",
    "model_root": "./MicroVideo/FM_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "train_data": "../data/MicroVideo/MicroVideo_1.7M_x0/train.csv",
    "valid_data": "../data/MicroVideo/MicroVideo_1.7M_x0/test.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-08-14 18:45:46,099 P47836 INFO Set up feature encoder...
2021-08-14 18:45:46,099 P47836 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-08-14 18:45:55,096 P47836 INFO Total number of parameters: 3421779.
2021-08-14 18:45:55,097 P47836 INFO Loading data...
2021-08-14 18:45:55,100 P47836 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-08-14 18:46:08,914 P47836 INFO Train samples: total/8970309, blocks/1
2021-08-14 18:46:08,914 P47836 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-08-14 18:46:14,741 P47836 INFO Validation samples: total/3767308, blocks/1%
2021-08-14 18:46:14,742 P47836 INFO Loading train data done.
2021-08-14 18:46:14,742 P47836 INFO Start training: 4381 batches/epoch
2021-08-14 18:46:14,742 P47836 INFO ************ Epoch=1 start ************
2021-08-14 18:47:43,734 P47836 INFO [Metrics] AUC: 0.656287 - logloss: 0.441270
2021-08-14 18:47:43,736 P47836 INFO Save best model: monitor(max): 0.656287
2021-08-14 18:47:43,749 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:47:43,847 P47836 INFO Train loss: 0.462342
2021-08-14 18:47:43,847 P47836 INFO ************ Epoch=1 end ************
2021-08-14 18:49:12,718 P47836 INFO [Metrics] AUC: 0.657531 - logloss: 0.440559
2021-08-14 18:49:12,721 P47836 INFO Save best model: monitor(max): 0.657531
2021-08-14 18:49:12,738 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:49:12,839 P47836 INFO Train loss: 0.447493
2021-08-14 18:49:12,839 P47836 INFO ************ Epoch=2 end ************
2021-08-14 18:50:41,752 P47836 INFO [Metrics] AUC: 0.660081 - logloss: 0.440379
2021-08-14 18:50:41,756 P47836 INFO Save best model: monitor(max): 0.660081
2021-08-14 18:50:41,774 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:50:41,869 P47836 INFO Train loss: 0.445470
2021-08-14 18:50:41,870 P47836 INFO ************ Epoch=3 end ************
2021-08-14 18:52:10,479 P47836 INFO [Metrics] AUC: 0.659187 - logloss: 0.440401
2021-08-14 18:52:10,482 P47836 INFO Monitor(max) STOP: 0.659187 !
2021-08-14 18:52:10,483 P47836 INFO Reduce learning rate on plateau: 0.000050
2021-08-14 18:52:10,483 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:52:10,580 P47836 INFO Train loss: 0.444719
2021-08-14 18:52:10,580 P47836 INFO ************ Epoch=4 end ************
2021-08-14 18:53:38,660 P47836 INFO [Metrics] AUC: 0.661139 - logloss: 0.439106
2021-08-14 18:53:38,663 P47836 INFO Save best model: monitor(max): 0.661139
2021-08-14 18:53:38,681 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:53:38,790 P47836 INFO Train loss: 0.441857
2021-08-14 18:53:38,791 P47836 INFO ************ Epoch=5 end ************
2021-08-14 18:55:07,298 P47836 INFO [Metrics] AUC: 0.661663 - logloss: 0.438777
2021-08-14 18:55:07,302 P47836 INFO Save best model: monitor(max): 0.661663
2021-08-14 18:55:07,320 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:55:07,432 P47836 INFO Train loss: 0.441286
2021-08-14 18:55:07,432 P47836 INFO ************ Epoch=6 end ************
2021-08-14 18:56:33,224 P47836 INFO [Metrics] AUC: 0.661844 - logloss: 0.438770
2021-08-14 18:56:33,227 P47836 INFO Save best model: monitor(max): 0.661844
2021-08-14 18:56:33,244 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:56:33,350 P47836 INFO Train loss: 0.441150
2021-08-14 18:56:33,350 P47836 INFO ************ Epoch=7 end ************
2021-08-14 18:58:00,855 P47836 INFO [Metrics] AUC: 0.661780 - logloss: 0.438737
2021-08-14 18:58:00,859 P47836 INFO Monitor(max) STOP: 0.661780 !
2021-08-14 18:58:00,859 P47836 INFO Reduce learning rate on plateau: 0.000005
2021-08-14 18:58:00,859 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:58:00,969 P47836 INFO Train loss: 0.441091
2021-08-14 18:58:00,969 P47836 INFO ************ Epoch=8 end ************
2021-08-14 18:59:26,576 P47836 INFO [Metrics] AUC: 0.661856 - logloss: 0.438654
2021-08-14 18:59:26,580 P47836 INFO Save best model: monitor(max): 0.661856
2021-08-14 18:59:26,596 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 18:59:26,723 P47836 INFO Train loss: 0.440638
2021-08-14 18:59:26,723 P47836 INFO ************ Epoch=9 end ************
2021-08-14 19:00:38,114 P47836 INFO [Metrics] AUC: 0.661868 - logloss: 0.438600
2021-08-14 19:00:38,118 P47836 INFO Save best model: monitor(max): 0.661868
2021-08-14 19:00:38,134 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 19:00:38,217 P47836 INFO Train loss: 0.440615
2021-08-14 19:00:38,217 P47836 INFO ************ Epoch=10 end ************
2021-08-14 19:01:46,952 P47836 INFO [Metrics] AUC: 0.661847 - logloss: 0.438626
2021-08-14 19:01:46,955 P47836 INFO Monitor(max) STOP: 0.661847 !
2021-08-14 19:01:46,955 P47836 INFO Reduce learning rate on plateau: 0.000001
2021-08-14 19:01:46,955 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 19:01:47,044 P47836 INFO Train loss: 0.440624
2021-08-14 19:01:47,044 P47836 INFO ************ Epoch=11 end ************
2021-08-14 19:02:55,906 P47836 INFO [Metrics] AUC: 0.661850 - logloss: 0.438632
2021-08-14 19:02:55,909 P47836 INFO Monitor(max) STOP: 0.661850 !
2021-08-14 19:02:55,909 P47836 INFO Reduce learning rate on plateau: 0.000001
2021-08-14 19:02:55,910 P47836 INFO Early stopping at epoch=12
2021-08-14 19:02:55,910 P47836 INFO --- 4381/4381 batches finished ---
2021-08-14 19:02:56,009 P47836 INFO Train loss: 0.440548
2021-08-14 19:02:56,009 P47836 INFO Training finished.
2021-08-14 19:02:56,009 P47836 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/FM_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/LR_microvideo_1.7m_x0_005_1c991629.model
2021-08-14 19:02:58,767 P47836 INFO ****** Train/validation evaluation ******
2021-08-14 19:03:17,617 P47836 INFO [Metrics] AUC: 0.661868 - logloss: 0.438600
2021-08-14 19:03:17,675 P47836 INFO ******** Test evaluation ********
2021-08-14 19:03:17,676 P47836 INFO Loading data...
2021-08-14 19:03:17,676 P47836 INFO Loading test data done.


```
