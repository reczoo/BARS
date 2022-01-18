## DeepIM_microvideo1.7m_x0_001

A notebook to benchmark DeepIM on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.734781 - logloss: 0.410982
```


### Logs
```python
2021-09-04 09:12:06,937 P48837 INFO {
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
    "im_batch_norm": "True",
    "im_order": "2",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_microvideo_1.7m_x0_008_3d20f1b6",
    "model_root": "./MicroVideo/DeepIM_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_batch_norm": "True",
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
2021-09-04 09:12:06,938 P48837 INFO Set up feature encoder...
2021-09-04 09:12:06,938 P48837 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-09-04 09:12:20,728 P48837 INFO Total number of parameters: 1733314.
2021-09-04 09:12:20,728 P48837 INFO Loading data...
2021-09-04 09:12:20,731 P48837 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-09-04 09:12:33,731 P48837 INFO Train samples: total/8970309, blocks/1
2021-09-04 09:12:33,731 P48837 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-09-04 09:12:39,571 P48837 INFO Validation samples: total/3767308, blocks/1%
2021-09-04 09:12:39,571 P48837 INFO Loading train data done.
2021-09-04 09:12:39,572 P48837 INFO Start training: 4381 batches/epoch
2021-09-04 09:12:39,572 P48837 INFO ************ Epoch=1 start ************
2021-09-04 09:16:51,471 P48837 INFO [Metrics] AUC: 0.717410 - logloss: 0.417300
2021-09-04 09:16:51,474 P48837 INFO Save best model: monitor(max): 0.717410
2021-09-04 09:16:52,373 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:16:52,469 P48837 INFO Train loss: 0.464866
2021-09-04 09:16:52,469 P48837 INFO ************ Epoch=1 end ************
2021-09-04 09:20:48,025 P48837 INFO [Metrics] AUC: 0.722127 - logloss: 0.413633
2021-09-04 09:20:48,027 P48837 INFO Save best model: monitor(max): 0.722127
2021-09-04 09:20:50,028 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:20:50,132 P48837 INFO Train loss: 0.443117
2021-09-04 09:20:50,132 P48837 INFO ************ Epoch=2 end ************
2021-09-04 09:25:05,814 P48837 INFO [Metrics] AUC: 0.723892 - logloss: 0.414505
2021-09-04 09:25:05,815 P48837 INFO Save best model: monitor(max): 0.723892
2021-09-04 09:25:07,791 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:25:07,937 P48837 INFO Train loss: 0.441370
2021-09-04 09:25:07,937 P48837 INFO ************ Epoch=3 end ************
2021-09-04 09:29:23,342 P48837 INFO [Metrics] AUC: 0.724925 - logloss: 0.414444
2021-09-04 09:29:23,344 P48837 INFO Save best model: monitor(max): 0.724925
2021-09-04 09:29:24,660 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:29:24,817 P48837 INFO Train loss: 0.440466
2021-09-04 09:29:24,817 P48837 INFO ************ Epoch=4 end ************
2021-09-04 09:33:39,751 P48837 INFO [Metrics] AUC: 0.726028 - logloss: 0.412581
2021-09-04 09:33:39,753 P48837 INFO Save best model: monitor(max): 0.726028
2021-09-04 09:33:41,760 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:33:41,961 P48837 INFO Train loss: 0.439698
2021-09-04 09:33:41,961 P48837 INFO ************ Epoch=5 end ************
2021-09-04 09:37:59,950 P48837 INFO [Metrics] AUC: 0.726702 - logloss: 0.412650
2021-09-04 09:37:59,951 P48837 INFO Save best model: monitor(max): 0.726702
2021-09-04 09:38:02,539 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:38:02,797 P48837 INFO Train loss: 0.438910
2021-09-04 09:38:02,797 P48837 INFO ************ Epoch=6 end ************
2021-09-04 09:42:08,436 P48837 INFO [Metrics] AUC: 0.726190 - logloss: 0.412926
2021-09-04 09:42:08,438 P48837 INFO Monitor(max) STOP: 0.726190 !
2021-09-04 09:42:08,438 P48837 INFO Reduce learning rate on plateau: 0.000050
2021-09-04 09:42:08,438 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:42:08,601 P48837 INFO Train loss: 0.438218
2021-09-04 09:42:08,601 P48837 INFO ************ Epoch=7 end ************
2021-09-04 09:46:28,100 P48837 INFO [Metrics] AUC: 0.733625 - logloss: 0.410175
2021-09-04 09:46:28,101 P48837 INFO Save best model: monitor(max): 0.733625
2021-09-04 09:46:30,088 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:46:30,230 P48837 INFO Train loss: 0.426682
2021-09-04 09:46:30,230 P48837 INFO ************ Epoch=8 end ************
2021-09-04 09:50:46,000 P48837 INFO [Metrics] AUC: 0.734400 - logloss: 0.410069
2021-09-04 09:50:46,001 P48837 INFO Save best model: monitor(max): 0.734400
2021-09-04 09:50:47,982 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:50:48,127 P48837 INFO Train loss: 0.421900
2021-09-04 09:50:48,127 P48837 INFO ************ Epoch=9 end ************
2021-09-04 09:55:02,440 P48837 INFO [Metrics] AUC: 0.734631 - logloss: 0.410429
2021-09-04 09:55:02,441 P48837 INFO Save best model: monitor(max): 0.734631
2021-09-04 09:55:04,381 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:55:04,533 P48837 INFO Train loss: 0.419930
2021-09-04 09:55:04,533 P48837 INFO ************ Epoch=10 end ************
2021-09-04 09:59:18,546 P48837 INFO [Metrics] AUC: 0.734666 - logloss: 0.410401
2021-09-04 09:59:18,548 P48837 INFO Save best model: monitor(max): 0.734666
2021-09-04 09:59:20,539 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 09:59:20,700 P48837 INFO Train loss: 0.418365
2021-09-04 09:59:20,701 P48837 INFO ************ Epoch=11 end ************
2021-09-04 10:03:32,770 P48837 INFO [Metrics] AUC: 0.734781 - logloss: 0.410982
2021-09-04 10:03:32,771 P48837 INFO Save best model: monitor(max): 0.734781
2021-09-04 10:03:34,738 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 10:03:34,841 P48837 INFO Train loss: 0.417028
2021-09-04 10:03:34,841 P48837 INFO ************ Epoch=12 end ************
2021-09-04 10:07:47,740 P48837 INFO [Metrics] AUC: 0.734755 - logloss: 0.410453
2021-09-04 10:07:47,741 P48837 INFO Monitor(max) STOP: 0.734755 !
2021-09-04 10:07:47,741 P48837 INFO Reduce learning rate on plateau: 0.000005
2021-09-04 10:07:47,741 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 10:07:47,885 P48837 INFO Train loss: 0.416008
2021-09-04 10:07:47,885 P48837 INFO ************ Epoch=13 end ************
2021-09-04 10:12:02,085 P48837 INFO [Metrics] AUC: 0.734600 - logloss: 0.412334
2021-09-04 10:12:02,087 P48837 INFO Monitor(max) STOP: 0.734600 !
2021-09-04 10:12:02,087 P48837 INFO Reduce learning rate on plateau: 0.000001
2021-09-04 10:12:02,087 P48837 INFO Early stopping at epoch=14
2021-09-04 10:12:02,087 P48837 INFO --- 4381/4381 batches finished ---
2021-09-04 10:12:02,216 P48837 INFO Train loss: 0.410749
2021-09-04 10:12:02,216 P48837 INFO Training finished.
2021-09-04 10:12:02,216 P48837 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/DeepIM_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/DeepIM_microvideo_1.7m_x0_008_3d20f1b6.model
2021-09-04 10:12:02,883 P48837 INFO ****** Train/validation evaluation ******
2021-09-04 10:12:36,288 P48837 INFO [Metrics] AUC: 0.734781 - logloss: 0.410982
2021-09-04 10:12:36,349 P48837 INFO ******** Test evaluation ********
2021-09-04 10:12:36,349 P48837 INFO Loading data...
2021-09-04 10:12:36,350 P48837 INFO Loading test data done.


```
