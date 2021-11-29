## xDeepFM_microvideo1.7m_x0_001

A notebook to benchmark xDeepFM on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.736236 - logloss: 0.410663
```


### Logs
```python
2021-09-03 16:18:55,889 P39042 INFO {
    "batch_norm": "True",
    "batch_size": "2048",
    "cin_hidden_units": "[64, 32]",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/MicroVideo/",
    "dataset_id": "microvideo_1.7m_x0_710d1f85",
    "debug_mode": "False",
    "dnn_hidden_units": "[1024, 512, 256]",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "eval_interval": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': 'nn.Linear(64, 64, bias=False)', 'name': 'item_id', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'embedding_hook': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'max_len': 128, 'name': 'clicked_items', 'padding': 'pre', 'pretrained_emb': '../data/MicroVideo/MicroVideo_1.7M_x0/item_image_emb_dim64.h5', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_hook': 'layers.MaskedAveragePooling()', 'max_len': 128, 'name': 'clicked_categories', 'padding': 'pre', 'share_embedding': 'cate_id', 'splitter': '^', 'type': 'sequence'}, {'active': False, 'dtype': 'str', 'name': 'timestamp', 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_microvideo_1.7m_x0_001_3a5bfa86",
    "model_root": "./MicroVideo/xDeepFM_microvideo_1.7m_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
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
2021-09-03 16:18:55,890 P39042 INFO Set up feature encoder...
2021-09-03 16:18:55,890 P39042 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-09-03 16:19:10,714 P39042 INFO Total number of parameters: 5166740.
2021-09-03 16:19:10,715 P39042 INFO Loading data...
2021-09-03 16:19:10,720 P39042 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-09-03 16:20:48,892 P39042 INFO Train samples: total/8970309, blocks/1
2021-09-03 16:20:48,892 P39042 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-09-03 16:21:43,123 P39042 INFO Validation samples: total/3767308, blocks/1%
2021-09-03 16:21:43,124 P39042 INFO Loading train data done.
2021-09-03 16:21:43,124 P39042 INFO Start training: 4381 batches/epoch
2021-09-03 16:21:43,124 P39042 INFO ************ Epoch=1 start ************
2021-09-03 16:36:42,312 P39042 INFO [Metrics] AUC: 0.717730 - logloss: 0.415699
2021-09-03 16:36:42,322 P39042 INFO Save best model: monitor(max): 0.717730
2021-09-03 16:36:44,288 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 16:36:44,662 P39042 INFO Train loss: 0.465957
2021-09-03 16:36:44,662 P39042 INFO ************ Epoch=1 end ************
2021-09-03 16:51:49,665 P39042 INFO [Metrics] AUC: 0.721484 - logloss: 0.414940
2021-09-03 16:51:49,667 P39042 INFO Save best model: monitor(max): 0.721484
2021-09-03 16:52:21,372 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 16:52:21,724 P39042 INFO Train loss: 0.449230
2021-09-03 16:52:21,724 P39042 INFO ************ Epoch=2 end ************
2021-09-03 17:07:27,497 P39042 INFO [Metrics] AUC: 0.723888 - logloss: 0.413890
2021-09-03 17:07:27,498 P39042 INFO Save best model: monitor(max): 0.723888
2021-09-03 17:07:48,130 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 17:07:48,474 P39042 INFO Train loss: 0.445687
2021-09-03 17:07:48,474 P39042 INFO ************ Epoch=3 end ************
2021-09-03 17:22:43,598 P39042 INFO [Metrics] AUC: 0.725898 - logloss: 0.413580
2021-09-03 17:22:43,599 P39042 INFO Save best model: monitor(max): 0.725898
2021-09-03 17:22:57,301 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 17:22:57,574 P39042 INFO Train loss: 0.443627
2021-09-03 17:22:57,574 P39042 INFO ************ Epoch=4 end ************
2021-09-03 17:38:04,702 P39042 INFO [Metrics] AUC: 0.725867 - logloss: 0.414071
2021-09-03 17:38:04,703 P39042 INFO Monitor(max) STOP: 0.725867 !
2021-09-03 17:38:04,703 P39042 INFO Reduce learning rate on plateau: 0.000050
2021-09-03 17:38:04,704 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 17:38:05,046 P39042 INFO Train loss: 0.441979
2021-09-03 17:38:05,048 P39042 INFO ************ Epoch=5 end ************
2021-09-03 17:53:11,047 P39042 INFO [Metrics] AUC: 0.734779 - logloss: 0.410274
2021-09-03 17:53:11,048 P39042 INFO Save best model: monitor(max): 0.734779
2021-09-03 17:53:25,442 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 17:53:25,729 P39042 INFO Train loss: 0.423832
2021-09-03 17:53:25,729 P39042 INFO ************ Epoch=6 end ************
2021-09-03 18:07:23,777 P39042 INFO [Metrics] AUC: 0.736236 - logloss: 0.410663
2021-09-03 18:07:23,778 P39042 INFO Save best model: monitor(max): 0.736236
2021-09-03 18:07:35,088 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 18:07:35,374 P39042 INFO Train loss: 0.415775
2021-09-03 18:07:35,374 P39042 INFO ************ Epoch=7 end ************
2021-09-03 18:18:26,363 P39042 INFO [Metrics] AUC: 0.735709 - logloss: 0.412432
2021-09-03 18:18:26,364 P39042 INFO Monitor(max) STOP: 0.735709 !
2021-09-03 18:18:26,365 P39042 INFO Reduce learning rate on plateau: 0.000005
2021-09-03 18:18:26,365 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 18:18:26,634 P39042 INFO Train loss: 0.411295
2021-09-03 18:18:26,634 P39042 INFO ************ Epoch=8 end ************
2021-09-03 18:28:48,658 P39042 INFO [Metrics] AUC: 0.735397 - logloss: 0.414420
2021-09-03 18:28:48,659 P39042 INFO Monitor(max) STOP: 0.735397 !
2021-09-03 18:28:48,659 P39042 INFO Reduce learning rate on plateau: 0.000001
2021-09-03 18:28:48,659 P39042 INFO Early stopping at epoch=9
2021-09-03 18:28:48,659 P39042 INFO --- 4381/4381 batches finished ---
2021-09-03 18:28:48,911 P39042 INFO Train loss: 0.399945
2021-09-03 18:28:48,911 P39042 INFO Training finished.
2021-09-03 18:28:48,912 P39042 INFO Load best model: /home/xxx/xxx/GroupCTR/benchmark/MicroVideo/xDeepFM_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/xDeepFM_microvideo_1.7m_x0_001_3a5bfa86.model
2021-09-03 18:28:49,511 P39042 INFO ****** Train/validation evaluation ******
2021-09-03 18:29:15,524 P39042 INFO [Metrics] AUC: 0.736236 - logloss: 0.410663
2021-09-03 18:29:15,573 P39042 INFO ******** Test evaluation ********
2021-09-03 18:29:15,573 P39042 INFO Loading data...
2021-09-03 18:29:15,574 P39042 INFO Loading test data done.



```
