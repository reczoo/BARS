## FwFM_microvideo1.7m_x0_001

A notebook to benchmark FwFM on microvideo1.7m_x0_001 dataset.

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
[Metrics] AUC: 0.725914 - logloss: 0.415049
```


### Logs
```python
2021-09-08 14:21:31,609 P56627 INFO {
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
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.0005",
    "linear_type": "FeLV",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FwFM",
    "model_id": "FwFM_microvideo_1.7m_x0_006_c5277cb0",
    "model_root": "./MicroVideo/FwFM_microvideo_1.7m_x0/",
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
2021-09-08 14:21:31,610 P56627 INFO Set up feature encoder...
2021-09-08 14:21:31,610 P56627 INFO Load feature_encoder from pickle: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/feature_encoder.pkl
2021-09-08 14:21:44,540 P56627 INFO Total number of parameters: 1488523.
2021-09-08 14:21:44,541 P56627 INFO Loading data...
2021-09-08 14:21:44,543 P56627 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/train.h5
2021-09-08 14:21:57,908 P56627 INFO Train samples: total/8970309, blocks/1
2021-09-08 14:21:57,908 P56627 INFO Loading data from h5: ../data/MicroVideo/microvideo_1.7m_x0_710d1f85/valid.h5
2021-09-08 14:22:03,564 P56627 INFO Validation samples: total/3767308, blocks/1%
2021-09-08 14:22:03,564 P56627 INFO Loading train data done.
2021-09-08 14:22:03,564 P56627 INFO Start training: 4381 batches/epoch
2021-09-08 14:22:03,564 P56627 INFO ************ Epoch=1 start ************
2021-09-08 14:24:09,408 P56627 INFO [Metrics] AUC: 0.697697 - logloss: 0.427986
2021-09-08 14:24:09,409 P56627 INFO Save best model: monitor(max): 0.697697
2021-09-08 14:24:11,299 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:24:11,384 P56627 INFO Train loss: 0.534883
2021-09-08 14:24:11,384 P56627 INFO ************ Epoch=1 end ************
2021-09-08 14:26:15,988 P56627 INFO [Metrics] AUC: 0.706211 - logloss: 0.425192
2021-09-08 14:26:15,990 P56627 INFO Save best model: monitor(max): 0.706211
2021-09-08 14:26:42,547 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:26:42,633 P56627 INFO Train loss: 0.457513
2021-09-08 14:26:42,633 P56627 INFO ************ Epoch=2 end ************
2021-09-08 14:28:47,000 P56627 INFO [Metrics] AUC: 0.707229 - logloss: 0.424411
2021-09-08 14:28:47,003 P56627 INFO Save best model: monitor(max): 0.707229
2021-09-08 14:29:12,811 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:29:12,895 P56627 INFO Train loss: 0.453294
2021-09-08 14:29:12,895 P56627 INFO ************ Epoch=3 end ************
2021-09-08 14:31:18,014 P56627 INFO [Metrics] AUC: 0.706774 - logloss: 0.424414
2021-09-08 14:31:18,019 P56627 INFO Monitor(max) STOP: 0.706774 !
2021-09-08 14:31:18,019 P56627 INFO Reduce learning rate on plateau: 0.000050
2021-09-08 14:31:18,019 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:31:18,101 P56627 INFO Train loss: 0.450215
2021-09-08 14:31:18,101 P56627 INFO ************ Epoch=4 end ************
2021-09-08 14:33:23,945 P56627 INFO [Metrics] AUC: 0.723460 - logloss: 0.415352
2021-09-08 14:33:23,949 P56627 INFO Save best model: monitor(max): 0.723460
2021-09-08 14:33:51,450 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:33:51,537 P56627 INFO Train loss: 0.429045
2021-09-08 14:33:51,537 P56627 INFO ************ Epoch=5 end ************
2021-09-08 14:35:58,512 P56627 INFO [Metrics] AUC: 0.724857 - logloss: 0.414604
2021-09-08 14:35:58,515 P56627 INFO Save best model: monitor(max): 0.724857
2021-09-08 14:36:25,731 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:36:25,817 P56627 INFO Train loss: 0.424629
2021-09-08 14:36:25,817 P56627 INFO ************ Epoch=6 end ************
2021-09-08 14:38:33,195 P56627 INFO [Metrics] AUC: 0.724951 - logloss: 0.414889
2021-09-08 14:38:33,198 P56627 INFO Save best model: monitor(max): 0.724951
2021-09-08 14:39:01,132 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:39:01,218 P56627 INFO Train loss: 0.422193
2021-09-08 14:39:01,218 P56627 INFO ************ Epoch=7 end ************
2021-09-08 14:41:08,803 P56627 INFO [Metrics] AUC: 0.725161 - logloss: 0.414906
2021-09-08 14:41:08,808 P56627 INFO Save best model: monitor(max): 0.725161
2021-09-08 14:41:50,450 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:41:50,535 P56627 INFO Train loss: 0.420210
2021-09-08 14:41:50,535 P56627 INFO ************ Epoch=8 end ************
2021-09-08 14:43:57,829 P56627 INFO [Metrics] AUC: 0.724687 - logloss: 0.415690
2021-09-08 14:43:57,833 P56627 INFO Monitor(max) STOP: 0.724687 !
2021-09-08 14:43:57,833 P56627 INFO Reduce learning rate on plateau: 0.000005
2021-09-08 14:43:57,833 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:43:57,915 P56627 INFO Train loss: 0.418532
2021-09-08 14:43:57,915 P56627 INFO ************ Epoch=9 end ************
2021-09-08 14:46:05,203 P56627 INFO [Metrics] AUC: 0.725575 - logloss: 0.415153
2021-09-08 14:46:05,207 P56627 INFO Save best model: monitor(max): 0.725575
2021-09-08 14:46:32,159 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:46:32,261 P56627 INFO Train loss: 0.413974
2021-09-08 14:46:32,261 P56627 INFO ************ Epoch=10 end ************
2021-09-08 14:48:37,319 P56627 INFO [Metrics] AUC: 0.725897 - logloss: 0.415045
2021-09-08 14:48:37,323 P56627 INFO Save best model: monitor(max): 0.725897
2021-09-08 14:49:05,457 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:49:05,543 P56627 INFO Train loss: 0.413633
2021-09-08 14:49:05,544 P56627 INFO ************ Epoch=11 end ************
2021-09-08 14:51:12,538 P56627 INFO [Metrics] AUC: 0.725914 - logloss: 0.415049
2021-09-08 14:51:12,542 P56627 INFO Save best model: monitor(max): 0.725914
2021-09-08 14:51:42,710 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:51:42,795 P56627 INFO Train loss: 0.413402
2021-09-08 14:51:42,795 P56627 INFO ************ Epoch=12 end ************
2021-09-08 14:53:49,776 P56627 INFO [Metrics] AUC: 0.725824 - logloss: 0.415114
2021-09-08 14:53:49,781 P56627 INFO Monitor(max) STOP: 0.725824 !
2021-09-08 14:53:49,781 P56627 INFO Reduce learning rate on plateau: 0.000001
2021-09-08 14:53:49,781 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:53:49,887 P56627 INFO Train loss: 0.413222
2021-09-08 14:53:49,887 P56627 INFO ************ Epoch=13 end ************
2021-09-08 14:55:55,608 P56627 INFO [Metrics] AUC: 0.725853 - logloss: 0.415082
2021-09-08 14:55:55,612 P56627 INFO Monitor(max) STOP: 0.725853 !
2021-09-08 14:55:55,613 P56627 INFO Reduce learning rate on plateau: 0.000001
2021-09-08 14:55:55,613 P56627 INFO Early stopping at epoch=14
2021-09-08 14:55:55,613 P56627 INFO --- 4381/4381 batches finished ---
2021-09-08 14:55:55,697 P56627 INFO Train loss: 0.412727
2021-09-08 14:55:55,697 P56627 INFO Training finished.
2021-09-08 14:55:55,697 P56627 INFO Load best model: /home/ma-user/work/FuxiCTR/benchmark/MicroVideo/FwFM_microvideo_1.7m_x0/microvideo_1.7m_x0_710d1f85/FwFM_microvideo_1.7m_x0_006_c5277cb0.model
2021-09-08 14:55:56,777 P56627 INFO ****** Train/validation evaluation ******
2021-09-08 14:56:23,764 P56627 INFO [Metrics] AUC: 0.725914 - logloss: 0.415049
2021-09-08 14:56:23,878 P56627 INFO ******** Test evaluation ********
2021-09-08 14:56:23,879 P56627 INFO Loading data...
2021-09-08 14:56:23,879 P56627 INFO Loading test data done.

```
