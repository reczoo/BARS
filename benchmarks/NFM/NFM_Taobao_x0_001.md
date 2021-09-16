## NFM_Taobao_x0_001 

A notebook to benchmark NFM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.195566 - AUC: 0.639169
```


### Logs
```python
2020-06-29 09:56:00,056 P11417 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "5e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[500, 500, 500]",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "NFM",
    "model_id": "NFM_taobao_x0_001_ff47d0dd",
    "model_root": "./Taobao/NFM_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/Taobao_x0/test.csv",
    "train_data": "../data/Taobao/Taobao_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Taobao/Taobao_x0/test.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-29 09:56:00,057 P11417 INFO Set up feature encoder...
2020-06-29 09:56:00,057 P11417 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-29 09:56:01,118 P11417 INFO Total number of parameters: 23500325.
2020-06-29 09:56:01,118 P11417 INFO Loading data...
2020-06-29 09:56:01,120 P11417 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-29 09:56:07,881 P11417 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-29 09:56:09,666 P11417 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-29 09:56:09,807 P11417 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-29 09:56:09,807 P11417 INFO Loading train data done.
2020-06-29 09:56:12,885 P11417 INFO Start training: 2371 batches/epoch
2020-06-29 09:56:12,885 P11417 INFO ************ Epoch=1 start ************
2020-06-29 10:00:58,760 P11417 INFO [Metrics] logloss: 0.196576 - AUC: 0.608535
2020-06-29 10:00:58,763 P11417 INFO Save best model: monitor(max): 0.411959
2020-06-29 10:00:58,855 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:00:58,909 P11417 INFO Train loss: 0.206176
2020-06-29 10:00:58,909 P11417 INFO ************ Epoch=1 end ************
2020-06-29 10:05:40,027 P11417 INFO [Metrics] logloss: 0.196016 - AUC: 0.612866
2020-06-29 10:05:40,031 P11417 INFO Save best model: monitor(max): 0.416850
2020-06-29 10:05:40,215 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:05:40,285 P11417 INFO Train loss: 0.204054
2020-06-29 10:05:40,285 P11417 INFO ************ Epoch=2 end ************
2020-06-29 10:10:24,652 P11417 INFO [Metrics] logloss: 0.195596 - AUC: 0.615142
2020-06-29 10:10:24,655 P11417 INFO Save best model: monitor(max): 0.419546
2020-06-29 10:10:24,836 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:10:24,895 P11417 INFO Train loss: 0.203923
2020-06-29 10:10:24,895 P11417 INFO ************ Epoch=3 end ************
2020-06-29 10:15:11,754 P11417 INFO [Metrics] logloss: 0.195813 - AUC: 0.616800
2020-06-29 10:15:11,758 P11417 INFO Save best model: monitor(max): 0.420988
2020-06-29 10:15:11,928 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:15:11,997 P11417 INFO Train loss: 0.203759
2020-06-29 10:15:11,997 P11417 INFO ************ Epoch=4 end ************
2020-06-29 10:19:59,421 P11417 INFO [Metrics] logloss: 0.195801 - AUC: 0.616289
2020-06-29 10:19:59,424 P11417 INFO Monitor(max) STOP: 0.420488 !
2020-06-29 10:19:59,424 P11417 INFO Reduce learning rate on plateau: 0.000100
2020-06-29 10:19:59,425 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:19:59,481 P11417 INFO Train loss: 0.203738
2020-06-29 10:19:59,481 P11417 INFO ************ Epoch=5 end ************
2020-06-29 10:24:36,496 P11417 INFO [Metrics] logloss: 0.195151 - AUC: 0.633212
2020-06-29 10:24:36,499 P11417 INFO Save best model: monitor(max): 0.438061
2020-06-29 10:24:36,655 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:24:36,705 P11417 INFO Train loss: 0.197252
2020-06-29 10:24:36,705 P11417 INFO ************ Epoch=6 end ************
2020-06-29 10:29:19,112 P11417 INFO [Metrics] logloss: 0.195142 - AUC: 0.638484
2020-06-29 10:29:19,114 P11417 INFO Save best model: monitor(max): 0.443342
2020-06-29 10:29:19,264 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:29:19,326 P11417 INFO Train loss: 0.193278
2020-06-29 10:29:19,326 P11417 INFO ************ Epoch=7 end ************
2020-06-29 10:33:58,012 P11417 INFO [Metrics] logloss: 0.195566 - AUC: 0.639169
2020-06-29 10:33:58,014 P11417 INFO Save best model: monitor(max): 0.443603
2020-06-29 10:33:58,175 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:33:58,225 P11417 INFO Train loss: 0.190942
2020-06-29 10:33:58,225 P11417 INFO ************ Epoch=8 end ************
2020-06-29 10:38:39,520 P11417 INFO [Metrics] logloss: 0.195723 - AUC: 0.639148
2020-06-29 10:38:39,521 P11417 INFO Monitor(max) STOP: 0.443425 !
2020-06-29 10:38:39,522 P11417 INFO Reduce learning rate on plateau: 0.000010
2020-06-29 10:38:39,522 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:38:39,580 P11417 INFO Train loss: 0.189598
2020-06-29 10:38:39,580 P11417 INFO ************ Epoch=9 end ************
2020-06-29 10:43:21,979 P11417 INFO [Metrics] logloss: 0.199390 - AUC: 0.638528
2020-06-29 10:43:21,980 P11417 INFO Monitor(max) STOP: 0.439138 !
2020-06-29 10:43:21,981 P11417 INFO Reduce learning rate on plateau: 0.000001
2020-06-29 10:43:21,981 P11417 INFO Early stopping at epoch=10
2020-06-29 10:43:21,981 P11417 INFO --- 2371/2371 batches finished ---
2020-06-29 10:43:22,040 P11417 INFO Train loss: 0.183868
2020-06-29 10:43:22,040 P11417 INFO Training finished.
2020-06-29 10:43:22,040 P11417 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/NFM_taobao/taobao_x0_87391c5c/NFM_taobao_x0_001_ff47d0dd_model.ckpt
2020-06-29 10:43:22,195 P11417 INFO ****** Train/validation evaluation ******
2020-06-29 10:43:42,676 P11417 INFO [Metrics] logloss: 0.195566 - AUC: 0.639169
2020-06-29 10:43:42,717 P11417 INFO ******** Test evaluation ********
2020-06-29 10:43:42,717 P11417 INFO Loading data...
2020-06-29 10:43:42,717 P11417 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-29 10:43:43,565 P11417 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-29 10:43:43,565 P11417 INFO Loading test data done.
2020-06-29 10:44:03,208 P11417 INFO [Metrics] logloss: 0.195566 - AUC: 0.639169

```
