## xDeepFM_Taobao_x0_001 

A notebook to benchmark xDeepFM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193562 - AUC: 0.643633
```


### Logs
```python
2020-06-24 05:03:59,497 P2227 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[32, 32]",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "dnn_hidden_units": "[500, 500, 500]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "xDeepFM",
    "model_id": "xDeepFM_taobao_x0_015_c4e2ac08",
    "model_root": "./Taobao/xDeepFM_taobao/",
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
2020-06-24 05:03:59,498 P2227 INFO Set up feature encoder...
2020-06-24 05:03:59,498 P2227 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-24 05:04:00,385 P2227 INFO Total number of parameters: 23645030.
2020-06-24 05:04:00,385 P2227 INFO Loading data...
2020-06-24 05:04:00,387 P2227 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-24 05:04:06,962 P2227 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-24 05:04:08,419 P2227 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-24 05:04:08,501 P2227 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-24 05:04:08,501 P2227 INFO Loading train data done.
2020-06-24 05:04:11,814 P2227 INFO Start training: 2371 batches/epoch
2020-06-24 05:04:11,814 P2227 INFO ************ Epoch=1 start ************
2020-06-24 05:17:33,382 P2227 INFO [Metrics] logloss: 0.193562 - AUC: 0.643633
2020-06-24 05:17:33,384 P2227 INFO Save best model: monitor(max): 0.450071
2020-06-24 05:17:33,463 P2227 INFO --- 2371/2371 batches finished ---
2020-06-24 05:17:33,510 P2227 INFO Train loss: 0.202413
2020-06-24 05:17:33,510 P2227 INFO ************ Epoch=1 end ************
2020-06-24 05:32:54,498 P2227 INFO [Metrics] logloss: 0.196339 - AUC: 0.635261
2020-06-24 05:32:54,500 P2227 INFO Monitor(max) STOP: 0.438923 !
2020-06-24 05:32:54,500 P2227 INFO Reduce learning rate on plateau: 0.000100
2020-06-24 05:32:54,500 P2227 INFO --- 2371/2371 batches finished ---
2020-06-24 05:32:54,550 P2227 INFO Train loss: 0.195840
2020-06-24 05:32:54,550 P2227 INFO ************ Epoch=2 end ************
2020-06-24 05:49:43,192 P2227 INFO [Metrics] logloss: 0.210136 - AUC: 0.622924
2020-06-24 05:49:43,196 P2227 INFO Monitor(max) STOP: 0.412788 !
2020-06-24 05:49:43,196 P2227 INFO Reduce learning rate on plateau: 0.000010
2020-06-24 05:49:43,196 P2227 INFO Early stopping at epoch=3
2020-06-24 05:49:43,196 P2227 INFO --- 2371/2371 batches finished ---
2020-06-24 05:49:43,245 P2227 INFO Train loss: 0.179521
2020-06-24 05:49:43,245 P2227 INFO Training finished.
2020-06-24 05:49:43,245 P2227 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/xDeepFM_taobao/taobao_x0_87391c5c/xDeepFM_taobao_x0_015_c4e2ac08_model.ckpt
2020-06-24 05:49:43,526 P2227 INFO ****** Train/validation evaluation ******
2020-06-24 05:50:10,552 P2227 INFO [Metrics] logloss: 0.193562 - AUC: 0.643633
2020-06-24 05:50:10,597 P2227 INFO ******** Test evaluation ********
2020-06-24 05:50:10,598 P2227 INFO Loading data...
2020-06-24 05:50:10,598 P2227 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-24 05:50:11,622 P2227 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-24 05:50:11,622 P2227 INFO Loading test data done.
2020-06-24 05:50:38,526 P2227 INFO [Metrics] logloss: 0.193562 - AUC: 0.643633

```
