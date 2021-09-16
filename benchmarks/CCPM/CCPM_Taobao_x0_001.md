## CCPM_Taobao_x0_001

A notebook to benchmark CCPM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193910 - AUC: 0.643164
```


### Logs
```python
2020-07-08 15:59:11,534 P23667 INFO {
    "activation": "Tanh",
    "batch_size": "5000",
    "channels": "[64, 128, 256]",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "0",
    "kernel_heights": "[7, 5, 3]",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "CCPM",
    "model_id": "CCPM_taobao_x0_004_d711aad6",
    "model_root": "./Taobao/CCPM_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
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
2020-07-08 15:59:11,535 P23667 INFO Set up feature encoder...
2020-07-08 15:59:11,535 P23667 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-07-08 15:59:12,693 P23667 INFO Total number of parameters: 21790401.
2020-07-08 15:59:12,694 P23667 INFO Loading data...
2020-07-08 15:59:12,698 P23667 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-07-08 15:59:23,358 P23667 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-08 15:59:25,934 P23667 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-07-08 15:59:26,128 P23667 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-08 15:59:26,128 P23667 INFO Loading train data done.
2020-07-08 15:59:30,449 P23667 INFO Start training: 4742 batches/epoch
2020-07-08 15:59:30,449 P23667 INFO ************ Epoch=1 start ************
2020-07-08 17:34:08,742 P23667 INFO [Metrics] logloss: 0.193910 - AUC: 0.643164
2020-07-08 17:34:08,748 P23667 INFO Save best model: monitor(max): 0.449253
2020-07-08 17:34:08,875 P23667 INFO --- 4742/4742 batches finished ---
2020-07-08 17:34:08,948 P23667 INFO Train loss: 0.200581
2020-07-08 17:34:08,949 P23667 INFO ************ Epoch=1 end ************
2020-07-08 19:07:43,392 P23667 INFO [Metrics] logloss: 0.195000 - AUC: 0.641444
2020-07-08 19:07:43,396 P23667 INFO Monitor(max) STOP: 0.446445 !
2020-07-08 19:07:43,397 P23667 INFO Reduce learning rate on plateau: 0.000100
2020-07-08 19:07:43,397 P23667 INFO --- 4742/4742 batches finished ---
2020-07-08 19:07:43,480 P23667 INFO Train loss: 0.194643
2020-07-08 19:07:43,480 P23667 INFO ************ Epoch=2 end ************
2020-07-08 20:41:32,122 P23667 INFO [Metrics] logloss: 0.210106 - AUC: 0.617734
2020-07-08 20:41:32,125 P23667 INFO Monitor(max) STOP: 0.407628 !
2020-07-08 20:41:32,125 P23667 INFO Reduce learning rate on plateau: 0.000010
2020-07-08 20:41:32,125 P23667 INFO Early stopping at epoch=3
2020-07-08 20:41:32,125 P23667 INFO --- 4742/4742 batches finished ---
2020-07-08 20:41:32,237 P23667 INFO Train loss: 0.174019
2020-07-08 20:41:32,237 P23667 INFO Training finished.
2020-07-08 20:41:32,237 P23667 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/CCPM_taobao/taobao_x0_87391c5c/CCPM_taobao_x0_004_d711aad6_model.ckpt
2020-07-08 20:41:32,414 P23667 INFO ****** Train/validation evaluation ******
2020-07-08 20:50:15,475 P23667 INFO [Metrics] logloss: 0.193910 - AUC: 0.643164
2020-07-08 20:50:15,548 P23667 INFO ******** Test evaluation ********
2020-07-08 20:50:15,548 P23667 INFO Loading data...
2020-07-08 20:50:15,548 P23667 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-08 20:50:17,010 P23667 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-08 20:50:17,010 P23667 INFO Loading test data done.
2020-07-08 20:58:59,994 P23667 INFO [Metrics] logloss: 0.193910 - AUC: 0.643164

```
