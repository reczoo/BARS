## DeepFM_Taobao_x0_001

A notebook to benchmark DeepFM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193661 - AUC: 0.642694
```


### Logs
```python
2020-06-23 18:31:57,589 P45438 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[500, 500, 500]",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DeepFM",
    "model_id": "DeepFM_taobao_x0_010_337344fb",
    "model_root": "./Taobao/DeepFM_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
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
2020-06-23 18:31:57,590 P45438 INFO Set up feature encoder...
2020-06-23 18:31:57,590 P45438 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-23 18:31:58,438 P45438 INFO Total number of parameters: 19488181.
2020-06-23 18:31:58,439 P45438 INFO Loading data...
2020-06-23 18:31:58,441 P45438 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-23 18:32:04,707 P45438 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-23 18:32:06,300 P45438 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-23 18:32:06,382 P45438 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-23 18:32:06,382 P45438 INFO Loading train data done.
2020-06-23 18:32:11,901 P45438 INFO Start training: 2371 batches/epoch
2020-06-23 18:32:11,901 P45438 INFO ************ Epoch=1 start ************
2020-06-23 18:58:57,003 P45438 INFO [Metrics] logloss: 0.193661 - AUC: 0.642694
2020-06-23 18:58:57,007 P45438 INFO Save best model: monitor(max): 0.449033
2020-06-23 18:58:57,078 P45438 INFO --- 2371/2371 batches finished ---
2020-06-23 18:58:57,126 P45438 INFO Train loss: 0.203362
2020-06-23 18:58:57,126 P45438 INFO ************ Epoch=1 end ************
2020-06-23 19:25:45,373 P45438 INFO [Metrics] logloss: 0.196023 - AUC: 0.637685
2020-06-23 19:25:45,376 P45438 INFO Monitor(max) STOP: 0.441662 !
2020-06-23 19:25:45,376 P45438 INFO Reduce learning rate on plateau: 0.000100
2020-06-23 19:25:45,376 P45438 INFO --- 2371/2371 batches finished ---
2020-06-23 19:25:45,425 P45438 INFO Train loss: 0.196348
2020-06-23 19:25:45,425 P45438 INFO ************ Epoch=2 end ************
2020-06-23 19:52:12,083 P45438 INFO [Metrics] logloss: 0.215067 - AUC: 0.617916
2020-06-23 19:52:12,086 P45438 INFO Monitor(max) STOP: 0.402849 !
2020-06-23 19:52:12,086 P45438 INFO Reduce learning rate on plateau: 0.000010
2020-06-23 19:52:12,087 P45438 INFO Early stopping at epoch=3
2020-06-23 19:52:12,087 P45438 INFO --- 2371/2371 batches finished ---
2020-06-23 19:52:12,135 P45438 INFO Train loss: 0.176176
2020-06-23 19:52:12,135 P45438 INFO Training finished.
2020-06-23 19:52:12,135 P45438 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/DeepFM_taobao/taobao_x0_87391c5c/DeepFM_taobao_x0_010_337344fb_model.ckpt
2020-06-23 19:52:12,584 P45438 INFO ****** Train/validation evaluation ******
2020-06-23 19:52:54,384 P45438 INFO [Metrics] logloss: 0.193661 - AUC: 0.642694
2020-06-23 19:52:54,428 P45438 INFO ******** Test evaluation ********
2020-06-23 19:52:54,429 P45438 INFO Loading data...
2020-06-23 19:52:54,429 P45438 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-23 19:52:55,280 P45438 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-23 19:52:55,280 P45438 INFO Loading test data done.
2020-06-23 19:53:37,003 P45438 INFO [Metrics] logloss: 0.193661 - AUC: 0.642694

```