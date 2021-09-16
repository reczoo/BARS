## FwFM_Taobao_x0_001

A notebook to benchmark FwFM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193561 - AUC: 0.642748
```


### Logs
```python
2020-06-26 09:21:07,935 P7580 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "16",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "linear_type": "FiLV",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FwFM",
    "model_id": "FwFM_taobao_x0_009_ac0c8212",
    "model_root": "./Taobao/FwFM_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-07",
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
2020-06-26 09:21:07,936 P7580 INFO Set up feature encoder...
2020-06-26 09:21:07,936 P7580 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-26 09:21:08,791 P7580 INFO Total number of parameters: 21638329.
2020-06-26 09:21:08,791 P7580 INFO Loading data...
2020-06-26 09:21:08,793 P7580 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-26 09:21:15,324 P7580 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-26 09:21:16,782 P7580 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-26 09:21:16,864 P7580 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-26 09:21:16,864 P7580 INFO Loading train data done.
2020-06-26 09:21:19,642 P7580 INFO Start training: 2371 batches/epoch
2020-06-26 09:21:19,642 P7580 INFO ************ Epoch=1 start ************
2020-06-26 09:24:00,196 P7580 INFO [Metrics] logloss: 0.193561 - AUC: 0.642748
2020-06-26 09:24:00,199 P7580 INFO Save best model: monitor(max): 0.449186
2020-06-26 09:24:00,274 P7580 INFO --- 2371/2371 batches finished ---
2020-06-26 09:24:00,321 P7580 INFO Train loss: 0.206774
2020-06-26 09:24:00,321 P7580 INFO ************ Epoch=1 end ************
2020-06-26 09:26:39,318 P7580 INFO [Metrics] logloss: 0.194774 - AUC: 0.641758
2020-06-26 09:26:39,320 P7580 INFO Monitor(max) STOP: 0.446984 !
2020-06-26 09:26:39,320 P7580 INFO Reduce learning rate on plateau: 0.000100
2020-06-26 09:26:39,320 P7580 INFO --- 2371/2371 batches finished ---
2020-06-26 09:26:39,400 P7580 INFO Train loss: 0.191805
2020-06-26 09:26:39,400 P7580 INFO ************ Epoch=2 end ************
2020-06-26 09:29:18,767 P7580 INFO [Metrics] logloss: 0.198885 - AUC: 0.639458
2020-06-26 09:29:18,769 P7580 INFO Monitor(max) STOP: 0.440574 !
2020-06-26 09:29:18,769 P7580 INFO Reduce learning rate on plateau: 0.000010
2020-06-26 09:29:18,769 P7580 INFO Early stopping at epoch=3
2020-06-26 09:29:18,769 P7580 INFO --- 2371/2371 batches finished ---
2020-06-26 09:29:18,819 P7580 INFO Train loss: 0.182966
2020-06-26 09:29:18,819 P7580 INFO Training finished.
2020-06-26 09:29:18,819 P7580 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/FwFM_taobao/taobao_x0_87391c5c/FwFM_taobao_x0_009_ac0c8212_model.ckpt
2020-06-26 09:29:18,928 P7580 INFO ****** Train/validation evaluation ******
2020-06-26 09:29:36,845 P7580 INFO [Metrics] logloss: 0.193561 - AUC: 0.642748
2020-06-26 09:29:36,888 P7580 INFO ******** Test evaluation ********
2020-06-26 09:29:36,889 P7580 INFO Loading data...
2020-06-26 09:29:36,889 P7580 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-26 09:29:37,910 P7580 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-26 09:29:37,910 P7580 INFO Loading test data done.
2020-06-26 09:29:56,105 P7580 INFO [Metrics] logloss: 0.193561 - AUC: 0.642748

```
