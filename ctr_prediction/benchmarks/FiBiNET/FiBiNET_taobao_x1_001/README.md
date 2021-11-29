## FiBiNET_Taobao_x0_001

A notebook to benchmark FiBiNET on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193770 - AUC: 0.643298
```


### Logs
```python
2020-06-25 11:26:10,561 P44036 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "bilinear_type": "field_interaction",
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
    "hidden_units": "[1000, 1000, 1000]",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FiBiNET",
    "model_id": "FiBiNET_taobao_x0_008_35dc3909",
    "model_root": "./Taobao/FiBiNET_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
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
2020-06-25 11:26:10,561 P44036 INFO Set up feature encoder...
2020-06-25 11:26:10,561 P44036 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-25 11:26:11,564 P44036 INFO Total number of parameters: 28865205.
2020-06-25 11:26:11,564 P44036 INFO Loading data...
2020-06-25 11:26:11,567 P44036 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-25 11:26:17,767 P44036 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-25 11:26:19,275 P44036 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-25 11:26:19,358 P44036 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-25 11:26:19,358 P44036 INFO Loading train data done.
2020-06-25 11:26:22,111 P44036 INFO Start training: 2371 batches/epoch
2020-06-25 11:26:22,111 P44036 INFO ************ Epoch=1 start ************
2020-06-25 11:35:13,559 P44036 INFO [Metrics] logloss: 0.193770 - AUC: 0.643298
2020-06-25 11:35:13,561 P44036 INFO Save best model: monitor(max): 0.449528
2020-06-25 11:35:13,666 P44036 INFO --- 2371/2371 batches finished ---
2020-06-25 11:35:13,716 P44036 INFO Train loss: 0.203203
2020-06-25 11:35:13,716 P44036 INFO ************ Epoch=1 end ************
2020-06-25 11:43:56,440 P44036 INFO [Metrics] logloss: 0.196249 - AUC: 0.635397
2020-06-25 11:43:56,442 P44036 INFO Monitor(max) STOP: 0.439147 !
2020-06-25 11:43:56,442 P44036 INFO Reduce learning rate on plateau: 0.000100
2020-06-25 11:43:56,442 P44036 INFO --- 2371/2371 batches finished ---
2020-06-25 11:43:56,492 P44036 INFO Train loss: 0.196988
2020-06-25 11:43:56,492 P44036 INFO ************ Epoch=2 end ************
2020-06-25 11:52:37,285 P44036 INFO [Metrics] logloss: 0.227653 - AUC: 0.603323
2020-06-25 11:52:37,287 P44036 INFO Monitor(max) STOP: 0.375670 !
2020-06-25 11:52:37,287 P44036 INFO Reduce learning rate on plateau: 0.000010
2020-06-25 11:52:37,287 P44036 INFO Early stopping at epoch=3
2020-06-25 11:52:37,287 P44036 INFO --- 2371/2371 batches finished ---
2020-06-25 11:52:37,338 P44036 INFO Train loss: 0.154853
2020-06-25 11:52:37,338 P44036 INFO Training finished.
2020-06-25 11:52:37,338 P44036 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/FiBiNET_taobao/taobao_x0_87391c5c/FiBiNET_taobao_x0_008_35dc3909_model.ckpt
2020-06-25 11:52:37,518 P44036 INFO ****** Train/validation evaluation ******
2020-06-25 11:52:56,477 P44036 INFO [Metrics] logloss: 0.193770 - AUC: 0.643298
2020-06-25 11:52:56,511 P44036 INFO ******** Test evaluation ********
2020-06-25 11:52:56,511 P44036 INFO Loading data...
2020-06-25 11:52:56,511 P44036 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-25 11:52:57,318 P44036 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-25 11:52:57,318 P44036 INFO Loading test data done.
2020-06-25 11:53:16,198 P44036 INFO [Metrics] logloss: 0.193770 - AUC: 0.643298

```
