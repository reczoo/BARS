## DeepCross_Taobao_x0_001

A notebook to benchmark DeepCross on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193437 - AUC: 0.644225
```


### Logs
```python
2020-07-03 10:45:07,563 P44192 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DeepCrossing",
    "model_id": "DeepCrossing_taobao_x0_002_e9789bce",
    "model_root": "./Taobao/DeepCrossing_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "residual_blocks": "[100, 100, 100]",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Taobao/Taobao_x0/test.csv",
    "train_data": "../data/Taobao/Taobao_x0/train.csv",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Taobao/Taobao_x0/test.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-03 10:45:07,564 P44192 INFO Set up feature encoder...
2020-07-03 10:45:07,564 P44192 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-07-03 10:45:08,541 P44192 INFO Total number of parameters: 21792877.
2020-07-03 10:45:08,541 P44192 INFO Loading data...
2020-07-03 10:45:08,543 P44192 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-07-03 10:45:17,976 P44192 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-03 10:45:20,286 P44192 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-07-03 10:45:20,463 P44192 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-03 10:45:20,463 P44192 INFO Loading train data done.
2020-07-03 10:45:23,702 P44192 INFO Start training: 2371 batches/epoch
2020-07-03 10:45:23,703 P44192 INFO ************ Epoch=1 start ************
2020-07-03 10:49:30,243 P44192 INFO [Metrics] logloss: 0.193437 - AUC: 0.644225
2020-07-03 10:49:30,247 P44192 INFO Save best model: monitor(max): 0.450788
2020-07-03 10:49:30,339 P44192 INFO --- 2371/2371 batches finished ---
2020-07-03 10:49:30,404 P44192 INFO Train loss: 0.199712
2020-07-03 10:49:30,404 P44192 INFO ************ Epoch=1 end ************
2020-07-03 10:53:28,078 P44192 INFO [Metrics] logloss: 0.198098 - AUC: 0.633074
2020-07-03 10:53:28,080 P44192 INFO Monitor(max) STOP: 0.434976 !
2020-07-03 10:53:28,080 P44192 INFO Reduce learning rate on plateau: 0.000100
2020-07-03 10:53:28,080 P44192 INFO --- 2371/2371 batches finished ---
2020-07-03 10:53:28,137 P44192 INFO Train loss: 0.188909
2020-07-03 10:53:28,137 P44192 INFO ************ Epoch=2 end ************
2020-07-03 10:57:34,200 P44192 INFO [Metrics] logloss: 0.227597 - AUC: 0.611973
2020-07-03 10:57:34,202 P44192 INFO Monitor(max) STOP: 0.384376 !
2020-07-03 10:57:34,202 P44192 INFO Reduce learning rate on plateau: 0.000010
2020-07-03 10:57:34,202 P44192 INFO Early stopping at epoch=3
2020-07-03 10:57:34,202 P44192 INFO --- 2371/2371 batches finished ---
2020-07-03 10:57:34,253 P44192 INFO Train loss: 0.167355
2020-07-03 10:57:34,253 P44192 INFO Training finished.
2020-07-03 10:57:34,254 P44192 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/DeepCrossing_taobao/taobao_x0_87391c5c/DeepCrossing_taobao_x0_002_e9789bce_model.ckpt
2020-07-03 10:57:34,370 P44192 INFO ****** Train/validation evaluation ******
2020-07-03 10:57:53,674 P44192 INFO [Metrics] logloss: 0.193437 - AUC: 0.644225
2020-07-03 10:57:53,725 P44192 INFO ******** Test evaluation ********
2020-07-03 10:57:53,726 P44192 INFO Loading data...
2020-07-03 10:57:53,726 P44192 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-03 10:57:54,726 P44192 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-03 10:57:54,727 P44192 INFO Loading test data done.
2020-07-03 10:58:17,197 P44192 INFO [Metrics] logloss: 0.193437 - AUC: 0.644225

```