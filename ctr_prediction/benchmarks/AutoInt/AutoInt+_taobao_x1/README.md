## AutoInt+_Taobao_x0_001

A notebook to benchmark AutoInt+ on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193510 - AUC: 0.643238
```


### Logs
```python
2020-07-09 10:05:16,095 P38317 INFO {
    "attention_dim": "128",
    "attention_layers": "4",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[200, 200, 200]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "AutoInt",
    "model_id": "AutoInt_taobao_x0_002_88772411",
    "model_root": "./Taobao/AutoInt_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
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
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Taobao/Taobao_x0/test.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-09 10:05:16,096 P38317 INFO Set up feature encoder...
2020-07-09 10:05:16,096 P38317 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-07-09 10:05:17,048 P38317 INFO Total number of parameters: 23280022.
2020-07-09 10:05:17,048 P38317 INFO Loading data...
2020-07-09 10:05:17,050 P38317 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-07-09 10:05:23,851 P38317 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-09 10:05:25,380 P38317 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-07-09 10:05:25,479 P38317 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-09 10:05:25,480 P38317 INFO Loading train data done.
2020-07-09 10:05:28,704 P38317 INFO Start training: 2371 batches/epoch
2020-07-09 10:05:28,704 P38317 INFO ************ Epoch=1 start ************
2020-07-09 10:16:16,357 P38317 INFO [Metrics] logloss: 0.193510 - AUC: 0.643238
2020-07-09 10:16:16,359 P38317 INFO Save best model: monitor(max): 0.449728
2020-07-09 10:16:16,442 P38317 INFO --- 2371/2371 batches finished ---
2020-07-09 10:16:16,488 P38317 INFO Train loss: 0.200145
2020-07-09 10:16:16,488 P38317 INFO ************ Epoch=1 end ************
2020-07-09 10:27:34,585 P38317 INFO [Metrics] logloss: 0.198601 - AUC: 0.633776
2020-07-09 10:27:34,587 P38317 INFO Monitor(max) STOP: 0.435175 !
2020-07-09 10:27:34,587 P38317 INFO Reduce learning rate on plateau: 0.000100
2020-07-09 10:27:34,587 P38317 INFO --- 2371/2371 batches finished ---
2020-07-09 10:27:34,637 P38317 INFO Train loss: 0.188227
2020-07-09 10:27:34,637 P38317 INFO ************ Epoch=2 end ************
2020-07-09 10:40:26,643 P38317 INFO [Metrics] logloss: 0.235231 - AUC: 0.610688
2020-07-09 10:40:26,645 P38317 INFO Monitor(max) STOP: 0.375457 !
2020-07-09 10:40:26,645 P38317 INFO Reduce learning rate on plateau: 0.000010
2020-07-09 10:40:26,645 P38317 INFO Early stopping at epoch=3
2020-07-09 10:40:26,645 P38317 INFO --- 2371/2371 batches finished ---
2020-07-09 10:40:26,693 P38317 INFO Train loss: 0.166044
2020-07-09 10:40:26,693 P38317 INFO Training finished.
2020-07-09 10:40:26,693 P38317 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/AutoInt_taobao/taobao_x0_87391c5c/AutoInt_taobao_x0_002_88772411_model.ckpt
2020-07-09 10:40:26,881 P38317 INFO ****** Train/validation evaluation ******
2020-07-09 10:40:49,831 P38317 INFO [Metrics] logloss: 0.193510 - AUC: 0.643238
2020-07-09 10:40:49,868 P38317 INFO ******** Test evaluation ********
2020-07-09 10:40:49,868 P38317 INFO Loading data...
2020-07-09 10:40:49,869 P38317 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-07-09 10:40:50,742 P38317 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-07-09 10:40:50,742 P38317 INFO Loading test data done.
2020-07-09 10:41:13,490 P38317 INFO [Metrics] logloss: 0.193510 - AUC: 0.643238

```
