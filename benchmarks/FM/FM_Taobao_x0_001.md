## FM_Taobao_x0_001

A notebook to benchmark FM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.194236 - AUC: 0.637279
```


### Logs
```python
2020-06-26 00:28:34,983 P55563 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'share_embedding': 'adgroup_id', 'type': 'sequence'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "FM",
    "model_id": "FM_taobao_x0_003_9c2c29a6",
    "model_root": "./Taobao/FM_taobao/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "l2(1.e-7)",
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
2020-06-26 00:28:34,984 P55563 INFO Set up feature encoder...
2020-06-26 00:28:34,984 P55563 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-26 00:28:36,014 P55563 INFO Total number of parameters: 22990325.
2020-06-26 00:28:36,014 P55563 INFO Loading data...
2020-06-26 00:28:36,017 P55563 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-26 00:28:44,632 P55563 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-26 00:28:46,822 P55563 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-26 00:28:46,995 P55563 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-26 00:28:46,995 P55563 INFO Loading train data done.
2020-06-26 00:28:50,994 P55563 INFO Start training: 2371 batches/epoch
2020-06-26 00:28:50,994 P55563 INFO ************ Epoch=1 start ************
2020-06-26 00:42:15,813 P55563 INFO [Metrics] logloss: 0.194236 - AUC: 0.637279
2020-06-26 00:42:15,816 P55563 INFO Save best model: monitor(max): 0.443044
2020-06-26 00:42:15,908 P55563 INFO --- 2371/2371 batches finished ---
2020-06-26 00:42:15,960 P55563 INFO Train loss: 0.210723
2020-06-26 00:42:15,960 P55563 INFO ************ Epoch=1 end ************
2020-06-26 00:55:38,600 P55563 INFO [Metrics] logloss: 0.196576 - AUC: 0.634637
2020-06-26 00:55:38,602 P55563 INFO Monitor(max) STOP: 0.438061 !
2020-06-26 00:55:38,602 P55563 INFO Reduce learning rate on plateau: 0.000100
2020-06-26 00:55:38,602 P55563 INFO --- 2371/2371 batches finished ---
2020-06-26 00:55:38,654 P55563 INFO Train loss: 0.189564
2020-06-26 00:55:38,654 P55563 INFO ************ Epoch=2 end ************
2020-06-26 01:08:58,547 P55563 INFO [Metrics] logloss: 0.200328 - AUC: 0.632740
2020-06-26 01:08:58,549 P55563 INFO Monitor(max) STOP: 0.432411 !
2020-06-26 01:08:58,549 P55563 INFO Reduce learning rate on plateau: 0.000010
2020-06-26 01:08:58,549 P55563 INFO Early stopping at epoch=3
2020-06-26 01:08:58,549 P55563 INFO --- 2371/2371 batches finished ---
2020-06-26 01:08:58,602 P55563 INFO Train loss: 0.174875
2020-06-26 01:08:58,602 P55563 INFO Training finished.
2020-06-26 01:08:58,602 P55563 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/FM_taobao/taobao_x0_87391c5c/FM_taobao_x0_003_9c2c29a6_model.ckpt
2020-06-26 01:08:58,786 P55563 INFO ****** Train/validation evaluation ******
2020-06-26 01:09:22,297 P55563 INFO [Metrics] logloss: 0.194236 - AUC: 0.637279
2020-06-26 01:09:22,343 P55563 INFO ******** Test evaluation ********
2020-06-26 01:09:22,343 P55563 INFO Loading data...
2020-06-26 01:09:22,343 P55563 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-26 01:09:23,162 P55563 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-26 01:09:23,162 P55563 INFO Loading test data done.
2020-06-26 01:09:47,134 P55563 INFO [Metrics] logloss: 0.194236 - AUC: 0.637279

```
