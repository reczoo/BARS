## FFM_Taobao_x0_001

A notebook to benchmark FFM on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193566 - AUC: 0.643304
```


### Logs
```python
2020-06-27 01:43:12,970 P24914 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_87391c5c",
    "debug": "False",
    "embedding_dim": "4",
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
    "model": "FFM",
    "model_id": "FFM_taobao_x0_003_5900c10b",
    "model_root": "./Taobao/FFM_taobao/",
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
2020-06-27 01:43:12,970 P24914 INFO Set up feature encoder...
2020-06-27 01:43:12,970 P24914 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-27 01:43:15,547 P24914 INFO Total number of parameters: 82494693.
2020-06-27 01:43:15,548 P24914 INFO Loading data...
2020-06-27 01:43:15,551 P24914 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-27 01:43:22,428 P24914 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-27 01:43:24,140 P24914 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-27 01:43:24,247 P24914 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-27 01:43:24,247 P24914 INFO Loading train data done.
2020-06-27 01:43:27,454 P24914 INFO Start training: 2371 batches/epoch
2020-06-27 01:43:27,455 P24914 INFO ************ Epoch=1 start ************
2020-06-27 02:12:27,341 P24914 INFO [Metrics] logloss: 0.193566 - AUC: 0.643304
2020-06-27 02:12:27,345 P24914 INFO Save best model: monitor(max): 0.449738
2020-06-27 02:12:27,681 P24914 INFO --- 2371/2371 batches finished ---
2020-06-27 02:12:27,747 P24914 INFO Train loss: 0.208228
2020-06-27 02:12:27,747 P24914 INFO ************ Epoch=1 end ************
2020-06-27 02:41:19,572 P24914 INFO [Metrics] logloss: 0.194369 - AUC: 0.642112
2020-06-27 02:41:19,575 P24914 INFO Monitor(max) STOP: 0.447743 !
2020-06-27 02:41:19,575 P24914 INFO Reduce learning rate on plateau: 0.000100
2020-06-27 02:41:19,575 P24914 INFO --- 2371/2371 batches finished ---
2020-06-27 02:41:19,640 P24914 INFO Train loss: 0.193677
2020-06-27 02:41:19,641 P24914 INFO ************ Epoch=2 end ************
2020-06-27 03:10:10,817 P24914 INFO [Metrics] logloss: 0.197764 - AUC: 0.639217
2020-06-27 03:10:10,820 P24914 INFO Monitor(max) STOP: 0.441453 !
2020-06-27 03:10:10,820 P24914 INFO Reduce learning rate on plateau: 0.000010
2020-06-27 03:10:10,820 P24914 INFO Early stopping at epoch=3
2020-06-27 03:10:10,821 P24914 INFO --- 2371/2371 batches finished ---
2020-06-27 03:10:10,888 P24914 INFO Train loss: 0.181446
2020-06-27 03:10:10,888 P24914 INFO Training finished.
2020-06-27 03:10:10,888 P24914 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/FFM_taobao/taobao_x0_87391c5c/FFM_taobao_x0_003_5900c10b_model.ckpt
2020-06-27 03:10:11,390 P24914 INFO ****** Train/validation evaluation ******
2020-06-27 03:10:37,577 P24914 INFO [Metrics] logloss: 0.193566 - AUC: 0.643304
2020-06-27 03:10:37,646 P24914 INFO ******** Test evaluation ********
2020-06-27 03:10:37,647 P24914 INFO Loading data...
2020-06-27 03:10:37,647 P24914 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-27 03:10:38,504 P24914 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-27 03:10:38,504 P24914 INFO Loading test data done.
2020-06-27 03:11:03,738 P24914 INFO [Metrics] logloss: 0.193566 - AUC: 0.643304


```
