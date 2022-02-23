## YoutubeDNN_Taobao_x0_001 

A notebook to benchmark YoutubeDNN on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193505 - AUC: 0.643565
```


### Logs
```python
2020-06-23 10:06:00,062 P33452 INFO {
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
    "model": "DNN",
    "model_id": "DNN_taobao_x0_003_e719a45c",
    "model_root": "./Taobao/DNN_taobao/",
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
2020-06-23 10:06:00,062 P33452 INFO Set up feature encoder...
2020-06-23 10:06:00,063 P33452 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-23 10:06:00,897 P33452 INFO Total number of parameters: 18135809.
2020-06-23 10:06:00,898 P33452 INFO Loading data...
2020-06-23 10:06:00,900 P33452 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-23 10:06:07,440 P33452 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-23 10:06:09,199 P33452 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-23 10:06:09,319 P33452 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-23 10:06:09,320 P33452 INFO Loading train data done.
2020-06-23 10:06:13,716 P33452 INFO Start training: 2371 batches/epoch
2020-06-23 10:06:13,716 P33452 INFO ************ Epoch=1 start ************
2020-06-23 10:17:06,080 P33452 INFO [Metrics] logloss: 0.193505 - AUC: 0.643565
2020-06-23 10:17:06,083 P33452 INFO Save best model: monitor(max): 0.450060
2020-06-23 10:17:06,149 P33452 INFO --- 2371/2371 batches finished ---
2020-06-23 10:17:06,196 P33452 INFO Train loss: 0.202840
2020-06-23 10:17:06,196 P33452 INFO ************ Epoch=1 end ************
2020-06-23 10:26:49,752 P33452 INFO [Metrics] logloss: 0.196170 - AUC: 0.634937
2020-06-23 10:26:49,774 P33452 INFO Monitor(max) STOP: 0.438766 !
2020-06-23 10:26:49,774 P33452 INFO Reduce learning rate on plateau: 0.000100
2020-06-23 10:26:49,774 P33452 INFO --- 2371/2371 batches finished ---
2020-06-23 10:26:49,866 P33452 INFO Train loss: 0.195168
2020-06-23 10:26:49,866 P33452 INFO ************ Epoch=2 end ************
2020-06-23 10:36:52,629 P33452 INFO [Metrics] logloss: 0.215681 - AUC: 0.615430
2020-06-23 10:36:52,633 P33452 INFO Monitor(max) STOP: 0.399749 !
2020-06-23 10:36:52,633 P33452 INFO Reduce learning rate on plateau: 0.000010
2020-06-23 10:36:52,633 P33452 INFO Early stopping at epoch=3
2020-06-23 10:36:52,633 P33452 INFO --- 2371/2371 batches finished ---
2020-06-23 10:36:52,677 P33452 INFO Train loss: 0.176415
2020-06-23 10:36:52,678 P33452 INFO Training finished.
2020-06-23 10:36:52,678 P33452 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/DNN_taobao/taobao_x0_87391c5c/DNN_taobao_x0_003_e719a45c_model.ckpt
2020-06-23 10:36:52,786 P33452 INFO ****** Train/validation evaluation ******
2020-06-23 10:37:15,153 P33452 INFO [Metrics] logloss: 0.193505 - AUC: 0.643565
2020-06-23 10:37:15,204 P33452 INFO ******** Test evaluation ********
2020-06-23 10:37:15,204 P33452 INFO Loading data...
2020-06-23 10:37:15,204 P33452 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-23 10:37:16,156 P33452 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-23 10:37:16,157 P33452 INFO Loading test data done.
2020-06-23 10:37:40,254 P33452 INFO [Metrics] logloss: 0.193505 - AUC: 0.643565

```
