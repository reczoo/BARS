## IPNN_Taobao_x0_001 

A notebook to benchmark IPNN on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193563 - AUC: 0.643675
```


### Logs
```python
2020-06-27 00:10:07,879 P23190 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
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
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[500, 500]",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "PNN",
    "model_id": "PNN_taobao_x0_003_71bc2720",
    "model_root": "./Taobao/PNN_taobao/",
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
2020-06-27 00:10:07,880 P23190 INFO Set up feature encoder...
2020-06-27 00:10:07,880 P23190 INFO Load feature_encoder from pickle: ../data/Taobao/taobao_x0_87391c5c/feature_encoder.pkl
2020-06-27 00:10:08,862 P23190 INFO Total number of parameters: 22077453.
2020-06-27 00:10:08,862 P23190 INFO Loading data...
2020-06-27 00:10:08,865 P23190 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/train.h5
2020-06-27 00:10:17,277 P23190 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-27 00:10:19,392 P23190 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-27 00:10:19,527 P23190 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-27 00:10:19,527 P23190 INFO Loading train data done.
2020-06-27 00:10:23,192 P23190 INFO Start training: 2371 batches/epoch
2020-06-27 00:10:23,193 P23190 INFO ************ Epoch=1 start ************
2020-06-27 00:16:36,602 P23190 INFO [Metrics] logloss: 0.193563 - AUC: 0.643675
2020-06-27 00:16:36,603 P23190 INFO Save best model: monitor(max): 0.450112
2020-06-27 00:16:36,691 P23190 INFO --- 2371/2371 batches finished ---
2020-06-27 00:16:36,741 P23190 INFO Train loss: 0.200101
2020-06-27 00:16:36,742 P23190 INFO ************ Epoch=1 end ************
2020-06-27 00:22:46,190 P23190 INFO [Metrics] logloss: 0.197828 - AUC: 0.635072
2020-06-27 00:22:46,192 P23190 INFO Monitor(max) STOP: 0.437244 !
2020-06-27 00:22:46,192 P23190 INFO Reduce learning rate on plateau: 0.000100
2020-06-27 00:22:46,192 P23190 INFO --- 2371/2371 batches finished ---
2020-06-27 00:22:46,244 P23190 INFO Train loss: 0.188460
2020-06-27 00:22:46,244 P23190 INFO ************ Epoch=2 end ************
2020-06-27 00:28:53,299 P23190 INFO [Metrics] logloss: 0.236756 - AUC: 0.604754
2020-06-27 00:28:53,301 P23190 INFO Monitor(max) STOP: 0.367998 !
2020-06-27 00:28:53,301 P23190 INFO Reduce learning rate on plateau: 0.000010
2020-06-27 00:28:53,301 P23190 INFO Early stopping at epoch=3
2020-06-27 00:28:53,301 P23190 INFO --- 2371/2371 batches finished ---
2020-06-27 00:28:53,355 P23190 INFO Train loss: 0.154605
2020-06-27 00:28:53,355 P23190 INFO Training finished.
2020-06-27 00:28:53,355 P23190 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/PNN_taobao/taobao_x0_87391c5c/PNN_taobao_x0_003_71bc2720_model.ckpt
2020-06-27 00:28:53,487 P23190 INFO ****** Train/validation evaluation ******
2020-06-27 00:29:13,008 P23190 INFO [Metrics] logloss: 0.193563 - AUC: 0.643675
2020-06-27 00:29:13,051 P23190 INFO ******** Test evaluation ********
2020-06-27 00:29:13,051 P23190 INFO Loading data...
2020-06-27 00:29:13,051 P23190 INFO Loading data from h5: ../data/Taobao/taobao_x0_87391c5c/test.h5
2020-06-27 00:29:13,890 P23190 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-27 00:29:13,890 P23190 INFO Loading test data done.
2020-06-27 00:29:35,840 P23190 INFO [Metrics] logloss: 0.193563 - AUC: 0.643675

```
