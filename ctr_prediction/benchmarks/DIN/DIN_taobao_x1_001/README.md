## DIN_Taobao_x0_001

A notebook to benchmark DIN on Taobao_x0_001 dataset.

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
[Metrics] logloss: 0.193716 - AUC: 0.644035
```

### Logs
```python
2020-06-22 16:52:40,640 P7636 INFO {
    "attention_activations": "Dice",
    "attention_final_activation": "None",
    "attention_units": "[]",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Taobao/",
    "dataset_id": "taobao_x0_b98dbce2",
    "debug": "False",
    "din_history_field": "click_history",
    "din_query_field": "adgroup_id",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[500, 500, 500]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'float', 'name': 'price', 'type': 'numeric'}, {'active': True, 'dtype': 'str', 'name': ['userid', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'], 'type': 'categorical'}, {'active': False, 'dtype': 'str', 'name': 'new_user_class_level', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': None, 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'type': 'sequence'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'clk'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DIN",
    "model_id": "DIN_taobao_x0_001_4ba9f963",
    "model_root": "./Taobao/DIN_taobao/",
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
2020-06-22 16:52:40,640 P7636 INFO Set up feature encoder...
2020-06-22 16:52:40,640 P7636 INFO Reading file: ../data/Taobao/Taobao_x0/train.csv
2020-06-22 16:54:07,662 P7636 INFO Preprocess feature columns...
2020-06-22 16:54:59,681 P7636 INFO Fit feature encoder...
2020-06-22 16:54:59,681 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'userid', 'type': 'categorical'}
2020-06-22 16:55:07,207 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'adgroup_id', 'type': 'categorical'}
2020-06-22 16:55:19,182 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'pid', 'type': 'categorical'}
2020-06-22 16:55:23,145 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'cate_id', 'type': 'categorical'}
2020-06-22 16:55:27,378 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'campaign_id', 'type': 'categorical'}
2020-06-22 16:55:36,975 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'customer', 'type': 'categorical'}
2020-06-22 16:55:45,460 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'brand', 'type': 'categorical'}
2020-06-22 16:55:50,976 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'cms_segid', 'type': 'categorical'}
2020-06-22 16:55:54,599 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'cms_group_id', 'type': 'categorical'}
2020-06-22 16:55:57,860 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'final_gender_code', 'type': 'categorical'}
2020-06-22 16:56:01,068 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'age_level', 'type': 'categorical'}
2020-06-22 16:56:04,223 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'pvalue_level', 'type': 'categorical'}
2020-06-22 16:56:06,727 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'shopping_level', 'type': 'categorical'}
2020-06-22 16:56:09,935 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'occupation', 'type': 'categorical'}
2020-06-22 16:56:13,226 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}
2020-06-22 16:56:18,308 P7636 INFO Processing column: {'active': True, 'dtype': 'str', 'encoder': None, 'max_len': 64, 'name': 'click_history', 'padding': 'pre', 'type': 'sequence'}
2020-06-22 16:57:28,017 P7636 INFO Set feature index...
2020-06-22 16:57:28,018 P7636 INFO Pickle feature_encode: ../data/Taobao/taobao_x0_b98dbce2/feature_encoder.pkl
2020-06-22 16:57:28,686 P7636 INFO Save feature_map to json: ../data/Taobao/taobao_x0_b98dbce2/feature_map.json
2020-06-22 16:57:28,687 P7636 INFO Set feature encoder done.
2020-06-22 16:57:36,247 P7636 INFO Total number of parameters: 21209778.
2020-06-22 16:57:36,248 P7636 INFO Loading data...
2020-06-22 16:57:36,251 P7636 INFO Reading file: ../data/Taobao/Taobao_x0/train.csv
2020-06-22 16:59:03,606 P7636 INFO Preprocess feature columns...
2020-06-22 17:01:11,152 P7636 INFO Transform feature columns...
2020-06-22 17:08:11,466 P7636 INFO Saving data to h5: ../data/Taobao/taobao_x0_b98dbce2/train.h5
2020-06-22 17:08:31,855 P7636 INFO Reading file: ../data/Taobao/Taobao_x0/test.csv
2020-06-22 17:08:42,853 P7636 INFO Preprocess feature columns...
2020-06-22 17:08:48,247 P7636 INFO Transform feature columns...
2020-06-22 17:09:29,246 P7636 INFO Saving data to h5: ../data/Taobao/taobao_x0_b98dbce2/test.h5
2020-06-22 17:09:34,461 P7636 INFO Train samples: total/23709456, pos/1222521, neg/22486935, ratio/5.16%
2020-06-22 17:09:34,544 P7636 INFO Validation samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-22 17:09:34,545 P7636 INFO Loading train data done.
2020-06-22 17:09:38,720 P7636 INFO Start training: 2371 batches/epoch
2020-06-22 17:09:38,720 P7636 INFO ************ Epoch=1 start ************
2020-06-22 17:20:13,820 P7636 INFO [Metrics] logloss: 0.193716 - AUC: 0.644035
2020-06-22 17:20:13,822 P7636 INFO Save best model: monitor(max): 0.450320
2020-06-22 17:20:13,889 P7636 INFO --- 2371/2371 batches finished ---
2020-06-22 17:20:14,316 P7636 INFO Train loss: 0.203096
2020-06-22 17:20:14,317 P7636 INFO ************ Epoch=1 end ************
2020-06-22 17:30:04,046 P7636 INFO [Metrics] logloss: 0.196884 - AUC: 0.635190
2020-06-22 17:30:04,048 P7636 INFO Monitor(max) STOP: 0.438306 !
2020-06-22 17:30:04,048 P7636 INFO Reduce learning rate on plateau: 0.000100
2020-06-22 17:30:04,048 P7636 INFO --- 2371/2371 batches finished ---
2020-06-22 17:30:04,464 P7636 INFO Train loss: 0.195726
2020-06-22 17:30:04,465 P7636 INFO ************ Epoch=2 end ************
2020-06-22 17:38:17,574 P7636 INFO [Metrics] logloss: 0.212287 - AUC: 0.618973
2020-06-22 17:38:17,575 P7636 INFO Monitor(max) STOP: 0.406686 !
2020-06-22 17:38:17,575 P7636 INFO Reduce learning rate on plateau: 0.000010
2020-06-22 17:38:17,575 P7636 INFO Early stopping at epoch=3
2020-06-22 17:38:17,575 P7636 INFO --- 2371/2371 batches finished ---
2020-06-22 17:38:18,020 P7636 INFO Train loss: 0.178195
2020-06-22 17:38:18,020 P7636 INFO Training finished.
2020-06-22 17:38:18,020 P7636 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Taobao/DIN_taobao/taobao_x0_b98dbce2/DIN_taobao_x0_001_4ba9f963_model.ckpt
2020-06-22 17:38:18,170 P7636 INFO ****** Train/validation evaluation ******
2020-06-22 17:38:37,845 P7636 INFO [Metrics] logloss: 0.193716 - AUC: 0.644035
2020-06-22 17:38:37,890 P7636 INFO ******** Test evaluation ********
2020-06-22 17:38:37,890 P7636 INFO Loading data...
2020-06-22 17:38:37,890 P7636 INFO Loading data from h5: ../data/Taobao/taobao_x0_b98dbce2/test.h5
2020-06-22 17:38:38,763 P7636 INFO Test samples: total/2848505, pos/143535, neg/2704970, ratio/5.04%
2020-06-22 17:38:38,763 P7636 INFO Loading test data done.
2020-06-22 17:38:58,676 P7636 INFO [Metrics] logloss: 0.193716 - AUC: 0.644035


```