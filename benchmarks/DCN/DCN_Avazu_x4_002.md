## DCN_Avazu_x4_002

A notebook to benchmark DCN on Avazu_x4_002 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default <OOV> token. Note that we found that min_category_count=1 performs the best, which is surprising.

We fix embedding_dim=40 following the existing FGCNN work.
### Code




### Results
```python
[Metrics] logloss: 0.369933 - AUC: 0.796517
```


### Logs
```python
2020-01-31 15:13:40,177 P587 INFO {
    "batch_norm": "True",
    "batch_size": "10000",
    "crossing_layers": "3",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1000, 1000]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_avazu_x4_008_94ed77b8",
    "model_root": "./Avazu/DCN_avazu/",
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
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "7"
}
2020-01-31 15:13:40,245 P587 INFO Set up feature encoder...
2020-01-31 15:13:40,245 P587 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-01-31 15:13:40,245 P587 INFO Loading data...
2020-01-31 15:13:40,253 P587 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-01-31 15:13:43,479 P587 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-01-31 15:13:45,636 P587 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-01-31 15:13:45,752 P587 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-01-31 15:13:45,752 P587 INFO Loading train data done.
2020-01-31 15:14:03,552 P587 INFO **** Start training: 3235 batches/epoch ****
2020-01-31 15:21:07,508 P587 INFO [Metrics] logloss: 0.370086 - AUC: 0.796210
2020-01-31 15:21:07,600 P587 INFO Save best model: monitor(max): 0.426124
2020-01-31 15:21:10,478 P587 INFO --- 3235/3235 batches finished ---
2020-01-31 15:21:10,537 P587 INFO Train loss: 0.379529
2020-01-31 15:21:10,537 P587 INFO ************ Epoch=1 end ************
2020-01-31 15:28:11,149 P587 INFO [Metrics] logloss: 0.392640 - AUC: 0.778845
2020-01-31 15:28:11,240 P587 INFO Monitor(max) STOP: 0.386205 !
2020-01-31 15:28:11,240 P587 INFO Reduce learning rate on plateau: 0.000100
2020-01-31 15:28:11,240 P587 INFO --- 3235/3235 batches finished ---
2020-01-31 15:28:11,300 P587 INFO Train loss: 0.289539
2020-01-31 15:28:11,300 P587 INFO ************ Epoch=2 end ************
2020-01-31 15:35:12,308 P587 INFO [Metrics] logloss: 0.484118 - AUC: 0.765913
2020-01-31 15:35:12,401 P587 INFO Monitor(max) STOP: 0.281795 !
2020-01-31 15:35:12,401 P587 INFO Reduce learning rate on plateau: 0.000010
2020-01-31 15:35:12,401 P587 INFO Early stopping at epoch=3
2020-01-31 15:35:12,401 P587 INFO --- 3235/3235 batches finished ---
2020-01-31 15:35:12,471 P587 INFO Train loss: 0.236384
2020-01-31 15:35:12,471 P587 INFO Training finished.
2020-01-31 15:35:12,471 P587 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/DCN_avazu/avazu_x4_001_d45ad60e/DCN_avazu_x4_008_94ed77b8_avazu_x4_001_d45ad60e_model.ckpt
2020-01-31 15:35:14,414 P587 INFO ****** Train/validation evaluation ******
2020-01-31 15:39:41,799 P587 INFO [Metrics] logloss: 0.318755 - AUC: 0.873160
2020-01-31 15:40:12,717 P587 INFO [Metrics] logloss: 0.370086 - AUC: 0.796210
2020-01-31 15:40:12,903 P587 INFO ******** Test evaluation ********
2020-01-31 15:40:12,903 P587 INFO Loading data...
2020-01-31 15:40:12,904 P587 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-01-31 15:40:13,387 P587 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-01-31 15:40:13,387 P587 INFO Loading test data done.
2020-01-31 15:40:41,336 P587 INFO [Metrics] logloss: 0.369933 - AUC: 0.796517


```
