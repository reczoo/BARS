## FiBiNET_Avazu_x4_002

A notebook to benchmark FiBiNET on Avazu_x4_002 dataset.

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
[Metrics] logloss: 0.367529 - AUC: 0.800275
```


### Logs
```python
2020-05-11 16:30:23,177 P19355 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "bilinear_type": "field_interaction",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[2000, 2000, 2000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiBiNET",
    "model_id": "FiBiNET_avazu_x4_012_30e632d8",
    "model_root": "./Avazu/FiBiNET_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
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
    "gpu": "1"
}
2020-05-11 16:30:23,178 P19355 INFO Set up feature encoder...
2020-05-11 16:30:23,178 P19355 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-05-11 16:30:23,178 P19355 INFO Loading data...
2020-05-11 16:30:23,180 P19355 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-05-11 16:30:25,509 P19355 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-05-11 16:30:26,905 P19355 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-11 16:30:27,024 P19355 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-11 16:30:27,024 P19355 INFO Loading train data done.
2020-05-11 16:30:40,909 P19355 INFO **** Start training: 3235 batches/epoch ****
2020-05-11 17:17:03,178 P19355 INFO [Metrics] logloss: 0.367648 - AUC: 0.800055
2020-05-11 17:17:03,239 P19355 INFO Save best model: monitor(max): 0.432407
2020-05-11 17:17:04,704 P19355 INFO --- 3235/3235 batches finished ---
2020-05-11 17:17:04,754 P19355 INFO Train loss: 0.378340
2020-05-11 17:17:04,754 P19355 INFO ************ Epoch=1 end ************
2020-05-11 18:03:27,269 P19355 INFO [Metrics] logloss: 0.399502 - AUC: 0.780383
2020-05-11 18:03:27,373 P19355 INFO Monitor(max) STOP: 0.380881 !
2020-05-11 18:03:27,373 P19355 INFO Reduce learning rate on plateau: 0.000100
2020-05-11 18:03:27,373 P19355 INFO --- 3235/3235 batches finished ---
2020-05-11 18:03:27,466 P19355 INFO Train loss: 0.276094
2020-05-11 18:03:27,467 P19355 INFO ************ Epoch=2 end ************
2020-05-11 18:49:43,003 P19355 INFO [Metrics] logloss: 0.534865 - AUC: 0.758510
2020-05-11 18:49:43,100 P19355 INFO Monitor(max) STOP: 0.223644 !
2020-05-11 18:49:43,101 P19355 INFO Reduce learning rate on plateau: 0.000010
2020-05-11 18:49:43,101 P19355 INFO Early stopping at epoch=3
2020-05-11 18:49:43,101 P19355 INFO --- 3235/3235 batches finished ---
2020-05-11 18:49:43,202 P19355 INFO Train loss: 0.225279
2020-05-11 18:49:43,202 P19355 INFO Training finished.
2020-05-11 18:49:43,203 P19355 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Avazu/FiBiNET_avazu/avazu_x4_001_d45ad60e/FiBiNET_avazu_x4_012_30e632d8_avazu_x4_001_d45ad60e_model.ckpt
2020-05-11 18:49:45,380 P19355 INFO ****** Train/validation evaluation ******
2020-05-11 19:02:46,509 P19355 INFO [Metrics] logloss: 0.312543 - AUC: 0.880290
2020-05-11 19:04:23,025 P19355 INFO [Metrics] logloss: 0.367648 - AUC: 0.800055
2020-05-11 19:04:23,245 P19355 INFO ******** Test evaluation ********
2020-05-11 19:04:23,245 P19355 INFO Loading data...
2020-05-11 19:04:23,245 P19355 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-05-11 19:04:23,664 P19355 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-11 19:04:23,664 P19355 INFO Loading test data done.
2020-05-11 19:05:58,738 P19355 INFO [Metrics] logloss: 0.367529 - AUC: 0.800275

```
