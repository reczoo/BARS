## InterHAt_Avazu_x4_002

A notebook to benchmark InterHAt on Avazu_x4_002 dataset.

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
[Metrics] logloss: 0.372247 - AUC: 0.792731
```


### Logs
```python
2020-06-05 21:27:51,037 P32957 INFO {
    "attention_dim": "40",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_dim": "500",
    "hidden_units": "[]",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "InterHAt",
    "model_id": "InterHAt_avazu_x4_001_d45ad60e_009_bfbac078",
    "model_root": "./Avazu/InterHAt_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "order": "4",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-05 21:27:51,038 P32957 INFO Set up feature encoder...
2020-06-05 21:27:51,038 P32957 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-06-05 21:27:51,038 P32957 INFO Loading data...
2020-06-05 21:27:51,040 P32957 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-06-05 21:27:53,724 P32957 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-06-05 21:27:55,056 P32957 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-05 21:27:55,170 P32957 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-05 21:27:55,170 P32957 INFO Loading train data done.
2020-06-05 21:28:05,958 P32957 INFO **** Start training: 3235 batches/epoch ****
2020-06-05 21:46:11,924 P32957 INFO [Metrics] logloss: 0.372400 - AUC: 0.792452
2020-06-05 21:46:11,927 P32957 INFO Save best model: monitor(max): 0.420051
2020-06-05 21:46:13,354 P32957 INFO --- 3235/3235 batches finished ---
2020-06-05 21:46:13,389 P32957 INFO Train loss: 0.382025
2020-06-05 21:46:13,389 P32957 INFO ************ Epoch=1 end ************
2020-06-05 22:04:19,873 P32957 INFO [Metrics] logloss: 0.421522 - AUC: 0.764044
2020-06-05 22:04:19,876 P32957 INFO Monitor(max) STOP: 0.342522 !
2020-06-05 22:04:19,876 P32957 INFO Reduce learning rate on plateau: 0.000100
2020-06-05 22:04:19,876 P32957 INFO --- 3235/3235 batches finished ---
2020-06-05 22:04:19,912 P32957 INFO Train loss: 0.288440
2020-06-05 22:04:19,912 P32957 INFO ************ Epoch=2 end ************
2020-06-05 22:22:25,507 P32957 INFO [Metrics] logloss: 0.577693 - AUC: 0.727019
2020-06-05 22:22:25,510 P32957 INFO Monitor(max) STOP: 0.149325 !
2020-06-05 22:22:25,510 P32957 INFO Reduce learning rate on plateau: 0.000010
2020-06-05 22:22:25,510 P32957 INFO Early stopping at epoch=3
2020-06-05 22:22:25,511 P32957 INFO --- 3235/3235 batches finished ---
2020-06-05 22:22:25,543 P32957 INFO Train loss: 0.252509
2020-06-05 22:22:25,544 P32957 INFO Training finished.
2020-06-05 22:22:25,544 P32957 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Avazu/InterHAt_avazu/avazu_x4_001_d45ad60e/InterHAt_avazu_x4_001_d45ad60e_009_bfbac078_model.ckpt
2020-06-05 22:22:28,999 P32957 INFO ****** Train/validation evaluation ******
2020-06-05 22:23:04,710 P32957 INFO [Metrics] logloss: 0.372400 - AUC: 0.792452
2020-06-05 22:23:04,842 P32957 INFO ******** Test evaluation ********
2020-06-05 22:23:04,842 P32957 INFO Loading data...
2020-06-05 22:23:04,842 P32957 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-06-05 22:23:05,312 P32957 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-05 22:23:05,312 P32957 INFO Loading test data done.
2020-06-05 22:23:41,402 P32957 INFO [Metrics] logloss: 0.372247 - AUC: 0.792731


```
