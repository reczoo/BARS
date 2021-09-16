## xDeepFM_Avazu_x4_002

A notebook to benchmark xDeepFM on Avazu_x4_002 dataset.

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
[Metrics] logloss: 0.369703 - AUC: 0.796744
```


### Logs
```python
2020-03-23 02:17:23,276 P1433 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[50, 50]",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_hidden_units": "[500, 500]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x4_008_3aefa952",
    "model_root": "./Avazu/xDeepFM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "1",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-03-23 02:17:23,278 P1433 INFO Set up feature encoder...
2020-03-23 02:17:23,278 P1433 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-03-23 02:17:23,279 P1433 INFO Loading data...
2020-03-23 02:17:23,288 P1433 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-03-23 02:17:26,818 P1433 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-03-23 02:17:28,141 P1433 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-03-23 02:17:28,243 P1433 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-03-23 02:17:28,243 P1433 INFO Loading train data done.
2020-03-23 02:17:40,642 P1433 INFO **** Start training: 3235 batches/epoch ****
2020-03-23 02:36:53,471 P1433 INFO [Metrics] logloss: 0.369772 - AUC: 0.796629
2020-03-23 02:36:53,473 P1433 INFO Save best model: monitor(max): 0.426857
2020-03-23 02:36:55,560 P1433 INFO --- 3235/3235 batches finished ---
2020-03-23 02:36:55,621 P1433 INFO Train loss: 0.380686
2020-03-23 02:36:55,621 P1433 INFO ************ Epoch=1 end ************
2020-03-23 02:56:05,356 P1433 INFO [Metrics] logloss: 0.392353 - AUC: 0.780802
2020-03-23 02:56:05,360 P1433 INFO Monitor(max) STOP: 0.388449 !
2020-03-23 02:56:05,360 P1433 INFO Reduce learning rate on plateau: 0.000100
2020-03-23 02:56:05,360 P1433 INFO Early stopping at epoch=2
2020-03-23 02:56:05,360 P1433 INFO --- 3235/3235 batches finished ---
2020-03-23 02:56:05,415 P1433 INFO Train loss: 0.287833
2020-03-23 02:56:05,416 P1433 INFO Training finished.
2020-03-23 02:56:05,416 P1433 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu/avazu_x4_001_d45ad60e/xDeepFM_avazu_x4_008_3aefa952_model.ckpt
2020-03-23 02:56:07,298 P1433 INFO ****** Train/validation evaluation ******
2020-03-23 02:59:36,028 P1433 INFO [Metrics] logloss: 0.317834 - AUC: 0.871821
2020-03-23 03:00:00,499 P1433 INFO [Metrics] logloss: 0.369772 - AUC: 0.796629
2020-03-23 03:00:00,584 P1433 INFO ******** Test evaluation ********
2020-03-23 03:00:00,584 P1433 INFO Loading data...
2020-03-23 03:00:00,584 P1433 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-03-23 03:00:01,158 P1433 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-03-23 03:00:01,158 P1433 INFO Loading test data done.
2020-03-23 03:00:25,817 P1433 INFO [Metrics] logloss: 0.369703 - AUC: 0.796744


```
