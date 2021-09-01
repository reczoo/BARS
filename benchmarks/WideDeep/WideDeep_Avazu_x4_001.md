## WideDeep_Avazu_x4_001 

A notebook to benchmark WideDeep on Avazu_x4_001 dataset.

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
In this setting, we preprocess the data split by removing the ``id`` field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default ``<OOV>`` token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless ``id`` field nor specially preprocess the timestamp field.

To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.


### Code


### Results
```python
[Metrics] logloss: 0.371969 - AUC: 0.792869
```


### Logs
```python
2020-06-12 02:18:01,179 P13114 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-09",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[2000, 2000, 2000, 2000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "WideDeep",
    "model_id": "WideDeep_avazu_x4_3bbbc4c9_016_2934bd82",
    "model_root": "./Avazu/WideDeep_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-12 02:18:01,180 P13114 INFO Set up feature encoder...
2020-06-12 02:18:01,180 P13114 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-12 02:18:10,470 P13114 INFO Total number of parameters: 76544576.
2020-06-12 02:18:10,470 P13114 INFO Loading data...
2020-06-12 02:18:10,472 P13114 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-12 02:18:13,631 P13114 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-12 02:18:15,229 P13114 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-12 02:18:15,402 P13114 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-12 02:18:15,402 P13114 INFO Loading train data done.
2020-06-12 02:18:54,785 P13114 INFO Start training: 3235 batches/epoch
2020-06-12 02:18:54,786 P13114 INFO ************ Epoch=1 start ************
2020-06-12 02:33:55,594 P13114 INFO [Metrics] logloss: 0.372114 - AUC: 0.792644
2020-06-12 02:33:55,597 P13114 INFO Save best model: monitor(max): 0.420530
2020-06-12 02:33:55,888 P13114 INFO --- 3235/3235 batches finished ---
2020-06-12 02:33:55,926 P13114 INFO Train loss: 0.380700
2020-06-12 02:33:55,926 P13114 INFO ************ Epoch=1 end ************
2020-06-12 02:48:57,068 P13114 INFO [Metrics] logloss: 0.380340 - AUC: 0.789085
2020-06-12 02:48:57,071 P13114 INFO Monitor(max) STOP: 0.408744 !
2020-06-12 02:48:57,071 P13114 INFO Reduce learning rate on plateau: 0.000100
2020-06-12 02:48:57,072 P13114 INFO --- 3235/3235 batches finished ---
2020-06-12 02:48:57,106 P13114 INFO Train loss: 0.334034
2020-06-12 02:48:57,107 P13114 INFO ************ Epoch=2 end ************
2020-06-12 03:03:57,063 P13114 INFO [Metrics] logloss: 0.423783 - AUC: 0.776741
2020-06-12 03:03:57,066 P13114 INFO Monitor(max) STOP: 0.352959 !
2020-06-12 03:03:57,066 P13114 INFO Reduce learning rate on plateau: 0.000010
2020-06-12 03:03:57,066 P13114 INFO Early stopping at epoch=3
2020-06-12 03:03:57,066 P13114 INFO --- 3235/3235 batches finished ---
2020-06-12 03:03:57,106 P13114 INFO Train loss: 0.292681
2020-06-12 03:03:57,106 P13114 INFO Training finished.
2020-06-12 03:03:57,106 P13114 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Avazu/WideDeep_avazu/min2/avazu_x4_3bbbc4c9/WideDeep_avazu_x4_3bbbc4c9_016_2934bd82_model.ckpt
2020-06-12 03:03:57,480 P13114 INFO ****** Train/validation evaluation ******
2020-06-12 03:04:22,176 P13114 INFO [Metrics] logloss: 0.372114 - AUC: 0.792644
2020-06-12 03:04:22,220 P13114 INFO ******** Test evaluation ********
2020-06-12 03:04:22,221 P13114 INFO Loading data...
2020-06-12 03:04:22,221 P13114 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-12 03:04:22,699 P13114 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-12 03:04:22,700 P13114 INFO Loading test data done.
2020-06-12 03:04:47,800 P13114 INFO [Metrics] logloss: 0.371969 - AUC: 0.792869
```
