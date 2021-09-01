## CCPM_Avazu_x4_001

A notebook to benchmark CCPM on Avazu_x4_001 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default <OOV> token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless id field nor specially preprocess the timestamp field.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.


### Code




### Results
```python
[Metrics] logloss: 0.374494 - AUC: 0.789239
```


### Logs
```python
2020-06-26 14:49:22,751 P7122 INFO {
    "activation": "Tanh",
    "batch_size": "10000",
    "channels": "[128, 256, 512]",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "kernel_heights": "[7, 5, 3]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "CCPM",
    "model_id": "CCPM_avazu_x4_3bbbc4c9_017_16dce5de",
    "model_root": "./Avazu/CCPM_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
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
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-26 14:49:22,753 P7122 INFO Set up feature encoder...
2020-06-26 14:49:22,753 P7122 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-26 14:49:22,754 P7122 INFO Loading data...
2020-06-26 14:49:22,758 P7122 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-26 14:49:26,291 P7122 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-26 14:49:27,631 P7122 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-26 14:49:27,730 P7122 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-26 14:49:27,730 P7122 INFO Loading train data done.
2020-06-26 14:50:11,611 P7122 INFO Start training: 3235 batches/epoch
2020-06-26 14:50:11,611 P7122 INFO ************ Epoch=1 start ************
2020-06-26 18:25:57,638 P7122 INFO [Metrics] logloss: 0.374473 - AUC: 0.789223
2020-06-26 18:25:57,639 P7122 INFO Save best model: monitor(max): 0.414750
2020-06-26 18:25:58,001 P7122 INFO --- 3235/3235 batches finished ---
2020-06-26 18:25:58,053 P7122 INFO Train loss: 0.387957
2020-06-26 18:25:58,054 P7122 INFO ************ Epoch=1 end ************
2020-06-26 21:59:31,656 P7122 INFO [Metrics] logloss: 0.378156 - AUC: 0.788378
2020-06-26 21:59:31,659 P7122 INFO Monitor(max) STOP: 0.410222 !
2020-06-26 21:59:31,659 P7122 INFO Reduce learning rate on plateau: 0.000100
2020-06-26 21:59:31,659 P7122 INFO --- 3235/3235 batches finished ---
2020-06-26 21:59:31,709 P7122 INFO Train loss: 0.342339
2020-06-26 21:59:31,709 P7122 INFO ************ Epoch=2 end ************
2020-06-27 01:38:33,799 P7122 INFO [Metrics] logloss: 0.401963 - AUC: 0.777150
2020-06-27 01:38:33,802 P7122 INFO Monitor(max) STOP: 0.375187 !
2020-06-27 01:38:33,802 P7122 INFO Reduce learning rate on plateau: 0.000010
2020-06-27 01:38:33,804 P7122 INFO Early stopping at epoch=3
2020-06-27 01:38:33,804 P7122 INFO --- 3235/3235 batches finished ---
2020-06-27 01:38:33,854 P7122 INFO Train loss: 0.298888
2020-06-27 01:38:33,855 P7122 INFO Training finished.
2020-06-27 01:38:33,855 P7122 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/CCPM_avazu/min2/avazu_x4_3bbbc4c9/CCPM_avazu_x4_3bbbc4c9_017_16dce5de_model.ckpt
2020-06-27 01:38:34,395 P7122 INFO ****** Train/validation evaluation ******
2020-06-27 04:26:32,179 P7122 INFO [Metrics] logloss: 0.342947 - AUC: 0.839513
2020-06-27 04:47:42,986 P7122 INFO [Metrics] logloss: 0.374473 - AUC: 0.789223
2020-06-27 04:47:43,047 P7122 INFO ******** Test evaluation ********
2020-06-27 04:47:43,047 P7122 INFO Loading data...
2020-06-27 04:47:43,047 P7122 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-27 04:47:43,773 P7122 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-27 04:47:43,773 P7122 INFO Loading test data done.
2020-06-27 05:08:22,545 P7122 INFO [Metrics] logloss: 0.374494 - AUC: 0.789239


```
