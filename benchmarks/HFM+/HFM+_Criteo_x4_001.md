## HFM+_Criteo_x4_001 

A notebook to benchmark HFM+ on Criteo_x4_001 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2
(x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better. 

To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.


### Code


### Results
```python
[Metrics] logloss: 0.439178 - AUC: 0.812710
```


### Logs
```python
2020-07-27 00:58:57,647 P13799 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_criteo_x4_5c863b0f_006_e5c408ac",
    "model_root": "./Criteo/HFM_criteo/min10/",
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
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-27 00:58:57,647 P13799 INFO Set up feature encoder...
2020-07-27 00:58:57,647 P13799 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-27 00:58:57,648 P13799 INFO Loading data...
2020-07-27 00:58:57,650 P13799 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-27 00:59:03,009 P13799 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-27 00:59:05,045 P13799 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-27 00:59:05,190 P13799 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-27 00:59:05,190 P13799 INFO Loading train data done.
2020-07-27 00:59:08,558 P13799 INFO **** Start training: 3668 batches/epoch ****
2020-07-27 01:39:44,038 P13799 INFO [Metrics] logloss: 0.441114 - AUC: 0.810581
2020-07-27 01:39:44,039 P13799 INFO Save best model: monitor(max): 0.369467
2020-07-27 01:39:44,150 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 01:39:44,198 P13799 INFO Train loss: 0.451195
2020-07-27 01:39:44,198 P13799 INFO ************ Epoch=1 end ************
2020-07-27 02:20:19,589 P13799 INFO [Metrics] logloss: 0.439555 - AUC: 0.812258
2020-07-27 02:20:19,590 P13799 INFO Save best model: monitor(max): 0.372703
2020-07-27 02:20:19,794 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 02:20:19,844 P13799 INFO Train loss: 0.442213
2020-07-27 02:20:19,844 P13799 INFO ************ Epoch=2 end ************
2020-07-27 03:00:55,418 P13799 INFO [Metrics] logloss: 0.441363 - AUC: 0.810539
2020-07-27 03:00:55,419 P13799 INFO Monitor(max) STOP: 0.369176 !
2020-07-27 03:00:55,419 P13799 INFO Reduce learning rate on plateau: 0.000100
2020-07-27 03:00:55,419 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 03:00:55,471 P13799 INFO Train loss: 0.436734
2020-07-27 03:00:55,471 P13799 INFO ************ Epoch=3 end ************
2020-07-27 03:41:36,566 P13799 INFO [Metrics] logloss: 0.476991 - AUC: 0.790653
2020-07-27 03:41:36,567 P13799 INFO Monitor(max) STOP: 0.313663 !
2020-07-27 03:41:36,567 P13799 INFO Reduce learning rate on plateau: 0.000010
2020-07-27 03:41:36,567 P13799 INFO Early stopping at epoch=4
2020-07-27 03:41:36,567 P13799 INFO --- 3668/3668 batches finished ---
2020-07-27 03:41:36,617 P13799 INFO Train loss: 0.393167
2020-07-27 03:41:36,617 P13799 INFO Training finished.
2020-07-27 03:41:36,617 P13799 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Criteo/HFM_criteo/min10/criteo_x4_5c863b0f/HFM_criteo_x4_5c863b0f_006_e5c408ac_model.ckpt
2020-07-27 03:41:36,777 P13799 INFO ****** Train/validation evaluation ******
2020-07-27 03:43:07,618 P13799 INFO [Metrics] logloss: 0.439555 - AUC: 0.812258
2020-07-27 03:43:07,733 P13799 INFO ******** Test evaluation ********
2020-07-27 03:43:07,733 P13799 INFO Loading data...
2020-07-27 03:43:07,733 P13799 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-27 03:43:08,529 P13799 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-27 03:43:08,530 P13799 INFO Loading test data done.
2020-07-27 03:44:39,219 P13799 INFO [Metrics] logloss: 0.439178 - AUC: 0.812710
```
