## AFN+_Criteo_x4_002

A notebook to benchmark AFN+ on Criteo_x4_002 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.
### Code




### Results
```python
[Metrics] logloss: 0.438653 - AUC: 0.813362
```


### Logs
```python
2020-02-13 05:41:08,286 P1618 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.1",
    "afn_hidden_units": "[400, 400, 400]",
    "batch_norm": "True",
    "batch_size": "2000",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[1000, 1000, 1000, 1000]",
    "embedding_dim": "20",
    "embedding_dropout": "0",
    "embedding_regularizer": "3e-06",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1500",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_criteo_x4_005_5dbc185d",
    "model_root": "./Criteo/AFN_criteo/",
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
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "3"
}
2020-02-13 05:41:08,288 P1618 INFO Set up feature encoder...
2020-02-13 05:41:08,288 P1618 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-13 05:41:08,288 P1618 INFO Loading data...
2020-02-13 05:41:08,292 P1618 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-13 05:41:15,605 P1618 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-13 05:41:18,285 P1618 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-13 05:41:18,494 P1618 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-13 05:41:18,494 P1618 INFO Loading train data done.
2020-02-13 05:41:27,215 P1618 INFO **** Start training: 18337 batches/epoch ****
2020-02-13 06:20:02,764 P1618 INFO [Metrics] logloss: 0.445314 - AUC: 0.805968
2020-02-13 06:20:02,898 P1618 INFO Save best model: monitor(max): 0.360653
2020-02-13 06:20:04,231 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 06:20:04,307 P1618 INFO Train loss: 0.460564
2020-02-13 06:20:04,307 P1618 INFO ************ Epoch=1 end ************
2020-02-13 06:58:29,935 P1618 INFO [Metrics] logloss: 0.443338 - AUC: 0.808110
2020-02-13 06:58:30,058 P1618 INFO Save best model: monitor(max): 0.364772
2020-02-13 06:58:31,843 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 06:58:31,915 P1618 INFO Train loss: 0.454344
2020-02-13 06:58:31,915 P1618 INFO ************ Epoch=2 end ************
2020-02-13 07:36:42,636 P1618 INFO [Metrics] logloss: 0.442171 - AUC: 0.809390
2020-02-13 07:36:42,725 P1618 INFO Save best model: monitor(max): 0.367219
2020-02-13 07:36:44,643 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 07:36:44,702 P1618 INFO Train loss: 0.452737
2020-02-13 07:36:44,702 P1618 INFO ************ Epoch=3 end ************
2020-02-13 08:14:45,373 P1618 INFO [Metrics] logloss: 0.441355 - AUC: 0.810164
2020-02-13 08:14:45,470 P1618 INFO Save best model: monitor(max): 0.368809
2020-02-13 08:14:47,653 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 08:14:47,710 P1618 INFO Train loss: 0.451872
2020-02-13 08:14:47,710 P1618 INFO ************ Epoch=4 end ************
2020-02-13 08:53:10,591 P1618 INFO [Metrics] logloss: 0.441399 - AUC: 0.810164
2020-02-13 08:53:10,684 P1618 INFO Monitor(max) STOP: 0.368765 !
2020-02-13 08:53:10,684 P1618 INFO Reduce learning rate on plateau: 0.000100
2020-02-13 08:53:10,684 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 08:53:10,743 P1618 INFO Train loss: 0.451210
2020-02-13 08:53:10,743 P1618 INFO ************ Epoch=5 end ************
2020-02-13 09:31:33,252 P1618 INFO [Metrics] logloss: 0.439039 - AUC: 0.812956
2020-02-13 09:31:33,365 P1618 INFO Save best model: monitor(max): 0.373917
2020-02-13 09:31:35,451 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 09:31:35,515 P1618 INFO Train loss: 0.437730
2020-02-13 09:31:35,515 P1618 INFO ************ Epoch=6 end ************
2020-02-13 10:09:56,152 P1618 INFO [Metrics] logloss: 0.439349 - AUC: 0.812758
2020-02-13 10:09:56,261 P1618 INFO Monitor(max) STOP: 0.373408 !
2020-02-13 10:09:56,262 P1618 INFO Reduce learning rate on plateau: 0.000010
2020-02-13 10:09:56,262 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 10:09:56,322 P1618 INFO Train loss: 0.432507
2020-02-13 10:09:56,323 P1618 INFO ************ Epoch=7 end ************
2020-02-13 10:47:51,311 P1618 INFO [Metrics] logloss: 0.440896 - AUC: 0.811596
2020-02-13 10:47:51,428 P1618 INFO Monitor(max) STOP: 0.370701 !
2020-02-13 10:47:51,428 P1618 INFO Reduce learning rate on plateau: 0.000001
2020-02-13 10:47:51,428 P1618 INFO Early stopping at epoch=8
2020-02-13 10:47:51,428 P1618 INFO --- 18337/18337 batches finished ---
2020-02-13 10:47:51,487 P1618 INFO Train loss: 0.426833
2020-02-13 10:47:51,488 P1618 INFO Training finished.
2020-02-13 10:47:51,488 P1618 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/AFN_criteo/criteo_x4_001_be98441d/AFN_criteo_x4_005_5dbc185d_criteo_x4_001_be98441d_model.ckpt
2020-02-13 10:47:53,409 P1618 INFO ****** Train/validation evaluation ******
2020-02-13 10:57:03,622 P1618 INFO [Metrics] logloss: 0.424487 - AUC: 0.827892
2020-02-13 10:58:12,952 P1618 INFO [Metrics] logloss: 0.439039 - AUC: 0.812956
2020-02-13 10:58:13,199 P1618 INFO ******** Test evaluation ********
2020-02-13 10:58:13,199 P1618 INFO Loading data...
2020-02-13 10:58:13,199 P1618 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-13 10:58:14,531 P1618 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-13 10:58:14,531 P1618 INFO Loading test data done.
2020-02-13 10:59:24,604 P1618 INFO [Metrics] logloss: 0.438653 - AUC: 0.813362

```
