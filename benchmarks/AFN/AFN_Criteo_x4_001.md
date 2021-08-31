## AFN_Criteo_x4_001

A notebook to benchmark AFN on Criteo_x4_001 dataset.

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
This dataset split follows the setting in the AutoInt work. Specifically, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting.

### Code




### Results
```python
[Metrics] logloss: 0.440225 - AUC: 0.811513
```


### Logs
```python
2020-07-24 02:25:36,558 P13372 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.2",
    "afn_hidden_units": "[1000, 1000, 1000, 1000]",
    "batch_norm": "True",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1200",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_criteo_x4_5c863b0f_012_363a38e8",
    "model_root": "./Criteo/AFN_criteo/min10/",
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
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-24 02:25:36,559 P13372 INFO Set up feature encoder...
2020-07-24 02:25:36,559 P13372 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-24 02:25:40,655 P13372 INFO Total number of parameters: 36833607.
2020-07-24 02:25:40,655 P13372 INFO Loading data...
2020-07-24 02:25:40,657 P13372 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-24 02:25:50,773 P13372 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-24 02:25:54,459 P13372 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-24 02:25:54,702 P13372 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-24 02:25:54,703 P13372 INFO Loading train data done.
2020-07-24 02:26:37,624 P13372 INFO Start training: 3668 batches/epoch
2020-07-24 02:26:37,624 P13372 INFO ************ Epoch=1 start ************
2020-07-24 02:48:16,994 P13372 INFO [Metrics] logloss: 0.450781 - AUC: 0.800109
2020-07-24 02:48:16,997 P13372 INFO Save best model: monitor(max): 0.349328
2020-07-24 02:48:17,153 P13372 INFO --- 3668/3668 batches finished ---
2020-07-24 02:48:17,269 P13372 INFO Train loss: 0.460929
2020-07-24 02:48:17,269 P13372 INFO ************ Epoch=1 end ************
2020-07-24 03:09:56,141 P13372 INFO [Metrics] logloss: 0.444814 - AUC: 0.806515
2020-07-24 03:09:56,142 P13372 INFO Save best model: monitor(max): 0.361701
2020-07-24 03:09:56,410 P13372 INFO --- 3668/3668 batches finished ---
2020-07-24 03:09:56,535 P13372 INFO Train loss: 0.450058
2020-07-24 03:09:56,535 P13372 INFO ************ Epoch=2 end ************
2020-07-24 03:31:30,987 P13372 INFO [Metrics] logloss: 0.442434 - AUC: 0.809230
2020-07-24 03:31:30,998 P13372 INFO Save best model: monitor(max): 0.366796
2020-07-24 03:31:32,805 P13372 INFO --- 3668/3668 batches finished ---
2020-07-24 03:31:32,970 P13372 INFO Train loss: 0.446568
2020-07-24 03:31:32,970 P13372 INFO ************ Epoch=3 end ************
2020-07-24 03:53:20,605 P13372 INFO [Metrics] logloss: 0.441153 - AUC: 0.810500
2020-07-24 03:53:20,606 P13372 INFO Save best model: monitor(max): 0.369346
2020-07-24 03:53:20,890 P13372 INFO --- 3668/3668 batches finished ---
2020-07-24 03:53:21,021 P13372 INFO Train loss: 0.444205
2020-07-24 03:53:21,021 P13372 INFO ************ Epoch=4 end ************
2020-07-24 04:14:47,960 P13372 INFO [Metrics] logloss: 0.440639 - AUC: 0.811021
2020-07-24 04:14:47,962 P13372 INFO Save best model: monitor(max): 0.370382
2020-07-24 04:14:48,259 P13372 INFO --- 3668/3668 batches finished ---
2020-07-24 04:14:48,388 P13372 INFO Train loss: 0.442115
2020-07-24 04:14:48,388 P13372 INFO ************ Epoch=5 end ************
2020-07-24 04:36:21,514 P13372 INFO [Metrics] logloss: 0.440984 - AUC: 0.810871
2020-07-24 04:36:21,515 P13372 INFO Monitor(max) STOP: 0.369887 !
2020-07-24 04:36:21,515 P13372 INFO Reduce learning rate on plateau: 0.000100
2020-07-24 04:36:21,515 P13372 INFO --- 3668/3668 batches finished ---
2020-07-24 04:36:21,628 P13372 INFO Train loss: 0.439794
2020-07-24 04:36:21,628 P13372 INFO ************ Epoch=6 end ************
2020-07-24 04:57:57,204 P13372 INFO [Metrics] logloss: 0.442888 - AUC: 0.809799
2020-07-24 04:57:57,205 P13372 INFO Monitor(max) STOP: 0.366912 !
2020-07-24 04:57:57,205 P13372 INFO Reduce learning rate on plateau: 0.000010
2020-07-24 04:57:57,205 P13372 INFO Early stopping at epoch=7
2020-07-24 04:57:57,205 P13372 INFO --- 3668/3668 batches finished ---
2020-07-24 04:57:57,327 P13372 INFO Train loss: 0.428021
2020-07-24 04:57:57,327 P13372 INFO Training finished.
2020-07-24 04:57:57,327 P13372 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Criteo/AFN_criteo/min10/criteo_x4_5c863b0f/AFN_criteo_x4_5c863b0f_012_363a38e8_model.ckpt
2020-07-24 04:57:57,537 P13372 INFO ****** Train/validation evaluation ******
2020-07-24 05:06:23,618 P13372 INFO [Metrics] logloss: 0.427785 - AUC: 0.825324
2020-07-24 05:07:17,836 P13372 INFO [Metrics] logloss: 0.440639 - AUC: 0.811021
2020-07-24 05:07:17,876 P13372 INFO ******** Test evaluation ********
2020-07-24 05:07:17,876 P13372 INFO Loading data...
2020-07-24 05:07:17,877 P13372 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-24 05:07:20,996 P13372 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-24 05:07:20,996 P13372 INFO Loading test data done.
2020-07-24 05:08:20,419 P13372 INFO [Metrics] logloss: 0.440225 - AUC: 0.811513

```
