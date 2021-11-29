## YoutubeDNN_Avazu_x0_001

A notebook to benchmark YoutubeDNN on Avazu_x0_001 dataset.

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
This dataset split follows the setting in the AFN work. That is, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. The data preprocessing script is provided on Github and we directly download the preprocessed data.
In this setting, we follow the AFN work to fix embedding_dim=16, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons.


### Code




### Results
```python
[Metrics] logloss: 0.367068 - AUC: 0.764551
```


### Logs
```python
2020-12-28 20:21:12,834 P24912 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x0_83355fc7",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "DNN",
    "model_id": "DNN_avazu_x0_012_0dfebcdd",
    "model_root": "./Avazu/DNN_avazu_x0/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x0/test.csv",
    "train_data": "../data/Avazu/Avazu_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2020-12-28 20:21:12,835 P24912 INFO Set up feature encoder...
2020-12-28 20:21:12,835 P24912 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x0_83355fc7/feature_encoder.pkl
2020-12-28 20:21:13,870 P24912 INFO Total number of parameters: 13395591.
2020-12-28 20:21:13,870 P24912 INFO Loading data...
2020-12-28 20:21:13,873 P24912 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/train.h5
2020-12-28 20:21:16,548 P24912 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/valid.h5
2020-12-28 20:21:16,926 P24912 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2020-12-28 20:21:16,926 P24912 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2020-12-28 20:21:16,926 P24912 INFO Loading train data done.
2020-12-28 20:21:21,915 P24912 INFO Start training: 6910 batches/epoch
2020-12-28 20:21:21,915 P24912 INFO ************ Epoch=1 start ************
2020-12-28 21:05:36,282 P24912 INFO [Metrics] logloss: 0.399131 - AUC: 0.740126
2020-12-28 21:05:36,286 P24912 INFO Save best model: monitor(max): 0.340995
2020-12-28 21:05:36,340 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 21:05:36,383 P24912 INFO Train loss: 0.429047
2020-12-28 21:05:36,384 P24912 INFO ************ Epoch=1 end ************
2020-12-28 21:34:47,930 P24912 INFO [Metrics] logloss: 0.398655 - AUC: 0.741279
2020-12-28 21:34:47,934 P24912 INFO Save best model: monitor(max): 0.342624
2020-12-28 21:34:48,018 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 21:34:48,078 P24912 INFO Train loss: 0.429014
2020-12-28 21:34:48,078 P24912 INFO ************ Epoch=2 end ************
2020-12-28 22:01:44,655 P24912 INFO [Metrics] logloss: 0.398752 - AUC: 0.741589
2020-12-28 22:01:44,659 P24912 INFO Save best model: monitor(max): 0.342837
2020-12-28 22:01:44,751 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 22:01:44,814 P24912 INFO Train loss: 0.428438
2020-12-28 22:01:44,814 P24912 INFO ************ Epoch=3 end ************
2020-12-28 22:28:39,092 P24912 INFO [Metrics] logloss: 0.398844 - AUC: 0.741310
2020-12-28 22:28:39,096 P24912 INFO Monitor(max) STOP: 0.342466 !
2020-12-28 22:28:39,096 P24912 INFO Reduce learning rate on plateau: 0.000100
2020-12-28 22:28:39,096 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 22:28:39,157 P24912 INFO Train loss: 0.428715
2020-12-28 22:28:39,158 P24912 INFO ************ Epoch=4 end ************
2020-12-28 22:53:33,884 P24912 INFO [Metrics] logloss: 0.397738 - AUC: 0.744362
2020-12-28 22:53:33,887 P24912 INFO Save best model: monitor(max): 0.346624
2020-12-28 22:53:33,974 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 22:53:34,038 P24912 INFO Train loss: 0.404087
2020-12-28 22:53:34,038 P24912 INFO ************ Epoch=5 end ************
2020-12-28 23:15:10,103 P24912 INFO [Metrics] logloss: 0.396411 - AUC: 0.745814
2020-12-28 23:15:10,105 P24912 INFO Save best model: monitor(max): 0.349403
2020-12-28 23:15:10,189 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 23:15:10,245 P24912 INFO Train loss: 0.404804
2020-12-28 23:15:10,245 P24912 INFO ************ Epoch=6 end ************
2020-12-28 23:34:06,364 P24912 INFO [Metrics] logloss: 0.397119 - AUC: 0.745171
2020-12-28 23:34:06,366 P24912 INFO Monitor(max) STOP: 0.348052 !
2020-12-28 23:34:06,367 P24912 INFO Reduce learning rate on plateau: 0.000010
2020-12-28 23:34:06,367 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 23:34:06,425 P24912 INFO Train loss: 0.404994
2020-12-28 23:34:06,426 P24912 INFO ************ Epoch=7 end ************
2020-12-28 23:51:28,954 P24912 INFO [Metrics] logloss: 0.398896 - AUC: 0.743046
2020-12-28 23:51:28,958 P24912 INFO Monitor(max) STOP: 0.344149 !
2020-12-28 23:51:28,958 P24912 INFO Reduce learning rate on plateau: 0.000001
2020-12-28 23:51:28,958 P24912 INFO Early stopping at epoch=8
2020-12-28 23:51:28,958 P24912 INFO --- 6910/6910 batches finished ---
2020-12-28 23:51:29,012 P24912 INFO Train loss: 0.392950
2020-12-28 23:51:29,013 P24912 INFO Training finished.
2020-12-28 23:51:29,013 P24912 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Avazu/DNN_avazu_x0/avazu_x0_83355fc7/DNN_avazu_x0_012_0dfebcdd_model.ckpt
2020-12-28 23:51:29,156 P24912 INFO ****** Train/validation evaluation ******
2020-12-28 23:51:43,561 P24912 INFO [Metrics] logloss: 0.396411 - AUC: 0.745814
2020-12-28 23:51:43,647 P24912 INFO ******** Test evaluation ********
2020-12-28 23:51:43,647 P24912 INFO Loading data...
2020-12-28 23:51:43,648 P24912 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/test.h5
2020-12-28 23:51:44,402 P24912 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2020-12-28 23:51:44,402 P24912 INFO Loading test data done.
2020-12-28 23:52:13,309 P24912 INFO [Metrics] logloss: 0.367068 - AUC: 0.764551


```
