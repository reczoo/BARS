## DCN_Criteo_x0_001

A notebook to benchmark DCN on Criteo_x0_001 dataset.

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

Reproducing steps:
Step1: Download the preprocessed data via the [https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Avazu_x0/download_criteo_x0.py](script).

Criteo_x0_001
In this setting, we follow the AFN work to fix embedding_dim=16, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons.

### Code



### Results
```python
[Metrics] AUC: 0.814118 - logloss: 0.437875
```

### Logs
```python
2021-08-15 01:10:31,897 P46215 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x0_ace9c1b9",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN_v2",
    "model_id": "DCN_v2_criteo_x0_005_6971f7bc",
    "model_root": "./Criteo/DCN_criteo_x0/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[500, 500, 500]",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[400, 400, 400]",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x0/test.csv",
    "train_data": "../data/Criteo/Criteo_x0/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Criteo/Criteo_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-08-15 01:10:31,898 P46215 INFO Set up feature encoder...
2021-08-15 01:10:31,898 P46215 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x0_ace9c1b9/feature_encoder.pkl
2021-08-15 01:10:33,851 P46215 INFO Total number of parameters: 22018021.
2021-08-15 01:10:33,852 P46215 INFO Loading data...
2021-08-15 01:10:33,856 P46215 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/train.h5
2021-08-15 01:10:49,171 P46215 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/valid.h5
2021-08-15 01:11:17,094 P46215 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2021-08-15 01:11:17,094 P46215 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2021-08-15 01:11:17,094 P46215 INFO Loading train data done.
2021-08-15 01:11:25,525 P46215 INFO Start training: 8058 batches/epoch
2021-08-15 01:11:25,525 P46215 INFO ************ Epoch=1 start ************
2021-08-15 01:35:53,655 P46215 INFO [Metrics] AUC: 0.805631 - logloss: 0.445776
2021-08-15 01:35:53,670 P46215 INFO Save best model: monitor(max): 0.805631
2021-08-15 01:35:54,093 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 01:35:54,432 P46215 INFO Train loss: 0.460219
2021-08-15 01:35:54,432 P46215 INFO ************ Epoch=1 end ************
2021-08-15 01:59:42,249 P46215 INFO [Metrics] AUC: 0.807809 - logloss: 0.443784
2021-08-15 01:59:42,268 P46215 INFO Save best model: monitor(max): 0.807809
2021-08-15 01:59:42,404 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 01:59:43,088 P46215 INFO Train loss: 0.453981
2021-08-15 01:59:43,088 P46215 INFO ************ Epoch=2 end ************
2021-08-15 02:23:00,519 P46215 INFO [Metrics] AUC: 0.809119 - logloss: 0.442565
2021-08-15 02:23:00,531 P46215 INFO Save best model: monitor(max): 0.809119
2021-08-15 02:23:00,689 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 02:23:01,017 P46215 INFO Train loss: 0.452322
2021-08-15 02:23:01,017 P46215 INFO ************ Epoch=3 end ************
2021-08-15 02:46:09,497 P46215 INFO [Metrics] AUC: 0.809660 - logloss: 0.442067
2021-08-15 02:46:09,506 P46215 INFO Save best model: monitor(max): 0.809660
2021-08-15 02:46:09,659 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 02:46:10,010 P46215 INFO Train loss: 0.451470
2021-08-15 02:46:10,010 P46215 INFO ************ Epoch=4 end ************
2021-08-15 03:08:57,182 P46215 INFO [Metrics] AUC: 0.809980 - logloss: 0.441693
2021-08-15 03:08:57,192 P46215 INFO Save best model: monitor(max): 0.809980
2021-08-15 03:08:57,348 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 03:08:57,645 P46215 INFO Train loss: 0.450895
2021-08-15 03:08:57,645 P46215 INFO ************ Epoch=5 end ************
2021-08-15 03:31:41,821 P46215 INFO [Metrics] AUC: 0.810293 - logloss: 0.441445
2021-08-15 03:31:41,829 P46215 INFO Save best model: monitor(max): 0.810293
2021-08-15 03:31:41,969 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 03:31:42,194 P46215 INFO Train loss: 0.450494
2021-08-15 03:31:42,194 P46215 INFO ************ Epoch=6 end ************
2021-08-15 03:54:40,788 P46215 INFO [Metrics] AUC: 0.810259 - logloss: 0.441564
2021-08-15 03:54:40,801 P46215 INFO Monitor(max) STOP: 0.810259 !
2021-08-15 03:54:40,801 P46215 INFO Reduce learning rate on plateau: 0.000100
2021-08-15 03:54:40,801 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 03:54:41,036 P46215 INFO Train loss: 0.450127
2021-08-15 03:54:41,036 P46215 INFO ************ Epoch=7 end ************
2021-08-15 04:16:26,566 P46215 INFO [Metrics] AUC: 0.813420 - logloss: 0.438577
2021-08-15 04:16:26,573 P46215 INFO Save best model: monitor(max): 0.813420
2021-08-15 04:16:26,705 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 04:16:26,898 P46215 INFO Train loss: 0.439104
2021-08-15 04:16:26,898 P46215 INFO ************ Epoch=8 end ************
2021-08-15 04:38:19,082 P46215 INFO [Metrics] AUC: 0.813803 - logloss: 0.438303
2021-08-15 04:38:19,090 P46215 INFO Save best model: monitor(max): 0.813803
2021-08-15 04:38:19,236 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 04:38:19,444 P46215 INFO Train loss: 0.435280
2021-08-15 04:38:19,444 P46215 INFO ************ Epoch=9 end ************
2021-08-15 05:00:11,176 P46215 INFO [Metrics] AUC: 0.813852 - logloss: 0.438296
2021-08-15 05:00:11,184 P46215 INFO Save best model: monitor(max): 0.813852
2021-08-15 05:00:11,323 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 05:00:11,517 P46215 INFO Train loss: 0.433623
2021-08-15 05:00:11,517 P46215 INFO ************ Epoch=10 end ************
2021-08-15 05:23:05,408 P46215 INFO [Metrics] AUC: 0.813782 - logloss: 0.438429
2021-08-15 05:23:05,421 P46215 INFO Monitor(max) STOP: 0.813782 !
2021-08-15 05:23:05,421 P46215 INFO Reduce learning rate on plateau: 0.000010
2021-08-15 05:23:05,421 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 05:23:05,633 P46215 INFO Train loss: 0.432388
2021-08-15 05:23:05,633 P46215 INFO ************ Epoch=11 end ************
2021-08-15 05:46:15,391 P46215 INFO [Metrics] AUC: 0.813571 - logloss: 0.438850
2021-08-15 05:46:15,409 P46215 INFO Monitor(max) STOP: 0.813571 !
2021-08-15 05:46:15,409 P46215 INFO Reduce learning rate on plateau: 0.000001
2021-08-15 05:46:15,409 P46215 INFO Early stopping at epoch=12
2021-08-15 05:46:15,409 P46215 INFO --- 8058/8058 batches finished ---
2021-08-15 05:46:15,745 P46215 INFO Train loss: 0.428479
2021-08-15 05:46:15,745 P46215 INFO Training finished.
2021-08-15 05:46:15,745 P46215 INFO Load best model: /home/xxx/xxx/FuxiCTR_v2/FuxiCTR/benchmarks/Criteo/DCN_criteo_x0/criteo_x0_ace9c1b9/DCN_v2_criteo_x0_005_6971f7bc.model
2021-08-15 05:46:28,142 P46215 INFO ****** Train/validation evaluation ******
2021-08-15 05:47:29,258 P46215 INFO [Metrics] AUC: 0.813852 - logloss: 0.438296
2021-08-15 05:47:29,782 P46215 INFO ******** Test evaluation ********
2021-08-15 05:47:29,783 P46215 INFO Loading data...
2021-08-15 05:47:29,783 P46215 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/test.h5
2021-08-15 05:47:30,622 P46215 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2021-08-15 05:47:30,622 P46215 INFO Loading test data done.
2021-08-15 05:47:55,355 P46215 INFO [Metrics] AUC: 0.814118 - logloss: 0.437875


```