## FmFM_Criteo_x0_001

A notebook to benchmark FmFM on Criteo_x0_001 dataset.

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
[Metrics] AUC: 0.805614 - logloss: 0.446196
```


### Logs
```python
2021-04-15 08:05:27,440 P38371 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x0_ace9c1b9",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "field_interaction_type": "matrixed",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_criteo_x0_001_5dff49e1",
    "model_root": "./Criteo/FmFM_criteo_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x0/test.csv",
    "train_data": "../data/Criteo/Criteo_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-04-15 08:05:27,441 P38371 INFO Set up feature encoder...
2021-04-15 08:05:27,441 P38371 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x0_ace9c1b9/feature_encoder.pkl
2021-04-15 08:05:31,773 P38371 INFO Total number of parameters: 23023577.
2021-04-15 08:05:31,774 P38371 INFO Loading data...
2021-04-15 08:05:31,777 P38371 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/train.h5
2021-04-15 08:05:36,386 P38371 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/valid.h5
2021-04-15 08:05:37,505 P38371 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2021-04-15 08:05:37,505 P38371 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2021-04-15 08:05:37,505 P38371 INFO Loading train data done.
2021-04-15 08:05:37,540 P38371 INFO Start training: 8058 batches/epoch
2021-04-15 08:05:37,540 P38371 INFO ************ Epoch=1 start ************
2021-04-15 08:36:33,517 P38371 INFO [Metrics] AUC: 0.800992 - logloss: 0.450324
2021-04-15 08:36:33,518 P38371 INFO Save best model: monitor(max): 0.800992
2021-04-15 08:36:33,667 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 08:36:33,747 P38371 INFO Train loss: 0.459908
2021-04-15 08:36:33,747 P38371 INFO ************ Epoch=1 end ************
2021-04-15 09:07:19,336 P38371 INFO [Metrics] AUC: 0.803183 - logloss: 0.448210
2021-04-15 09:07:19,338 P38371 INFO Save best model: monitor(max): 0.803183
2021-04-15 09:07:19,479 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 09:07:19,552 P38371 INFO Train loss: 0.453593
2021-04-15 09:07:19,552 P38371 INFO ************ Epoch=2 end ************
2021-04-15 09:38:11,106 P38371 INFO [Metrics] AUC: 0.803730 - logloss: 0.447796
2021-04-15 09:38:11,108 P38371 INFO Save best model: monitor(max): 0.803730
2021-04-15 09:38:11,251 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 09:38:11,320 P38371 INFO Train loss: 0.452169
2021-04-15 09:38:11,320 P38371 INFO ************ Epoch=3 end ************
2021-04-15 10:09:06,277 P38371 INFO [Metrics] AUC: 0.804129 - logloss: 0.447385
2021-04-15 10:09:06,279 P38371 INFO Save best model: monitor(max): 0.804129
2021-04-15 10:09:06,439 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 10:09:06,512 P38371 INFO Train loss: 0.451519
2021-04-15 10:09:06,512 P38371 INFO ************ Epoch=4 end ************
2021-04-15 10:39:46,378 P38371 INFO [Metrics] AUC: 0.804147 - logloss: 0.447332
2021-04-15 10:39:46,380 P38371 INFO Save best model: monitor(max): 0.804147
2021-04-15 10:39:46,535 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 10:39:46,601 P38371 INFO Train loss: 0.451167
2021-04-15 10:39:46,602 P38371 INFO ************ Epoch=5 end ************
2021-04-15 11:10:24,469 P38371 INFO [Metrics] AUC: 0.804258 - logloss: 0.447382
2021-04-15 11:10:24,470 P38371 INFO Save best model: monitor(max): 0.804258
2021-04-15 11:10:24,617 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 11:10:24,698 P38371 INFO Train loss: 0.450914
2021-04-15 11:10:24,698 P38371 INFO ************ Epoch=6 end ************
2021-04-15 11:41:00,157 P38371 INFO [Metrics] AUC: 0.804051 - logloss: 0.447472
2021-04-15 11:41:00,158 P38371 INFO Monitor(max) STOP: 0.804051 !
2021-04-15 11:41:00,158 P38371 INFO Reduce learning rate on plateau: 0.000100
2021-04-15 11:41:00,159 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 11:41:00,230 P38371 INFO Train loss: 0.450760
2021-04-15 11:41:00,230 P38371 INFO ************ Epoch=7 end ************
2021-04-15 12:11:36,583 P38371 INFO [Metrics] AUC: 0.805380 - logloss: 0.446526
2021-04-15 12:11:36,585 P38371 INFO Save best model: monitor(max): 0.805380
2021-04-15 12:11:36,740 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 12:11:36,823 P38371 INFO Train loss: 0.439168
2021-04-15 12:11:36,823 P38371 INFO ************ Epoch=8 end ************
2021-04-15 12:42:22,179 P38371 INFO [Metrics] AUC: 0.804912 - logloss: 0.447083
2021-04-15 12:42:22,180 P38371 INFO Monitor(max) STOP: 0.804912 !
2021-04-15 12:42:22,181 P38371 INFO Reduce learning rate on plateau: 0.000010
2021-04-15 12:42:22,181 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 12:42:22,257 P38371 INFO Train loss: 0.436202
2021-04-15 12:42:22,257 P38371 INFO ************ Epoch=9 end ************
2021-04-15 13:13:03,726 P38371 INFO [Metrics] AUC: 0.804810 - logloss: 0.447215
2021-04-15 13:13:03,728 P38371 INFO Monitor(max) STOP: 0.804810 !
2021-04-15 13:13:03,728 P38371 INFO Reduce learning rate on plateau: 0.000001
2021-04-15 13:13:03,728 P38371 INFO Early stopping at epoch=10
2021-04-15 13:13:03,728 P38371 INFO --- 8058/8058 batches finished ---
2021-04-15 13:13:03,800 P38371 INFO Train loss: 0.432729
2021-04-15 13:13:03,800 P38371 INFO Training finished.
2021-04-15 13:13:03,800 P38371 INFO Load best model: /home/xxx/xxx/FuxiCTR_v2/FuxiCTR/benchmarks/Criteo/FmFM_criteo_x0/criteo_x0_ace9c1b9/FmFM_criteo_x0_001_5dff49e1.model
2021-04-15 13:13:03,916 P38371 INFO ****** Train/validation evaluation ******
2021-04-15 13:15:25,356 P38371 INFO [Metrics] AUC: 0.805380 - logloss: 0.446526
2021-04-15 13:15:25,391 P38371 INFO ******** Test evaluation ********
2021-04-15 13:15:25,391 P38371 INFO Loading data...
2021-04-15 13:15:25,391 P38371 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/test.h5
2021-04-15 13:15:26,070 P38371 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2021-04-15 13:15:26,070 P38371 INFO Loading test data done.
2021-04-15 13:16:44,429 P38371 INFO [Metrics] AUC: 0.805614 - logloss: 0.446196

```
