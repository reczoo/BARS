## FwFM_Criteo_x4_002

A notebook to benchmark FwFM on Criteo_x4_002 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=2.

We fix embedding_dim=40 in this setting.
### Code




### Results
```python
[Metrics] logloss: 0.441891 - AUC: 0.809848
```


### Logs
```python
2020-02-29 09:25:04,623 P3445 INFO {
    "batch_size": "5000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "linear_type": "FiLV",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FwFM",
    "model_id": "FwFM_criteo_x4_001_49c28217",
    "model_root": "./Criteo/FwFM_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "0",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "1",
    "workers": "3",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'na_value': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'na_value': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "min_categr_count": "2",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "version": "pytorch",
    "gpu": "1"
}
2020-02-29 09:25:04,624 P3445 INFO Set up feature encoder...
2020-02-29 09:25:04,624 P3445 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x4_001_be98441d/feature_encoder.pkl
2020-02-29 09:25:19,590 P3445 INFO Loading data...
2020-02-29 09:25:19,748 P3445 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-29 09:25:24,146 P3445 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-29 09:25:25,780 P3445 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-29 09:25:25,912 P3445 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-29 09:25:25,912 P3445 INFO Loading train data done.
2020-02-29 09:25:33,973 P3445 INFO **** Start training: 7335 batches/epoch ****
2020-02-29 09:44:59,017 P3445 INFO [Metrics] logloss: 0.442250 - AUC: 0.809434
2020-02-29 09:44:59,107 P3445 INFO Save best model: monitor(max): 0.367184
2020-02-29 09:44:59,915 P3445 INFO --- 7335/7335 batches finished ---
2020-02-29 09:45:00,053 P3445 INFO Train loss: 0.448647
2020-02-29 09:45:00,053 P3445 INFO ************ Epoch=1 end ************
2020-02-29 10:04:16,699 P3445 INFO [Metrics] logloss: 0.457104 - AUC: 0.798466
2020-02-29 10:04:16,801 P3445 INFO Monitor(max) STOP: 0.341363 !
2020-02-29 10:04:16,801 P3445 INFO Reduce learning rate on plateau: 0.000100
2020-02-29 10:04:16,801 P3445 INFO --- 7335/7335 batches finished ---
2020-02-29 10:04:16,943 P3445 INFO Train loss: 0.408067
2020-02-29 10:04:16,943 P3445 INFO ************ Epoch=2 end ************
2020-02-29 10:23:26,042 P3445 INFO [Metrics] logloss: 0.485622 - AUC: 0.784681
2020-02-29 10:23:26,142 P3445 INFO Monitor(max) STOP: 0.299059 !
2020-02-29 10:23:26,143 P3445 INFO Reduce learning rate on plateau: 0.000010
2020-02-29 10:23:26,143 P3445 INFO Early stopping at epoch=3
2020-02-29 10:23:26,143 P3445 INFO --- 7335/7335 batches finished ---
2020-02-29 10:23:26,255 P3445 INFO Train loss: 0.342411
2020-02-29 10:23:26,255 P3445 INFO Training finished.
2020-02-29 10:23:26,255 P3445 INFO Load best model: /home/zhujieming/xxx/OpenCTR1030/benchmarks/Criteo/FwFM_criteo/criteo_x4_001_be98441d/FwFM_criteo_x4_001_49c28217_model.ckpt
2020-02-29 10:23:27,207 P3445 INFO ****** Train/validation evaluation ******
2020-02-29 10:31:21,462 P3445 INFO [Metrics] logloss: 0.421044 - AUC: 0.832439
2020-02-29 10:32:21,591 P3445 INFO [Metrics] logloss: 0.442250 - AUC: 0.809434
2020-02-29 10:32:22,169 P3445 INFO ******** Test evaluation ********
2020-02-29 10:32:22,170 P3445 INFO Loading data...
2020-02-29 10:32:22,170 P3445 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-29 10:32:22,941 P3445 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-29 10:32:22,941 P3445 INFO Loading test data done.
2020-02-29 10:33:22,813 P3445 INFO [Metrics] logloss: 0.441891 - AUC: 0.809848


```