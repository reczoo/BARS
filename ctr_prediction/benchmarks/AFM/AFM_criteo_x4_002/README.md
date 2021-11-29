## AFM_Criteo_x4_002

A notebook to benchmark AFM on Criteo_x4_002 dataset.

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
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [AFM_criteo_x4_tuner_config_011.yaml](./002/AFM_criteo_x4_tuner_config_011.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/AFM_criteo_x4_tuner_config_011.yaml --tag 010 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.444297 - AUC: 0.807320
```


### Logs
```python
2020-01-05 20:50:18,000 P25726 INFO {
    "attention_dim": "40",
    "attention_dropout": "0.3",
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-6)",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFM",
    "model_id": "AFM_criteo_x4_011_4bd97d26",
    "model_root": "./Criteo/AFM_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
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
    "gpu": "0"
}
2020-01-05 20:50:18,001 P25726 INFO Set up feature encoder...
2020-01-05 20:50:18,001 P25726 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x4_001_be98441d/feature_encoder.pkl
2020-01-05 20:50:33,019 P25726 INFO Loading data...
2020-01-05 20:50:33,024 P25726 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-01-05 20:50:37,701 P25726 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-01-05 20:50:39,606 P25726 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-01-05 20:50:39,738 P25726 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-01-05 20:50:39,739 P25726 INFO Loading train data done.
2020-01-05 20:50:48,029 P25726 INFO **** Start training: 3668 batches/epoch ****
2020-01-05 21:21:25,972 P25726 INFO [Metrics] logloss: 0.456468 - AUC: 0.793440
2020-01-05 21:21:26,045 P25726 INFO Save best model: monitor(max): 0.336972
2020-01-05 21:21:26,931 P25726 INFO --- 3668/3668 batches finished ---
2020-01-05 21:21:27,026 P25726 INFO Train loss: 0.466483
2020-01-05 21:21:27,027 P25726 INFO ************ Epoch=1 end ************
2020-01-05 21:52:07,718 P25726 INFO [Metrics] logloss: 0.454028 - AUC: 0.796419
2020-01-05 21:52:07,804 P25726 INFO Save best model: monitor(max): 0.342391
2020-01-05 21:52:09,613 P25726 INFO --- 3668/3668 batches finished ---
2020-01-05 21:52:09,718 P25726 INFO Train loss: 0.457651
2020-01-05 21:52:09,718 P25726 INFO ************ Epoch=2 end ************
2020-01-05 22:22:50,580 P25726 INFO [Metrics] logloss: 0.452388 - AUC: 0.798116
2020-01-05 22:22:50,682 P25726 INFO Save best model: monitor(max): 0.345728
2020-01-05 22:22:51,770 P25726 INFO --- 3668/3668 batches finished ---
2020-01-05 22:22:51,879 P25726 INFO Train loss: 0.455934
2020-01-05 22:22:51,879 P25726 INFO ************ Epoch=3 end ************
2020-01-05 22:53:27,729 P25726 INFO [Metrics] logloss: 0.451347 - AUC: 0.799376
2020-01-05 22:53:27,830 P25726 INFO Save best model: monitor(max): 0.348029
2020-01-05 22:53:29,653 P25726 INFO --- 3668/3668 batches finished ---
2020-01-05 22:53:29,757 P25726 INFO Train loss: 0.454805
2020-01-05 22:53:29,758 P25726 INFO ************ Epoch=4 end ************
2020-01-05 23:24:07,133 P25726 INFO [Metrics] logloss: 0.450491 - AUC: 0.800319
2020-01-05 23:24:07,234 P25726 INFO Save best model: monitor(max): 0.349828
2020-01-05 23:24:09,060 P25726 INFO --- 3668/3668 batches finished ---
2020-01-05 23:24:09,160 P25726 INFO Train loss: 0.453981
2020-01-05 23:24:09,160 P25726 INFO ************ Epoch=5 end ************
2020-01-05 23:54:45,707 P25726 INFO [Metrics] logloss: 0.449672 - AUC: 0.801245
2020-01-05 23:54:45,774 P25726 INFO Save best model: monitor(max): 0.351573
2020-01-05 23:54:47,576 P25726 INFO --- 3668/3668 batches finished ---
2020-01-05 23:54:47,676 P25726 INFO Train loss: 0.453251
2020-01-05 23:54:47,676 P25726 INFO ************ Epoch=6 end ************
2020-01-06 00:25:21,032 P25726 INFO [Metrics] logloss: 0.449185 - AUC: 0.801914
2020-01-06 00:25:21,099 P25726 INFO Save best model: monitor(max): 0.352729
2020-01-06 00:25:22,887 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 00:25:22,991 P25726 INFO Train loss: 0.452637
2020-01-06 00:25:22,991 P25726 INFO ************ Epoch=7 end ************
2020-01-06 00:56:02,194 P25726 INFO [Metrics] logloss: 0.448708 - AUC: 0.802528
2020-01-06 00:56:02,393 P25726 INFO Save best model: monitor(max): 0.353820
2020-01-06 00:56:04,203 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 00:56:04,306 P25726 INFO Train loss: 0.452140
2020-01-06 00:56:04,306 P25726 INFO ************ Epoch=8 end ************
2020-01-06 01:26:46,049 P25726 INFO [Metrics] logloss: 0.448074 - AUC: 0.803022
2020-01-06 01:26:46,125 P25726 INFO Save best model: monitor(max): 0.354948
2020-01-06 01:26:47,140 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 01:26:47,244 P25726 INFO Train loss: 0.451710
2020-01-06 01:26:47,244 P25726 INFO ************ Epoch=9 end ************
2020-01-06 01:57:31,268 P25726 INFO [Metrics] logloss: 0.447670 - AUC: 0.803557
2020-01-06 01:57:31,374 P25726 INFO Save best model: monitor(max): 0.355887
2020-01-06 01:57:32,429 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 01:57:32,535 P25726 INFO Train loss: 0.451327
2020-01-06 01:57:32,535 P25726 INFO ************ Epoch=10 end ************
2020-01-06 02:28:13,594 P25726 INFO [Metrics] logloss: 0.447259 - AUC: 0.803941
2020-01-06 02:28:13,661 P25726 INFO Save best model: monitor(max): 0.356682
2020-01-06 02:28:15,419 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 02:28:15,524 P25726 INFO Train loss: 0.450962
2020-01-06 02:28:15,525 P25726 INFO ************ Epoch=11 end ************
2020-01-06 02:59:00,695 P25726 INFO [Metrics] logloss: 0.446998 - AUC: 0.804270
2020-01-06 02:59:00,780 P25726 INFO Save best model: monitor(max): 0.357271
2020-01-06 02:59:02,566 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 02:59:02,674 P25726 INFO Train loss: 0.450695
2020-01-06 02:59:02,674 P25726 INFO ************ Epoch=12 end ************
2020-01-06 03:29:43,146 P25726 INFO [Metrics] logloss: 0.446706 - AUC: 0.804546
2020-01-06 03:29:43,215 P25726 INFO Save best model: monitor(max): 0.357840
2020-01-06 03:29:44,961 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 03:29:45,064 P25726 INFO Train loss: 0.450447
2020-01-06 03:29:45,064 P25726 INFO ************ Epoch=13 end ************
2020-01-06 04:00:28,967 P25726 INFO [Metrics] logloss: 0.446414 - AUC: 0.804926
2020-01-06 04:00:29,045 P25726 INFO Save best model: monitor(max): 0.358513
2020-01-06 04:00:30,831 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 04:00:30,941 P25726 INFO Train loss: 0.450203
2020-01-06 04:00:30,942 P25726 INFO ************ Epoch=14 end ************
2020-01-06 04:31:13,782 P25726 INFO [Metrics] logloss: 0.446386 - AUC: 0.804994
2020-01-06 04:31:13,873 P25726 INFO Save best model: monitor(max): 0.358608
2020-01-06 04:31:15,599 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 04:31:15,703 P25726 INFO Train loss: 0.449998
2020-01-06 04:31:15,703 P25726 INFO ************ Epoch=15 end ************
2020-01-06 05:01:55,946 P25726 INFO [Metrics] logloss: 0.445984 - AUC: 0.805353
2020-01-06 05:01:56,029 P25726 INFO Save best model: monitor(max): 0.359369
2020-01-06 05:01:57,002 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 05:01:57,107 P25726 INFO Train loss: 0.449795
2020-01-06 05:01:57,108 P25726 INFO ************ Epoch=16 end ************
2020-01-06 05:32:35,698 P25726 INFO [Metrics] logloss: 0.445812 - AUC: 0.805542
2020-01-06 05:32:35,767 P25726 INFO Save best model: monitor(max): 0.359730
2020-01-06 05:32:37,529 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 05:32:37,639 P25726 INFO Train loss: 0.449619
2020-01-06 05:32:37,639 P25726 INFO ************ Epoch=17 end ************
2020-01-06 06:03:16,790 P25726 INFO [Metrics] logloss: 0.445665 - AUC: 0.805780
2020-01-06 06:03:16,860 P25726 INFO Save best model: monitor(max): 0.360115
2020-01-06 06:03:18,610 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 06:03:18,710 P25726 INFO Train loss: 0.449468
2020-01-06 06:03:18,710 P25726 INFO ************ Epoch=18 end ************
2020-01-06 06:33:55,096 P25726 INFO [Metrics] logloss: 0.445387 - AUC: 0.806031
2020-01-06 06:33:55,195 P25726 INFO Save best model: monitor(max): 0.360643
2020-01-06 06:33:56,987 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 06:33:57,092 P25726 INFO Train loss: 0.449311
2020-01-06 06:33:57,092 P25726 INFO ************ Epoch=19 end ************
2020-01-06 07:04:36,495 P25726 INFO [Metrics] logloss: 0.445405 - AUC: 0.806017
2020-01-06 07:04:36,593 P25726 INFO Monitor(max) STOP: 0.360612 !
2020-01-06 07:04:36,594 P25726 INFO Reduce learning rate on plateau: 0.000100
2020-01-06 07:04:36,594 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 07:04:36,751 P25726 INFO Train loss: 0.449169
2020-01-06 07:04:36,751 P25726 INFO ************ Epoch=20 end ************
2020-01-06 07:35:18,089 P25726 INFO [Metrics] logloss: 0.444704 - AUC: 0.806852
2020-01-06 07:35:18,191 P25726 INFO Save best model: monitor(max): 0.362148
2020-01-06 07:35:20,025 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 07:35:20,122 P25726 INFO Train loss: 0.443295
2020-01-06 07:35:20,122 P25726 INFO ************ Epoch=21 end ************
2020-01-06 08:05:57,299 P25726 INFO [Metrics] logloss: 0.444716 - AUC: 0.806879
2020-01-06 08:05:57,421 P25726 INFO Save best model: monitor(max): 0.362163
2020-01-06 08:05:58,483 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 08:05:58,584 P25726 INFO Train loss: 0.441964
2020-01-06 08:05:58,584 P25726 INFO ************ Epoch=22 end ************
2020-01-06 08:36:39,566 P25726 INFO [Metrics] logloss: 0.444725 - AUC: 0.806859
2020-01-06 08:36:39,806 P25726 INFO Monitor(max) STOP: 0.362134 !
2020-01-06 08:36:39,806 P25726 INFO Reduce learning rate on plateau: 0.000010
2020-01-06 08:36:39,807 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 08:36:39,963 P25726 INFO Train loss: 0.441284
2020-01-06 08:36:39,963 P25726 INFO ************ Epoch=23 end ************
2020-01-06 09:07:19,423 P25726 INFO [Metrics] logloss: 0.444848 - AUC: 0.806757
2020-01-06 09:07:19,495 P25726 INFO Monitor(max) STOP: 0.361909 !
2020-01-06 09:07:19,496 P25726 INFO Reduce learning rate on plateau: 0.000001
2020-01-06 09:07:19,496 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 09:07:19,650 P25726 INFO Train loss: 0.439960
2020-01-06 09:07:19,650 P25726 INFO ************ Epoch=24 end ************
2020-01-06 09:37:58,950 P25726 INFO [Metrics] logloss: 0.444860 - AUC: 0.806748
2020-01-06 09:37:59,016 P25726 INFO Monitor(max) STOP: 0.361888 !
2020-01-06 09:37:59,016 P25726 INFO Reduce learning rate on plateau: 0.000001
2020-01-06 09:37:59,016 P25726 INFO Early stopping at epoch=25
2020-01-06 09:37:59,016 P25726 INFO --- 3668/3668 batches finished ---
2020-01-06 09:37:59,168 P25726 INFO Train loss: 0.439762
2020-01-06 09:37:59,168 P25726 INFO Training finished.
2020-01-06 09:37:59,168 P25726 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Criteo/AFM_criteo/criteo_x4_001_be98441d/AFM_criteo_x4_011_4bd97d26_criteo_x4_001_be98441d_model.ckpt
2020-01-06 09:38:00,338 P25726 INFO ****** Train/validation evaluation ******
2020-01-06 09:47:51,189 P25726 INFO [Metrics] logloss: 0.433299 - AUC: 0.819055
2020-01-06 09:49:03,695 P25726 INFO [Metrics] logloss: 0.444716 - AUC: 0.806879
2020-01-06 09:49:04,049 P25726 INFO ******** Test evaluation ********
2020-01-06 09:49:04,049 P25726 INFO Loading data...
2020-01-06 09:49:04,050 P25726 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-01-06 09:49:04,897 P25726 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-01-06 09:49:04,897 P25726 INFO Loading test data done.
2020-01-06 09:50:18,718 P25726 INFO [Metrics] logloss: 0.444297 - AUC: 0.807320



```
