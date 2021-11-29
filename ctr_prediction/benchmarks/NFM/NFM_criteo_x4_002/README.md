## NFM_Criteo_x4_002

A notebook to benchmark NFM on Criteo_x4_002 dataset.

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
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [NFM_criteo_x4_tuner_config_01.yaml](./002/NFM_criteo_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/NFM_criteo_x4_tuner_config_01.yaml --tag 023 --gpu 0
  ```
### Results
```python
[Metrics] logloss: 0.444309 - AUC: 0.807179
```


### Logs
```python
2019-11-23 03:05:51,900 P6485 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_0d63c1a1",
    "dropout_rates": "0",
    "embedding_dim": "40",
    "embedding_regularizer": "l2(1.e-7)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "kernel_regularizer": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "NFM",
    "model_dir": "./Criteo/",
    "model_id": "NFM_criteo_x4_023_90554c26",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'na_value': 0, 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'preprocess': 'convert_to_bucket', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'na_value': '', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "label_col": "{'dtype': 'float', 'name': 'Label'}",
    "min_categr_count": "2",
    "test_data": "../data/Criteo/Criteo_x4/test.csv",
    "train_data": "../data/Criteo/Criteo_x4/train.csv",
    "valid_data": "../data/Criteo/Criteo_x4/valid.csv",
    "version": "pytorch",
    "device": "0"
}
2019-11-23 03:05:51,900 P6485 INFO Set up feature encoder...
2019-11-23 03:06:06,944 P6485 INFO Load feature encoder cache from ./Criteo/criteo_x4_001_0d63c1a1/feature_encoder.pkl
2019-11-23 03:06:06,944 P6485 INFO Loading data...
2019-11-23 03:06:07,106 P6485 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/train.hdf5
2019-11-23 03:06:12,060 P6485 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/valid.hdf5
2019-11-23 03:06:13,770 P6485 INFO Train samples: total/36672493 - pos/9396350 - neg/27276143 - ratio/25.62%
2019-11-23 03:06:13,908 P6485 INFO Validation samples: total/4584062 - pos/1174544 - neg/3409518 - ratio/25.62%
2019-11-23 03:06:13,908 P6485 INFO Loading train data done.
2019-11-23 03:06:22,200 P6485 INFO **** Start training: 3668 batches/epoch ****
2019-11-23 03:24:25,449 P6485 INFO [Metrics] logloss: 0.445710 - AUC: 0.805459
2019-11-23 03:24:25,525 P6485 INFO Save best model: monitor(max): 0.359750
2019-11-23 03:24:26,434 P6485 INFO ******* 3668/3668 batches finished *******
2019-11-23 03:24:26,675 P6485 INFO [Train] loss: 0.455656
2019-11-23 03:24:26,675 P6485 INFO ************ Epoch=1 end ************
2019-11-23 03:42:34,083 P6485 INFO [Metrics] logloss: 0.444641 - AUC: 0.806734
2019-11-23 03:42:34,152 P6485 INFO Save best model: monitor(max): 0.362094
2019-11-23 03:42:35,172 P6485 INFO ******* 3668/3668 batches finished *******
2019-11-23 03:42:35,424 P6485 INFO [Train] loss: 0.440574
2019-11-23 03:42:35,424 P6485 INFO ************ Epoch=2 end ************
2019-11-23 04:00:45,407 P6485 INFO [Metrics] logloss: 0.448930 - AUC: 0.803635
2019-11-23 04:00:45,509 P6485 INFO Monitor(max) STOP: 0.354705 !!!
2019-11-23 04:00:45,509 P6485 INFO Reduce learning rate on plateau: 0.000100
2019-11-23 04:00:45,509 P6485 INFO ******* 3668/3668 batches finished *******
2019-11-23 04:00:45,756 P6485 INFO [Train] loss: 0.429898
2019-11-23 04:00:45,757 P6485 INFO ************ Epoch=3 end ************
2019-11-23 04:18:54,346 P6485 INFO [Metrics] logloss: 0.498651 - AUC: 0.784049
2019-11-23 04:18:54,432 P6485 INFO Monitor(max) STOP: 0.285398 !!!
2019-11-23 04:18:54,432 P6485 INFO Reduce learning rate on plateau: 0.000010
2019-11-23 04:18:54,432 P6485 INFO ******* 3668/3668 batches finished *******
2019-11-23 04:18:54,679 P6485 INFO [Train] loss: 0.383561
2019-11-23 04:18:54,679 P6485 INFO ************ Epoch=4 end ************
2019-11-23 04:37:00,792 P6485 INFO [Metrics] logloss: 0.522011 - AUC: 0.778305
2019-11-23 04:37:00,877 P6485 INFO Monitor(max) STOP: 0.256294 !!!
2019-11-23 04:37:00,877 P6485 INFO Reduce learning rate on plateau: 0.000001
2019-11-23 04:37:00,877 P6485 INFO Early stopping at epoch=5
2019-11-23 04:37:00,877 P6485 INFO ******* 3668/3668 batches finished *******
2019-11-23 04:37:01,012 P6485 INFO [Train] loss: 0.369109
2019-11-23 04:37:01,013 P6485 INFO Training finished.
2019-11-23 04:37:01,546 P6485 INFO ****** Train/validation evaluation ******
2019-11-23 04:44:47,792 P6485 INFO [Metrics] logloss: 0.413163 - AUC: 0.840492
2019-11-23 04:45:45,563 P6485 INFO [Metrics] logloss: 0.444641 - AUC: 0.806734
2019-11-23 04:45:45,995 P6485 INFO ******** Test evaluation ********
2019-11-23 04:45:45,995 P6485 INFO Loading data...
2019-11-23 04:45:45,995 P6485 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/test.hdf5
2019-11-23 04:45:46,748 P6485 INFO Test samples: total/4584062 - pos/1174544 - neg/3409518 - ratio/25.62%
2019-11-23 04:45:46,748 P6485 INFO Loading test data done.
2019-11-23 04:46:44,371 P6485 INFO [Metrics] logloss: 0.444309 - AUC: 0.807179

```