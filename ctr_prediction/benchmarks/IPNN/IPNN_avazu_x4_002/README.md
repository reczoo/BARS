## IPNN_Avazu_x4_002

A notebook to benchmark IPNN on Avazu_x4_002 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default <OOV> token. Note that we found that min_category_count=1 performs the best, which is surprising.

We fix embedding_dim=40 following the existing FGCNN work.
### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Avazu/Avazu_x4/split_avazu_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [IPNN_avazu_x4_tuner_config_01.yaml](./002/IPNN_avazu_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/IPNN_avazu_x4_tuner_config_01.yaml --tag 013 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.368390 - AUC: 0.798853

```


### Logs
```python
2019-11-04 13:57:49,962 P12395 INFO {
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_29cf4fdc",
    "dropout_rates": "0",
    "embedding_dim": "40",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[500, 500, 500]",
    "kernel_regularizer": "0",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "PNN",
    "model_dir": "./PNN_avazu/",
    "model_id": "PNN_avazu_x4_013_ffc73f9c",
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
    "workers": "2",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "min_categr_count": "1",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "version": "pytorch",
    "device": "0"
}
2019-11-04 13:57:49,963 P12395 INFO Set up feature encoder...
2019-11-04 13:57:56,414 P12395 INFO Load feature encoder cache from ./PNN_avazu/avazu_x4_001_29cf4fdc/feature_encoder.pkl
2019-11-04 13:57:56,415 P12395 INFO Loading data...
2019-11-04 13:57:56,420 P12395 INFO Loading data from ./PNN_avazu/avazu_x4_001_29cf4fdc/train.hdf5
2019-11-04 13:57:58,831 P12395 INFO Loading data from ./PNN_avazu/avazu_x4_001_29cf4fdc/valid.hdf5
2019-11-04 13:58:00,077 P12395 INFO Train samples: total/32343172 - pos/5492052 - neg/26851120 - ratio/16.98%
2019-11-04 13:58:00,194 P12395 INFO Validation samples: total/4042897 - pos/686507 - neg/3356390 - ratio/16.98%
2019-11-04 13:58:00,194 P12395 INFO Loading train data done.
2019-11-04 13:58:11,095 P12395 INFO **** Start training: 3235 batches/epoch ****
2019-11-04 14:09:15,317 P12395 INFO [Metrics] logloss: 0.368642 - AUC: 0.798446
2019-11-04 14:09:15,374 P12395 INFO Save best model: monitor(max): 0.429804
2019-11-04 14:09:16,628 P12395 INFO ******* 3235/3235 batches finished *******
2019-11-04 14:09:16,794 P12395 INFO [Train] loss: 0.379404
2019-11-04 14:09:16,794 P12395 INFO ************ Epoch=1 end ************
2019-11-04 14:20:22,662 P12395 INFO [Metrics] logloss: 0.408572 - AUC: 0.771366
2019-11-04 14:20:22,730 P12395 INFO Monitor(max) STOP: 0.362794 !!!
2019-11-04 14:20:22,730 P12395 INFO Reduce learning rate on plateau: 0.000200
2019-11-04 14:20:22,730 P12395 INFO ******* 3235/3235 batches finished *******
2019-11-04 14:20:22,923 P12395 INFO [Train] loss: 0.283605
2019-11-04 14:20:22,924 P12395 INFO ************ Epoch=2 end ************
2019-11-04 14:31:27,923 P12395 INFO [Metrics] logloss: 0.485962 - AUC: 0.762717
2019-11-04 14:31:28,014 P12395 INFO Monitor(max) STOP: 0.276754 !!!
2019-11-04 14:31:28,015 P12395 INFO Reduce learning rate on plateau: 0.000040
2019-11-04 14:31:28,015 P12395 INFO ******* 3235/3235 batches finished *******
2019-11-04 14:31:28,206 P12395 INFO [Train] loss: 0.233618
2019-11-04 14:31:28,206 P12395 INFO ************ Epoch=3 end ************
2019-11-04 14:42:35,181 P12395 INFO [Metrics] logloss: 0.566576 - AUC: 0.754176
2019-11-04 14:42:35,277 P12395 INFO Monitor(max) STOP: 0.187600 !!!
2019-11-04 14:42:35,277 P12395 INFO Reduce learning rate on plateau: 0.000008
2019-11-04 14:42:35,278 P12395 INFO Early stopping at epoch=4
2019-11-04 14:42:35,278 P12395 INFO ******* 3235/3235 batches finished *******
2019-11-04 14:42:35,384 P12395 INFO [Train] loss: 0.212558
2019-11-04 14:42:35,384 P12395 INFO Training finished.
2019-11-04 14:42:36,163 P12395 INFO ****** Train/validation evaluation ******
2019-11-04 14:47:43,231 P12395 INFO [Metrics] logloss: 0.317269 - AUC: 0.873208
2019-11-04 14:48:19,937 P12395 INFO [Metrics] logloss: 0.368642 - AUC: 0.798446
2019-11-04 14:48:20,124 P12395 INFO ******** Test evaluation ********
2019-11-04 14:48:20,124 P12395 INFO Loading data...
2019-11-04 14:48:20,124 P12395 INFO Loading data from ./PNN_avazu/avazu_x4_001_29cf4fdc/test.hdf5
2019-11-04 14:48:20,763 P12395 INFO Test samples: total/4042898 - pos/686507 - neg/3356391 - ratio/16.98%
2019-11-04 14:48:20,763 P12395 INFO Loading test data done.
2019-11-04 14:48:57,449 P12395 INFO [Metrics] logloss: 0.368390 - AUC: 0.798853

```
