## WideDeep_avazu_x4_002

A hands-on guide to run the WideDeep model on the Avazu_x4_002 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [WideDeep](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_avazu_x4_tuner_config_01](./WideDeep_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_avazu_x4_002
    nohup python run_expid.py --config ./WideDeep_avazu_x4_tuner_config_01 --expid WideDeep_avazu_x4_043_23ffe850 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.370343 | 0.795750  |


### Logs
```python
2019-11-20 23:10:01,303 P12606 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_29cf4fdc",
    "dropout_rates": "0",
    "embedding_dim": "40",
    "embedding_regularizer": "0",
    "epochs": "3",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[700, 700, 700, 700, 700]",
    "kernel_regularizer": "0",
    "layer_norm": "False",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "WideDeep",
    "model_dir": "./WideDeep_avazu/",
    "model_id": "WideDeep_avazu_x4_043_b17389ef",
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
    "device": "1"
}
2019-11-20 23:10:01,304 P12606 INFO Set up feature encoder...
2019-11-20 23:10:09,489 P12606 INFO Load feature encoder cache from ./WideDeep_avazu/avazu_x4_001_29cf4fdc/feature_encoder.pkl
2019-11-20 23:10:09,490 P12606 INFO Loading data...
2019-11-20 23:10:09,551 P12606 INFO Loading data from ./WideDeep_avazu/avazu_x4_001_29cf4fdc/train.hdf5
2019-11-20 23:10:12,443 P12606 INFO Loading data from ./WideDeep_avazu/avazu_x4_001_29cf4fdc/valid.hdf5
2019-11-20 23:10:14,223 P12606 INFO Train samples: total/32343172 - pos/5492052 - neg/26851120 - ratio/16.98%
2019-11-20 23:10:14,395 P12606 INFO Validation samples: total/4042897 - pos/686507 - neg/3356390 - ratio/16.98%
2019-11-20 23:10:14,395 P12606 INFO Loading train data done.
2019-11-20 23:10:25,639 P12606 INFO **** Start training: 3235 batches/epoch ****
2019-11-20 23:24:35,197 P12606 INFO [Metrics] logloss: 0.370547 - AUC: 0.795432
2019-11-20 23:24:35,279 P12606 INFO Save best model: monitor(max): 0.424886
2019-11-20 23:24:36,908 P12606 INFO ******* 3235/3235 batches finished *******
2019-11-20 23:24:37,086 P12606 INFO [Train] loss: 0.380004
2019-11-20 23:24:37,086 P12606 INFO ************ Epoch=1 end ************
2019-11-20 23:38:45,254 P12606 INFO [Metrics] logloss: 0.430659 - AUC: 0.756795
2019-11-20 23:38:45,316 P12606 INFO Monitor(max) STOP: 0.326136 !!!
2019-11-20 23:38:45,316 P12606 INFO Reduce learning rate on plateau: 0.000100
2019-11-20 23:38:45,316 P12606 INFO ******* 3235/3235 batches finished *******
2019-11-20 23:38:45,502 P12606 INFO [Train] loss: 0.288817
2019-11-20 23:38:45,502 P12606 INFO ************ Epoch=2 end ************
2019-11-20 23:52:53,095 P12606 INFO [Metrics] logloss: 0.500465 - AUC: 0.761818
2019-11-20 23:52:53,153 P12606 INFO Monitor(max) STOP: 0.261353 !!!
2019-11-20 23:52:53,153 P12606 INFO Reduce learning rate on plateau: 0.000010
2019-11-20 23:52:53,153 P12606 INFO ******* 3235/3235 batches finished *******
2019-11-20 23:52:53,339 P12606 INFO [Train] loss: 0.248536
2019-11-20 23:52:53,339 P12606 INFO ************ Epoch=3 end ************
2019-11-20 23:52:53,339 P12606 INFO Training finished.
2019-11-20 23:52:54,125 P12606 INFO ****** Train/validation evaluation ******
2019-11-20 23:57:51,477 P12606 INFO [Metrics] logloss: 0.323588 - AUC: 0.866457
2019-11-20 23:58:28,700 P12606 INFO [Metrics] logloss: 0.370547 - AUC: 0.795432
2019-11-20 23:58:28,890 P12606 INFO ******** Test evaluation ********
2019-11-20 23:58:28,891 P12606 INFO Loading data...
2019-11-20 23:58:28,891 P12606 INFO Loading data from ./WideDeep_avazu/avazu_x4_001_29cf4fdc/test.hdf5
2019-11-20 23:58:29,433 P12606 INFO Test samples: total/4042898 - pos/686507 - neg/3356391 - ratio/16.98%
2019-11-20 23:58:29,433 P12606 INFO Loading test data done.
2019-11-20 23:59:05,791 P12606 INFO [Metrics] logloss: 0.370343 - AUC: 0.795750

```
