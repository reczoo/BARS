## FFM_avazu_x4_002

A hands-on guide to run the FFM model on the Avazu_x4_002 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_avazu_x4_tuner_config_01](./FFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_avazu_x4_002
    nohup python run_expid.py --config ./FFM_avazu_x4_tuner_config_01 --expid FFM_avazu_x4_004_0fd87979 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371135 | 0.794841  |


### Logs
```python
2019-12-03 07:45:24,356 P56327 INFO {
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_29cf4fdc",
    "embedding_dim": "4",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FFM",
    "model_dir": "./Avazu/",
    "model_id": "FFM_avazu_x4_004_dd7b4016",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "regularizer": "0",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "min_categr_count": "1",
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "version": "pytorch",
    "device": "0"
}
2019-12-03 07:45:24,357 P56327 INFO Set up feature encoder...
2019-12-03 07:45:30,639 P56327 INFO Load feature encoder cache from ./Avazu/avazu_x4_001_29cf4fdc/feature_encoder.pkl
2019-12-03 07:45:30,639 P56327 INFO Loading data...
2019-12-03 07:45:30,688 P56327 INFO Loading data from ./Avazu/avazu_x4_001_29cf4fdc/train.hdf5
2019-12-03 07:45:33,314 P56327 INFO Loading data from ./Avazu/avazu_x4_001_29cf4fdc/valid.hdf5
2019-12-03 07:45:34,549 P56327 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2019-12-03 07:45:34,662 P56327 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2019-12-03 07:45:34,662 P56327 INFO Loading train data done.
2019-12-03 07:45:55,724 P56327 INFO **** Start training: 3235 batches/epoch ****
2019-12-03 09:40:03,878 P56327 INFO [Metrics] logloss: 0.371138 - AUC: 0.794795
2019-12-03 09:40:03,946 P56327 INFO Save best model: monitor(max): 0.423657
2019-12-03 09:40:06,844 P56327 INFO ---- 3235/3235 batches finished ----
2019-12-03 09:40:07,774 P56327 INFO [Train] loss: 0.382683
2019-12-03 09:40:07,774 P56327 INFO ************ Epoch=1 end ************
2019-12-03 11:34:13,288 P56327 INFO [Metrics] logloss: 0.389847 - AUC: 0.784462
2019-12-03 11:34:13,357 P56327 INFO Monitor(max) STOP: 0.394615 !!!
2019-12-03 11:34:13,357 P56327 INFO Reduce learning rate on plateau: 0.000100
2019-12-03 11:34:13,357 P56327 INFO ---- 3235/3235 batches finished ----
2019-12-03 11:34:14,309 P56327 INFO [Train] loss: 0.308642
2019-12-03 11:34:14,309 P56327 INFO ************ Epoch=2 end ************
2019-12-03 13:28:18,632 P56327 INFO [Metrics] logloss: 0.422587 - AUC: 0.772673
2019-12-03 13:28:18,692 P56327 INFO Monitor(max) STOP: 0.350086 !!!
2019-12-03 13:28:18,693 P56327 INFO Reduce learning rate on plateau: 0.000010
2019-12-03 13:28:18,693 P56327 INFO ---- 3235/3235 batches finished ----
2019-12-03 13:28:19,639 P56327 INFO [Train] loss: 0.243154
2019-12-03 13:28:19,639 P56327 INFO ************ Epoch=3 end ************
2019-12-03 15:22:27,954 P56327 INFO [Metrics] logloss: 0.425619 - AUC: 0.771796
2019-12-03 15:22:28,021 P56327 INFO Monitor(max) STOP: 0.346177 !!!
2019-12-03 15:22:28,021 P56327 INFO Reduce learning rate on plateau: 0.000001
2019-12-03 15:22:28,021 P56327 INFO Early stopping at epoch=4
2019-12-03 15:22:28,021 P56327 INFO ---- 3235/3235 batches finished ----
2019-12-03 15:22:28,123 P56327 INFO [Train] loss: 0.234798
2019-12-03 15:22:28,123 P56327 INFO Training finished.
2019-12-03 15:22:30,085 P56327 INFO ****** Train/validation evaluation ******
2019-12-03 15:27:42,614 P56327 INFO [Metrics] logloss: 0.322468 - AUC: 0.867630
2019-12-03 15:28:22,820 P56327 INFO [Metrics] logloss: 0.371138 - AUC: 0.794795
2019-12-03 15:28:23,132 P56327 INFO ******** Test evaluation ********
2019-12-03 15:28:23,132 P56327 INFO Loading data...
2019-12-03 15:28:23,132 P56327 INFO Loading data from ./Avazu/avazu_x4_001_29cf4fdc/test.hdf5
2019-12-03 15:28:23,795 P56327 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2019-12-03 15:28:23,796 P56327 INFO Loading test data done.
2019-12-03 15:29:02,997 P56327 INFO [Metrics] logloss: 0.371135 - AUC: 0.794841

```
