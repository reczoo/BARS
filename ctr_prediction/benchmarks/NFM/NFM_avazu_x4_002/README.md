## NFM_avazu_x4_002

A hands-on guide to run the NFM model on the Avazu_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [NFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/NFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [NFM_avazu_x4_tuner_config_01](./NFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd NFM_avazu_x4_002
    nohup python run_expid.py --config ./NFM_avazu_x4_tuner_config_01 --expid NFM_avazu_x4_015_2dd318c6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371461 | 0.793991  |


### Logs
```python
2019-11-08 09:54:38,071 P51739 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_29cf4fdc",
    "dropout_rates": "0",
    "embedding_dim": "40",
    "embedding_regularizer": "l2(1.e-9)",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "kernel_regularizer": "0",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "NFM",
    "model_dir": "./DeepFM_avazu/",
    "model_id": "NFM_avazu_x4_015_753dc097",
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
2019-11-08 09:54:38,072 P51739 INFO Set up feature encoder...
2019-11-08 09:54:44,424 P51739 INFO Load feature encoder cache from ./DeepFM_avazu/avazu_x4_001_29cf4fdc/feature_encoder.pkl
2019-11-08 09:54:44,424 P51739 INFO Loading data...
2019-11-08 09:54:44,427 P51739 INFO Loading data from ./DeepFM_avazu/avazu_x4_001_29cf4fdc/train.hdf5
2019-11-08 09:54:46,959 P51739 INFO Loading data from ./DeepFM_avazu/avazu_x4_001_29cf4fdc/valid.hdf5
2019-11-08 09:54:48,194 P51739 INFO Train samples: total/32343172 - pos/5492052 - neg/26851120 - ratio/16.98%
2019-11-08 09:54:48,309 P51739 INFO Validation samples: total/4042897 - pos/686507 - neg/3356390 - ratio/16.98%
2019-11-08 09:54:48,309 P51739 INFO Loading train data done.
2019-11-08 09:54:59,308 P51739 INFO **** Start training: 3235 batches/epoch ****
2019-11-08 10:10:57,104 P51739 INFO [Metrics] logloss: 0.372203 - AUC: 0.793141
2019-11-08 10:10:57,191 P51739 INFO Save best model: monitor(max): 0.420938
2019-11-08 10:10:58,824 P51739 INFO ******* 3235/3235 batches finished *******
2019-11-08 10:10:59,047 P51739 INFO [Train] loss: 0.381786
2019-11-08 10:10:59,048 P51739 INFO ************ Epoch=1 end ************
2019-11-08 10:26:54,944 P51739 INFO [Metrics] logloss: 0.404908 - AUC: 0.775331
2019-11-08 10:26:55,001 P51739 INFO Monitor(max) STOP: 0.370423 !!!
2019-11-08 10:26:55,001 P51739 INFO Reduce learning rate on plateau: 0.000100
2019-11-08 10:26:55,001 P51739 INFO ******* 3235/3235 batches finished *******
2019-11-08 10:26:55,231 P51739 INFO [Train] loss: 0.306827
2019-11-08 10:26:55,232 P51739 INFO ************ Epoch=2 end ************
2019-11-08 10:42:51,398 P51739 INFO [Metrics] logloss: 0.475624 - AUC: 0.760513
2019-11-08 10:42:51,457 P51739 INFO Monitor(max) STOP: 0.284890 !!!
2019-11-08 10:42:51,457 P51739 INFO Reduce learning rate on plateau: 0.000010
2019-11-08 10:42:51,457 P51739 INFO ******* 3235/3235 batches finished *******
2019-11-08 10:42:51,687 P51739 INFO [Train] loss: 0.241524
2019-11-08 10:42:51,687 P51739 INFO ************ Epoch=3 end ************
2019-11-08 10:58:48,303 P51739 INFO [Metrics] logloss: 0.505215 - AUC: 0.756009
2019-11-08 10:58:48,390 P51739 INFO Monitor(max) STOP: 0.250794 !!!
2019-11-08 10:58:48,390 P51739 INFO Reduce learning rate on plateau: 0.000001
2019-11-08 10:58:48,390 P51739 INFO Early stopping at epoch=4
2019-11-08 10:58:48,390 P51739 INFO ******* 3235/3235 batches finished *******
2019-11-08 10:58:48,487 P51739 INFO [Train] loss: 0.229717
2019-11-08 10:58:48,487 P51739 INFO Training finished.
2019-11-08 10:58:49,233 P51739 INFO ****** Train/validation evaluation ******
2019-11-08 11:03:53,042 P51739 INFO [Metrics] logloss: 0.330057 - AUC: 0.858290
2019-11-08 11:04:30,946 P51739 INFO [Metrics] logloss: 0.372203 - AUC: 0.793141
2019-11-08 11:04:31,179 P51739 INFO ******** Test evaluation ********
2019-11-08 11:04:31,179 P51739 INFO Loading data...
2019-11-08 11:04:31,180 P51739 INFO Loading data from ./DeepFM_avazu/avazu_x4_001_29cf4fdc/test.hdf5
2019-11-08 11:04:31,808 P51739 INFO Test samples: total/4042898 - pos/686507 - neg/3356391 - ratio/16.98%
2019-11-08 11:04:31,808 P51739 INFO Loading test data done.
2019-11-08 11:05:09,241 P51739 INFO [Metrics] logloss: 0.372079 - AUC: 0.793405

```
