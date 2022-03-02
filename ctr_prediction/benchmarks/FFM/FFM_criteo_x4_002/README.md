## FFM_criteo_x4_002

A hands-on guide to run the FFM model on the Criteo_x4_002 dataset.

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
Dataset ID: [Criteo_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [FFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_criteo_x4_tuner_config_01](./FFM_criteo_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_criteo_x4_002
    nohup python run_expid.py --config ./FFM_criteo_x4_tuner_config_01 --expid FFM_criteo_x4_023_0b654e65 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.440905 | 0.811085  |


### Logs
```python
2019-12-07 13:52:08,761 P21039 INFO {
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_0d63c1a1",
    "embedding_dim": "3",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FFM",
    "model_dir": "./Criteo/",
    "model_id": "FFM_criteo_x4_023_833b7b8a",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "3",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
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
    "device": "1"
}
2019-12-07 13:52:08,761 P21039 INFO Set up feature encoder...
2019-12-07 13:52:23,823 P21039 INFO Load feature encoder cache from ./Criteo/criteo_x4_001_0d63c1a1/feature_encoder.pkl
2019-12-07 13:52:23,823 P21039 INFO Loading data...
2019-12-07 13:52:23,825 P21039 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/train.hdf5
2019-12-07 13:52:28,248 P21039 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/valid.hdf5
2019-12-07 13:52:29,982 P21039 INFO Train samples: total/36672493 - pos/9396350 - neg/27276143 - ratio/25.62%
2019-12-07 13:52:30,118 P21039 INFO Validation samples: total/4584062 - pos/1174544 - neg/3409518 - ratio/25.62%
2019-12-07 13:52:30,118 P21039 INFO Loading train data done.
2019-12-07 13:52:48,665 P21039 INFO **** Start training: 3668 batches/epoch ****
2019-12-07 17:51:16,368 P21039 INFO [Metrics] logloss: 0.443444 - AUC: 0.808130
2019-12-07 17:51:16,433 P21039 INFO Save best model: monitor(max): 0.364686
2019-12-07 17:51:18,487 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-07 17:51:20,297 P21039 INFO [Train] loss: 0.459265
2019-12-07 17:51:20,297 P21039 INFO ************ Epoch=1 end ************
2019-12-07 21:50:23,603 P21039 INFO [Metrics] logloss: 0.441863 - AUC: 0.809884
2019-12-07 21:50:23,705 P21039 INFO Save best model: monitor(max): 0.368022
2019-12-07 21:50:27,976 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-07 21:50:29,844 P21039 INFO [Train] loss: 0.452403
2019-12-07 21:50:29,845 P21039 INFO ************ Epoch=2 end ************
2019-12-08 01:49:44,152 P21039 INFO [Metrics] logloss: 0.441639 - AUC: 0.810156
2019-12-08 01:49:44,239 P21039 INFO Save best model: monitor(max): 0.368517
2019-12-08 01:49:48,470 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-08 01:49:50,329 P21039 INFO [Train] loss: 0.450995
2019-12-08 01:49:50,329 P21039 INFO ************ Epoch=3 end ************
2019-12-08 05:47:39,349 P21039 INFO [Metrics] logloss: 0.442013 - AUC: 0.809776
2019-12-08 05:47:39,416 P21039 INFO Monitor(max) STOP: 0.367763 !!!
2019-12-08 05:47:39,416 P21039 INFO Reduce learning rate on plateau: 0.000100
2019-12-08 05:47:39,416 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-08 05:47:41,220 P21039 INFO [Train] loss: 0.450282
2019-12-08 05:47:41,220 P21039 INFO ************ Epoch=4 end ************
2019-12-08 09:45:21,713 P21039 INFO [Metrics] logloss: 0.441366 - AUC: 0.810563
2019-12-08 09:45:21,812 P21039 INFO Save best model: monitor(max): 0.369197
2019-12-08 09:45:26,075 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-08 09:45:27,925 P21039 INFO [Train] loss: 0.430549
2019-12-08 09:45:27,925 P21039 INFO ************ Epoch=5 end ************
2019-12-08 13:44:35,769 P21039 INFO [Metrics] logloss: 0.442077 - AUC: 0.809977
2019-12-08 13:44:35,832 P21039 INFO Monitor(max) STOP: 0.367900 !!!
2019-12-08 13:44:35,832 P21039 INFO Reduce learning rate on plateau: 0.000010
2019-12-08 13:44:35,832 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-08 13:44:37,719 P21039 INFO [Train] loss: 0.425769
2019-12-08 13:44:37,720 P21039 INFO ************ Epoch=6 end ************
2019-12-08 17:47:02,988 P21039 INFO [Metrics] logloss: 0.442161 - AUC: 0.809891
2019-12-08 17:47:03,054 P21039 INFO Monitor(max) STOP: 0.367730 !!!
2019-12-08 17:47:03,055 P21039 INFO Reduce learning rate on plateau: 0.000001
2019-12-08 17:47:03,055 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-08 17:47:04,945 P21039 INFO [Train] loss: 0.420307
2019-12-08 17:47:04,945 P21039 INFO ************ Epoch=7 end ************
2019-12-08 21:48:21,761 P21039 INFO [Metrics] logloss: 0.442171 - AUC: 0.809879
2019-12-08 21:48:21,915 P21039 INFO Monitor(max) STOP: 0.367708 !!!
2019-12-08 21:48:21,916 P21039 INFO Reduce learning rate on plateau: 0.000001
2019-12-08 21:48:21,916 P21039 INFO Early stopping at epoch=8
2019-12-08 21:48:21,916 P21039 INFO ******* 3668/3668 batches finished *******
2019-12-08 21:48:22,049 P21039 INFO [Train] loss: 0.419498
2019-12-08 21:48:22,050 P21039 INFO Training finished.
2019-12-08 21:48:24,084 P21039 INFO ****** Train/validation evaluation ******
2019-12-08 22:12:56,369 P21039 INFO [Metrics] logloss: 0.411515 - AUC: 0.841924
2019-12-08 22:16:15,700 P21039 INFO [Metrics] logloss: 0.441366 - AUC: 0.810563
2019-12-08 22:16:16,013 P21039 INFO ******** Test evaluation ********
2019-12-08 22:16:16,013 P21039 INFO Loading data...
2019-12-08 22:16:16,014 P21039 INFO Loading data from ./Criteo/criteo_x4_001_0d63c1a1/test.hdf5
2019-12-08 22:16:16,791 P21039 INFO Test samples: total/4584062 - pos/1174544 - neg/3409518 - ratio/25.62%
2019-12-08 22:16:16,792 P21039 INFO Loading test data done.
2019-12-08 22:18:48,948 P21039 INFO [Metrics] logloss: 0.440905 - AUC: 0.811085

```
