## xDeepFM_movielenslatest_x1

A hands-on guide to run the xDeepFM model on the Movielenslatest_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Movielenslatest/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_movielenslatest_x1_tuner_config_02](./xDeepFM_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_movielenslatest_x1
    nohup python run_expid.py --config ./xDeepFM_movielenslatest_x1_tuner_config_02 --expid xDeepFM_movielenslatest_x1_018_57ed221b --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.969674 | 0.240939  |
| 2 | 0.966895 | 0.224186  |
| 3 | 0.968227 | 0.228974  |
| 4 | 0.967941 | 0.228906  |
| 5 | 0.967755 | 0.227535  |
| | | | 
| Avg | 0.968098 | 0.230108 |
| Std | &#177;0.00090443 | &#177;0.00568738 |


### Logs
```python
2022-01-19 20:53:11,319 P15419 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "cin_layer_units": "[64, 64, 64]",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_movielenslatest_x1_018_57ed221b",
    "model_root": "./Movielens/xDeepFM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-19 20:53:11,320 P15419 INFO Set up feature encoder...
2022-01-19 20:53:11,320 P15419 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-19 20:53:11,320 P15419 INFO Loading data...
2022-01-19 20:53:11,322 P15419 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-19 20:53:11,349 P15419 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-19 20:53:11,357 P15419 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-19 20:53:11,357 P15419 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-19 20:53:11,357 P15419 INFO Loading train data done.
2022-01-19 20:53:15,405 P15419 INFO Total number of parameters: 1351767.
2022-01-19 20:53:15,406 P15419 INFO Start training: 343 batches/epoch
2022-01-19 20:53:15,406 P15419 INFO ************ Epoch=1 start ************
2022-01-19 20:53:53,862 P15419 INFO [Metrics] AUC: 0.928424 - logloss: 0.304126
2022-01-19 20:53:53,863 P15419 INFO Save best model: monitor(max): 0.928424
2022-01-19 20:53:53,872 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:53:53,915 P15419 INFO Train loss: 0.380009
2022-01-19 20:53:53,915 P15419 INFO ************ Epoch=1 end ************
2022-01-19 20:54:40,242 P15419 INFO [Metrics] AUC: 0.933998 - logloss: 0.293785
2022-01-19 20:54:40,243 P15419 INFO Save best model: monitor(max): 0.933998
2022-01-19 20:54:40,253 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:54:40,295 P15419 INFO Train loss: 0.347214
2022-01-19 20:54:40,295 P15419 INFO ************ Epoch=2 end ************
2022-01-19 20:55:26,976 P15419 INFO [Metrics] AUC: 0.942034 - logloss: 0.273673
2022-01-19 20:55:26,977 P15419 INFO Save best model: monitor(max): 0.942034
2022-01-19 20:55:26,985 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:55:27,024 P15419 INFO Train loss: 0.342338
2022-01-19 20:55:27,025 P15419 INFO ************ Epoch=3 end ************
2022-01-19 20:56:13,756 P15419 INFO [Metrics] AUC: 0.945532 - logloss: 0.266782
2022-01-19 20:56:13,757 P15419 INFO Save best model: monitor(max): 0.945532
2022-01-19 20:56:13,767 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:56:13,819 P15419 INFO Train loss: 0.339949
2022-01-19 20:56:13,819 P15419 INFO ************ Epoch=4 end ************
2022-01-19 20:57:00,478 P15419 INFO [Metrics] AUC: 0.948151 - logloss: 0.258065
2022-01-19 20:57:00,479 P15419 INFO Save best model: monitor(max): 0.948151
2022-01-19 20:57:00,487 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:57:00,524 P15419 INFO Train loss: 0.339723
2022-01-19 20:57:00,524 P15419 INFO ************ Epoch=5 end ************
2022-01-19 20:57:43,968 P15419 INFO [Metrics] AUC: 0.951575 - logloss: 0.248919
2022-01-19 20:57:43,968 P15419 INFO Save best model: monitor(max): 0.951575
2022-01-19 20:57:43,979 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:57:44,028 P15419 INFO Train loss: 0.341247
2022-01-19 20:57:44,028 P15419 INFO ************ Epoch=6 end ************
2022-01-19 20:58:32,549 P15419 INFO [Metrics] AUC: 0.954029 - logloss: 0.242299
2022-01-19 20:58:32,550 P15419 INFO Save best model: monitor(max): 0.954029
2022-01-19 20:58:32,558 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:58:32,613 P15419 INFO Train loss: 0.341127
2022-01-19 20:58:32,614 P15419 INFO ************ Epoch=7 end ************
2022-01-19 20:59:21,430 P15419 INFO [Metrics] AUC: 0.955701 - logloss: 0.237546
2022-01-19 20:59:21,431 P15419 INFO Save best model: monitor(max): 0.955701
2022-01-19 20:59:21,441 P15419 INFO --- 343/343 batches finished ---
2022-01-19 20:59:21,493 P15419 INFO Train loss: 0.342740
2022-01-19 20:59:21,493 P15419 INFO ************ Epoch=8 end ************
2022-01-19 21:00:10,202 P15419 INFO [Metrics] AUC: 0.956981 - logloss: 0.234017
2022-01-19 21:00:10,203 P15419 INFO Save best model: monitor(max): 0.956981
2022-01-19 21:00:10,213 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:00:10,275 P15419 INFO Train loss: 0.342984
2022-01-19 21:00:10,275 P15419 INFO ************ Epoch=9 end ************
2022-01-19 21:00:59,214 P15419 INFO [Metrics] AUC: 0.958016 - logloss: 0.231414
2022-01-19 21:00:59,215 P15419 INFO Save best model: monitor(max): 0.958016
2022-01-19 21:00:59,223 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:00:59,268 P15419 INFO Train loss: 0.340889
2022-01-19 21:00:59,268 P15419 INFO ************ Epoch=10 end ************
2022-01-19 21:01:48,233 P15419 INFO [Metrics] AUC: 0.958998 - logloss: 0.227302
2022-01-19 21:01:48,234 P15419 INFO Save best model: monitor(max): 0.958998
2022-01-19 21:01:48,242 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:01:48,292 P15419 INFO Train loss: 0.338508
2022-01-19 21:01:48,292 P15419 INFO ************ Epoch=11 end ************
2022-01-19 21:02:36,999 P15419 INFO [Metrics] AUC: 0.959436 - logloss: 0.226215
2022-01-19 21:02:37,000 P15419 INFO Save best model: monitor(max): 0.959436
2022-01-19 21:02:37,011 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:02:37,057 P15419 INFO Train loss: 0.337039
2022-01-19 21:02:37,057 P15419 INFO ************ Epoch=12 end ************
2022-01-19 21:03:26,124 P15419 INFO [Metrics] AUC: 0.960461 - logloss: 0.223721
2022-01-19 21:03:26,125 P15419 INFO Save best model: monitor(max): 0.960461
2022-01-19 21:03:26,136 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:03:26,175 P15419 INFO Train loss: 0.334841
2022-01-19 21:03:26,175 P15419 INFO ************ Epoch=13 end ************
2022-01-19 21:04:15,255 P15419 INFO [Metrics] AUC: 0.960749 - logloss: 0.223829
2022-01-19 21:04:15,256 P15419 INFO Save best model: monitor(max): 0.960749
2022-01-19 21:04:15,264 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:04:15,314 P15419 INFO Train loss: 0.334047
2022-01-19 21:04:15,314 P15419 INFO ************ Epoch=14 end ************
2022-01-19 21:05:04,504 P15419 INFO [Metrics] AUC: 0.960977 - logloss: 0.223660
2022-01-19 21:05:04,505 P15419 INFO Save best model: monitor(max): 0.960977
2022-01-19 21:05:04,517 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:05:04,557 P15419 INFO Train loss: 0.332129
2022-01-19 21:05:04,557 P15419 INFO ************ Epoch=15 end ************
2022-01-19 21:05:47,024 P15419 INFO [Metrics] AUC: 0.961497 - logloss: 0.220922
2022-01-19 21:05:47,026 P15419 INFO Save best model: monitor(max): 0.961497
2022-01-19 21:05:47,038 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:05:47,080 P15419 INFO Train loss: 0.330475
2022-01-19 21:05:47,081 P15419 INFO ************ Epoch=16 end ************
2022-01-19 21:06:33,160 P15419 INFO [Metrics] AUC: 0.961355 - logloss: 0.221015
2022-01-19 21:06:33,160 P15419 INFO Monitor(max) STOP: 0.961355 !
2022-01-19 21:06:33,160 P15419 INFO Reduce learning rate on plateau: 0.000100
2022-01-19 21:06:33,160 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:06:33,232 P15419 INFO Train loss: 0.330120
2022-01-19 21:06:33,233 P15419 INFO ************ Epoch=17 end ************
2022-01-19 21:07:19,106 P15419 INFO [Metrics] AUC: 0.968524 - logloss: 0.210254
2022-01-19 21:07:19,107 P15419 INFO Save best model: monitor(max): 0.968524
2022-01-19 21:07:19,115 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:07:19,153 P15419 INFO Train loss: 0.242449
2022-01-19 21:07:19,154 P15419 INFO ************ Epoch=18 end ************
2022-01-19 21:08:05,261 P15419 INFO [Metrics] AUC: 0.970061 - logloss: 0.220010
2022-01-19 21:08:05,262 P15419 INFO Save best model: monitor(max): 0.970061
2022-01-19 21:08:05,272 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:08:05,318 P15419 INFO Train loss: 0.169935
2022-01-19 21:08:05,318 P15419 INFO ************ Epoch=19 end ************
2022-01-19 21:08:51,496 P15419 INFO [Metrics] AUC: 0.970148 - logloss: 0.238138
2022-01-19 21:08:51,496 P15419 INFO Save best model: monitor(max): 0.970148
2022-01-19 21:08:51,507 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:08:51,554 P15419 INFO Train loss: 0.130883
2022-01-19 21:08:51,554 P15419 INFO ************ Epoch=20 end ************
2022-01-19 21:09:37,534 P15419 INFO [Metrics] AUC: 0.969609 - logloss: 0.256694
2022-01-19 21:09:37,535 P15419 INFO Monitor(max) STOP: 0.969609 !
2022-01-19 21:09:37,535 P15419 INFO Reduce learning rate on plateau: 0.000010
2022-01-19 21:09:37,535 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:09:37,572 P15419 INFO Train loss: 0.109121
2022-01-19 21:09:37,572 P15419 INFO ************ Epoch=21 end ************
2022-01-19 21:10:23,607 P15419 INFO [Metrics] AUC: 0.969557 - logloss: 0.273859
2022-01-19 21:10:23,607 P15419 INFO Monitor(max) STOP: 0.969557 !
2022-01-19 21:10:23,608 P15419 INFO Reduce learning rate on plateau: 0.000001
2022-01-19 21:10:23,608 P15419 INFO Early stopping at epoch=22
2022-01-19 21:10:23,608 P15419 INFO --- 343/343 batches finished ---
2022-01-19 21:10:23,652 P15419 INFO Train loss: 0.090166
2022-01-19 21:10:23,652 P15419 INFO Training finished.
2022-01-19 21:10:23,652 P15419 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/xDeepFM_movielenslatest_x1/movielenslatest_x1_cd32d937/xDeepFM_movielenslatest_x1_018_57ed221b.model
2022-01-19 21:10:23,681 P15419 INFO ****** Validation evaluation ******
2022-01-19 21:10:25,808 P15419 INFO [Metrics] AUC: 0.970148 - logloss: 0.238138
2022-01-19 21:10:25,845 P15419 INFO ******** Test evaluation ********
2022-01-19 21:10:25,846 P15419 INFO Loading data...
2022-01-19 21:10:25,846 P15419 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-19 21:10:25,851 P15419 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-19 21:10:25,852 P15419 INFO Loading test data done.
2022-01-19 21:10:26,908 P15419 INFO [Metrics] AUC: 0.969674 - logloss: 0.240939

```
