## CrossNet_movielenslatest_x1

A hands-on guide to run the DCN model on the Movielenslatest_x1 dataset.

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
Dataset ID: [Movielenslatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens/README.md#Movielenslatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [DCN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_movielenslatest_x1_tuner_config_01](./CrossNet_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_movielenslatest_x1
    nohup python run_expid.py --config ./CrossNet_movielenslatest_x1_tuner_config_01 --expid DCN_movielenslatest_x1_006_6ae4ad60 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.944032 | 0.279041  |


### Logs
```python
2022-01-20 20:43:40,223 P27112 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "7",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "None",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_movielenslatest_x1_006_6ae4ad60",
    "model_root": "./Movielens/DCN_MovielensLatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
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
2022-01-20 20:43:40,224 P27112 INFO Set up feature encoder...
2022-01-20 20:43:40,224 P27112 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-20 20:43:40,224 P27112 INFO Loading data...
2022-01-20 20:43:40,226 P27112 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-20 20:43:40,252 P27112 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-20 20:43:40,260 P27112 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-20 20:43:40,260 P27112 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-20 20:43:40,260 P27112 INFO Loading train data done.
2022-01-20 20:43:44,714 P27112 INFO Total number of parameters: 902841.
2022-01-20 20:43:44,715 P27112 INFO Start training: 343 batches/epoch
2022-01-20 20:43:44,715 P27112 INFO ************ Epoch=1 start ************
2022-01-20 20:43:53,155 P27112 INFO [Metrics] AUC: 0.931603 - logloss: 0.296562
2022-01-20 20:43:53,156 P27112 INFO Save best model: monitor(max): 0.931603
2022-01-20 20:43:53,347 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:43:53,394 P27112 INFO Train loss: 0.418492
2022-01-20 20:43:53,395 P27112 INFO ************ Epoch=1 end ************
2022-01-20 20:44:01,710 P27112 INFO [Metrics] AUC: 0.932605 - logloss: 0.295088
2022-01-20 20:44:01,711 P27112 INFO Save best model: monitor(max): 0.932605
2022-01-20 20:44:01,717 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:01,763 P27112 INFO Train loss: 0.326416
2022-01-20 20:44:01,764 P27112 INFO ************ Epoch=2 end ************
2022-01-20 20:44:09,795 P27112 INFO [Metrics] AUC: 0.933744 - logloss: 0.293351
2022-01-20 20:44:09,796 P27112 INFO Save best model: monitor(max): 0.933744
2022-01-20 20:44:09,804 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:09,852 P27112 INFO Train loss: 0.319291
2022-01-20 20:44:09,852 P27112 INFO ************ Epoch=3 end ************
2022-01-20 20:44:18,045 P27112 INFO [Metrics] AUC: 0.935150 - logloss: 0.290797
2022-01-20 20:44:18,047 P27112 INFO Save best model: monitor(max): 0.935150
2022-01-20 20:44:18,061 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:18,101 P27112 INFO Train loss: 0.314822
2022-01-20 20:44:18,101 P27112 INFO ************ Epoch=4 end ************
2022-01-20 20:44:25,730 P27112 INFO [Metrics] AUC: 0.935998 - logloss: 0.289391
2022-01-20 20:44:25,730 P27112 INFO Save best model: monitor(max): 0.935998
2022-01-20 20:44:25,736 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:25,784 P27112 INFO Train loss: 0.310688
2022-01-20 20:44:25,784 P27112 INFO ************ Epoch=5 end ************
2022-01-20 20:44:33,889 P27112 INFO [Metrics] AUC: 0.936743 - logloss: 0.288237
2022-01-20 20:44:33,890 P27112 INFO Save best model: monitor(max): 0.936743
2022-01-20 20:44:33,897 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:33,957 P27112 INFO Train loss: 0.307227
2022-01-20 20:44:33,957 P27112 INFO ************ Epoch=6 end ************
2022-01-20 20:44:41,924 P27112 INFO [Metrics] AUC: 0.937130 - logloss: 0.288088
2022-01-20 20:44:41,925 P27112 INFO Save best model: monitor(max): 0.937130
2022-01-20 20:44:41,933 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:41,988 P27112 INFO Train loss: 0.304345
2022-01-20 20:44:41,988 P27112 INFO ************ Epoch=7 end ************
2022-01-20 20:44:49,694 P27112 INFO [Metrics] AUC: 0.937951 - logloss: 0.286426
2022-01-20 20:44:49,695 P27112 INFO Save best model: monitor(max): 0.937951
2022-01-20 20:44:49,703 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:49,777 P27112 INFO Train loss: 0.302530
2022-01-20 20:44:49,777 P27112 INFO ************ Epoch=8 end ************
2022-01-20 20:44:57,444 P27112 INFO [Metrics] AUC: 0.938643 - logloss: 0.285187
2022-01-20 20:44:57,445 P27112 INFO Save best model: monitor(max): 0.938643
2022-01-20 20:44:57,453 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:44:57,497 P27112 INFO Train loss: 0.301346
2022-01-20 20:44:57,497 P27112 INFO ************ Epoch=9 end ************
2022-01-20 20:45:05,310 P27112 INFO [Metrics] AUC: 0.939102 - logloss: 0.284015
2022-01-20 20:45:05,311 P27112 INFO Save best model: monitor(max): 0.939102
2022-01-20 20:45:05,319 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:45:05,376 P27112 INFO Train loss: 0.301303
2022-01-20 20:45:05,377 P27112 INFO ************ Epoch=10 end ************
2022-01-20 20:45:13,344 P27112 INFO [Metrics] AUC: 0.939846 - logloss: 0.282589
2022-01-20 20:45:13,345 P27112 INFO Save best model: monitor(max): 0.939846
2022-01-20 20:45:13,353 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:45:13,415 P27112 INFO Train loss: 0.301036
2022-01-20 20:45:13,415 P27112 INFO ************ Epoch=11 end ************
2022-01-20 20:45:21,394 P27112 INFO [Metrics] AUC: 0.940131 - logloss: 0.282080
2022-01-20 20:45:21,395 P27112 INFO Save best model: monitor(max): 0.940131
2022-01-20 20:45:21,404 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:45:21,454 P27112 INFO Train loss: 0.299421
2022-01-20 20:45:21,454 P27112 INFO ************ Epoch=12 end ************
2022-01-20 20:45:28,969 P27112 INFO [Metrics] AUC: 0.940509 - logloss: 0.281485
2022-01-20 20:45:28,970 P27112 INFO Save best model: monitor(max): 0.940509
2022-01-20 20:45:28,978 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:45:29,021 P27112 INFO Train loss: 0.297724
2022-01-20 20:45:29,021 P27112 INFO ************ Epoch=13 end ************
2022-01-20 20:45:36,802 P27112 INFO [Metrics] AUC: 0.940725 - logloss: 0.281523
2022-01-20 20:45:36,803 P27112 INFO Save best model: monitor(max): 0.940725
2022-01-20 20:45:36,811 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:45:36,870 P27112 INFO Train loss: 0.296200
2022-01-20 20:45:36,870 P27112 INFO ************ Epoch=14 end ************
2022-01-20 20:45:44,459 P27112 INFO [Metrics] AUC: 0.940797 - logloss: 0.280946
2022-01-20 20:45:44,460 P27112 INFO Save best model: monitor(max): 0.940797
2022-01-20 20:45:44,467 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:45:44,526 P27112 INFO Train loss: 0.294650
2022-01-20 20:45:44,527 P27112 INFO ************ Epoch=15 end ************
2022-01-20 20:45:52,971 P27112 INFO [Metrics] AUC: 0.941078 - logloss: 0.280969
2022-01-20 20:45:52,973 P27112 INFO Save best model: monitor(max): 0.941078
2022-01-20 20:45:52,981 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:45:53,045 P27112 INFO Train loss: 0.293139
2022-01-20 20:45:53,045 P27112 INFO ************ Epoch=16 end ************
2022-01-20 20:46:01,243 P27112 INFO [Metrics] AUC: 0.941160 - logloss: 0.280694
2022-01-20 20:46:01,244 P27112 INFO Save best model: monitor(max): 0.941160
2022-01-20 20:46:01,251 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:01,333 P27112 INFO Train loss: 0.291832
2022-01-20 20:46:01,334 P27112 INFO ************ Epoch=17 end ************
2022-01-20 20:46:09,156 P27112 INFO [Metrics] AUC: 0.941170 - logloss: 0.280836
2022-01-20 20:46:09,157 P27112 INFO Save best model: monitor(max): 0.941170
2022-01-20 20:46:09,163 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:09,210 P27112 INFO Train loss: 0.290554
2022-01-20 20:46:09,210 P27112 INFO ************ Epoch=18 end ************
2022-01-20 20:46:17,391 P27112 INFO [Metrics] AUC: 0.941211 - logloss: 0.280709
2022-01-20 20:46:17,392 P27112 INFO Save best model: monitor(max): 0.941211
2022-01-20 20:46:17,399 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:17,447 P27112 INFO Train loss: 0.288844
2022-01-20 20:46:17,447 P27112 INFO ************ Epoch=19 end ************
2022-01-20 20:46:25,420 P27112 INFO [Metrics] AUC: 0.941209 - logloss: 0.280904
2022-01-20 20:46:25,421 P27112 INFO Monitor(max) STOP: 0.941209 !
2022-01-20 20:46:25,421 P27112 INFO Reduce learning rate on plateau: 0.000100
2022-01-20 20:46:25,421 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:25,482 P27112 INFO Train loss: 0.287683
2022-01-20 20:46:25,482 P27112 INFO ************ Epoch=20 end ************
2022-01-20 20:46:33,389 P27112 INFO [Metrics] AUC: 0.943277 - logloss: 0.278695
2022-01-20 20:46:33,390 P27112 INFO Save best model: monitor(max): 0.943277
2022-01-20 20:46:33,399 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:33,457 P27112 INFO Train loss: 0.260785
2022-01-20 20:46:33,458 P27112 INFO ************ Epoch=21 end ************
2022-01-20 20:46:41,479 P27112 INFO [Metrics] AUC: 0.943987 - logloss: 0.277925
2022-01-20 20:46:41,481 P27112 INFO Save best model: monitor(max): 0.943987
2022-01-20 20:46:41,489 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:41,540 P27112 INFO Train loss: 0.251996
2022-01-20 20:46:41,540 P27112 INFO ************ Epoch=22 end ************
2022-01-20 20:46:48,055 P27112 INFO [Metrics] AUC: 0.944170 - logloss: 0.278795
2022-01-20 20:46:48,056 P27112 INFO Save best model: monitor(max): 0.944170
2022-01-20 20:46:48,064 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:48,115 P27112 INFO Train loss: 0.247396
2022-01-20 20:46:48,115 P27112 INFO ************ Epoch=23 end ************
2022-01-20 20:46:54,833 P27112 INFO [Metrics] AUC: 0.944000 - logloss: 0.280265
2022-01-20 20:46:54,834 P27112 INFO Monitor(max) STOP: 0.944000 !
2022-01-20 20:46:54,834 P27112 INFO Reduce learning rate on plateau: 0.000010
2022-01-20 20:46:54,834 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:46:54,890 P27112 INFO Train loss: 0.244364
2022-01-20 20:46:54,890 P27112 INFO ************ Epoch=24 end ************
2022-01-20 20:47:02,927 P27112 INFO [Metrics] AUC: 0.943998 - logloss: 0.281232
2022-01-20 20:47:02,928 P27112 INFO Monitor(max) STOP: 0.943998 !
2022-01-20 20:47:02,928 P27112 INFO Reduce learning rate on plateau: 0.000001
2022-01-20 20:47:02,928 P27112 INFO Early stopping at epoch=25
2022-01-20 20:47:02,928 P27112 INFO --- 343/343 batches finished ---
2022-01-20 20:47:03,000 P27112 INFO Train loss: 0.237771
2022-01-20 20:47:03,000 P27112 INFO Training finished.
2022-01-20 20:47:03,000 P27112 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/DCN_MovielensLatest_x1/movielenslatest_x1_cd32d937/DCN_movielenslatest_x1_006_6ae4ad60.model
2022-01-20 20:47:05,922 P27112 INFO ****** Validation evaluation ******
2022-01-20 20:47:07,575 P27112 INFO [Metrics] AUC: 0.944170 - logloss: 0.278795
2022-01-20 20:47:07,633 P27112 INFO ******** Test evaluation ********
2022-01-20 20:47:07,634 P27112 INFO Loading data...
2022-01-20 20:47:07,634 P27112 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-20 20:47:07,639 P27112 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-20 20:47:07,639 P27112 INFO Loading test data done.
2022-01-20 20:47:08,671 P27112 INFO [Metrics] AUC: 0.944032 - logloss: 0.279041

```
