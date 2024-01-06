## DSSM_movielenslatest_x1

A hands-on guide to run the DSSM model on the MovielensLatest_x1 dataset.

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
  fuxictr: 1.2.1
  ```

### Dataset
Dataset ID: [MovielensLatest_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/MovieLens#MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/tree/v1.2.1) for this experiment. See the model code: [DSSM](https://github.com/xue-pai/FuxiCTR/blob/v1.2.1/fuxictr/pytorch/models/DSSM.py).

Running steps:

1. Download [FuxiCTR-v1.2.1](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.2.1.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DSSM_movielenslatest_x1_tuner_config_01](./DSSM_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DSSM_movielenslatest_x1
    nohup python run_expid.py --config ./DSSM_movielenslatest_x1_tuner_config_01 --expid DSSM_movielenslatest_x1_001_945a31b2 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.968624 | 0.212998  |


### Logs
```python
2022-06-13 08:57:52,198 P303 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_06dcf7a5",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': 'user_id', 'source': 'user', 'type': 'categorical'}, {'active': True, 'dtype': 'float', 'name': ['item_id', 'tag_id'], 'source': 'item', 'type': 'categorical'}]",
    "gpu": "0",
    "item_tower_activations": "ReLU",
    "item_tower_dropout": "0.2",
    "item_tower_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DSSM",
    "model_id": "DSSM_movielenslatest_x1_001_945a31b2",
    "model_root": "./Movielens/DSSM_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Movielens/MovielensLatest_x1/test.csv",
    "train_data": "../data/Movielens/MovielensLatest_x1/train.csv",
    "use_hdf5": "True",
    "user_tower_activations": "ReLU",
    "user_tower_dropout": "0.2",
    "user_tower_units": "[400, 400, 400]",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-06-13 08:57:52,199 P303 INFO Set up feature encoder...
2022-06-13 08:57:52,200 P303 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_06dcf7a5/feature_map.json
2022-06-13 08:57:52,200 P303 INFO Loading data...
2022-06-13 08:57:52,203 P303 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_06dcf7a5/train.h5
2022-06-13 08:57:52,232 P303 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_06dcf7a5/valid.h5
2022-06-13 08:57:52,244 P303 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-06-13 08:57:52,244 P303 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-06-13 08:57:52,244 P303 INFO Loading train data done.
2022-06-13 08:57:55,243 P303 INFO Total number of parameters: 1559990.
2022-06-13 08:57:55,244 P303 INFO Start training: 343 batches/epoch
2022-06-13 08:57:55,244 P303 INFO ************ Epoch=1 start ************
2022-06-13 08:58:13,835 P303 INFO [Metrics] AUC: 0.933130 - logloss: 0.298132
2022-06-13 08:58:13,836 P303 INFO Save best model: monitor(max): 0.933130
2022-06-13 08:58:13,850 P303 INFO --- 343/343 batches finished ---
2022-06-13 08:58:13,908 P303 INFO Train loss: 0.455415
2022-06-13 08:58:13,908 P303 INFO ************ Epoch=1 end ************
2022-06-13 08:58:32,351 P303 INFO [Metrics] AUC: 0.939931 - logloss: 0.296643
2022-06-13 08:58:32,352 P303 INFO Save best model: monitor(max): 0.939931
2022-06-13 08:58:32,365 P303 INFO --- 343/343 batches finished ---
2022-06-13 08:58:32,435 P303 INFO Train loss: 0.363026
2022-06-13 08:58:32,435 P303 INFO ************ Epoch=2 end ************
2022-06-13 08:58:50,371 P303 INFO [Metrics] AUC: 0.946660 - logloss: 0.281723
2022-06-13 08:58:50,372 P303 INFO Save best model: monitor(max): 0.946660
2022-06-13 08:58:50,385 P303 INFO --- 343/343 batches finished ---
2022-06-13 08:58:50,447 P303 INFO Train loss: 0.370799
2022-06-13 08:58:50,447 P303 INFO ************ Epoch=3 end ************
2022-06-13 08:59:08,319 P303 INFO [Metrics] AUC: 0.948210 - logloss: 0.265915
2022-06-13 08:59:08,320 P303 INFO Save best model: monitor(max): 0.948210
2022-06-13 08:59:08,332 P303 INFO --- 343/343 batches finished ---
2022-06-13 08:59:08,415 P303 INFO Train loss: 0.373187
2022-06-13 08:59:08,415 P303 INFO ************ Epoch=4 end ************
2022-06-13 08:59:26,250 P303 INFO [Metrics] AUC: 0.951196 - logloss: 0.249135
2022-06-13 08:59:26,252 P303 INFO Save best model: monitor(max): 0.951196
2022-06-13 08:59:26,265 P303 INFO --- 343/343 batches finished ---
2022-06-13 08:59:26,314 P303 INFO Train loss: 0.374471
2022-06-13 08:59:26,314 P303 INFO ************ Epoch=5 end ************
2022-06-13 08:59:44,485 P303 INFO [Metrics] AUC: 0.953276 - logloss: 0.243052
2022-06-13 08:59:44,486 P303 INFO Save best model: monitor(max): 0.953276
2022-06-13 08:59:44,499 P303 INFO --- 343/343 batches finished ---
2022-06-13 08:59:44,548 P303 INFO Train loss: 0.378173
2022-06-13 08:59:44,548 P303 INFO ************ Epoch=6 end ************
2022-06-13 09:00:02,650 P303 INFO [Metrics] AUC: 0.954119 - logloss: 0.240459
2022-06-13 09:00:02,651 P303 INFO Save best model: monitor(max): 0.954119
2022-06-13 09:00:02,661 P303 INFO --- 343/343 batches finished ---
2022-06-13 09:00:02,718 P303 INFO Train loss: 0.379224
2022-06-13 09:00:02,718 P303 INFO ************ Epoch=7 end ************
2022-06-13 09:00:19,077 P303 INFO [Metrics] AUC: 0.954879 - logloss: 0.239145
2022-06-13 09:00:19,078 P303 INFO Save best model: monitor(max): 0.954879
2022-06-13 09:00:19,093 P303 INFO --- 343/343 batches finished ---
2022-06-13 09:00:19,134 P303 INFO Train loss: 0.379803
2022-06-13 09:00:19,134 P303 INFO ************ Epoch=8 end ************
2022-06-13 09:00:34,525 P303 INFO [Metrics] AUC: 0.946124 - logloss: 0.312645
2022-06-13 09:00:34,526 P303 INFO Monitor(max) STOP: 0.946124 !
2022-06-13 09:00:34,526 P303 INFO Reduce learning rate on plateau: 0.000100
2022-06-13 09:00:34,526 P303 INFO --- 343/343 batches finished ---
2022-06-13 09:00:34,569 P303 INFO Train loss: 0.382595
2022-06-13 09:00:34,569 P303 INFO ************ Epoch=9 end ************
2022-06-13 09:00:49,796 P303 INFO [Metrics] AUC: 0.967127 - logloss: 0.208134
2022-06-13 09:00:49,797 P303 INFO Save best model: monitor(max): 0.967127
2022-06-13 09:00:49,806 P303 INFO --- 343/343 batches finished ---
2022-06-13 09:00:49,859 P303 INFO Train loss: 0.285253
2022-06-13 09:00:49,860 P303 INFO ************ Epoch=10 end ************
2022-06-13 09:01:05,454 P303 INFO [Metrics] AUC: 0.969027 - logloss: 0.211825
2022-06-13 09:01:05,454 P303 INFO Save best model: monitor(max): 0.969027
2022-06-13 09:01:05,467 P303 INFO --- 343/343 batches finished ---
2022-06-13 09:01:05,525 P303 INFO Train loss: 0.197720
2022-06-13 09:01:05,525 P303 INFO ************ Epoch=11 end ************
2022-06-13 09:01:20,880 P303 INFO [Metrics] AUC: 0.968178 - logloss: 0.234724
2022-06-13 09:01:20,881 P303 INFO Monitor(max) STOP: 0.968178 !
2022-06-13 09:01:20,881 P303 INFO Reduce learning rate on plateau: 0.000010
2022-06-13 09:01:20,881 P303 INFO --- 343/343 batches finished ---
2022-06-13 09:01:20,935 P303 INFO Train loss: 0.152829
2022-06-13 09:01:20,935 P303 INFO ************ Epoch=12 end ************
2022-06-13 09:01:32,498 P303 INFO [Metrics] AUC: 0.967691 - logloss: 0.269013
2022-06-13 09:01:32,499 P303 INFO Monitor(max) STOP: 0.967691 !
2022-06-13 09:01:32,499 P303 INFO Reduce learning rate on plateau: 0.000001
2022-06-13 09:01:32,499 P303 INFO Early stopping at epoch=13
2022-06-13 09:01:32,500 P303 INFO --- 343/343 batches finished ---
2022-06-13 09:01:32,566 P303 INFO Train loss: 0.118317
2022-06-13 09:01:32,567 P303 INFO Training finished.
2022-06-13 09:01:32,567 P303 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/DSSM_movielenslatest_x1/movielenslatest_x1_06dcf7a5/DSSM_movielenslatest_x1_001_945a31b2.model
2022-06-13 09:01:32,612 P303 INFO ****** Validation evaluation ******
2022-06-13 09:01:34,111 P303 INFO [Metrics] AUC: 0.969027 - logloss: 0.211825
2022-06-13 09:01:34,162 P303 INFO ******** Test evaluation ********
2022-06-13 09:01:34,162 P303 INFO Loading data...
2022-06-13 09:01:34,163 P303 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_06dcf7a5/test.h5
2022-06-13 09:01:34,168 P303 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-06-13 09:01:34,168 P303 INFO Loading test data done.
2022-06-13 09:01:34,867 P303 INFO [Metrics] AUC: 0.968624 - logloss: 0.212998

```
