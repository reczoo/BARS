## CIN_movielenslatest_x1

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_movielenslatest_x1_tuner_config_02](./CIN_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_movielenslatest_x1
    nohup python run_expid.py --config ./CIN_movielenslatest_x1_tuner_config_02 --expid xDeepFM_movielenslatest_x1_001_0de9d14a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.967421 | 0.208490  |


### Logs
```python
2022-01-20 08:19:08,267 P2894 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "cin_layer_units": "[32, 32, 32, 32]",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_hidden_units": "None",
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
    "model_id": "xDeepFM_movielenslatest_x1_001_0de9d14a",
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
2022-01-20 08:19:08,268 P2894 INFO Set up feature encoder...
2022-01-20 08:19:08,269 P2894 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-20 08:19:08,269 P2894 INFO Loading data...
2022-01-20 08:19:08,272 P2894 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-20 08:19:08,312 P2894 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-20 08:19:08,325 P2894 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-20 08:19:08,325 P2894 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-20 08:19:08,325 P2894 INFO Loading train data done.
2022-01-20 08:19:12,084 P2894 INFO Total number of parameters: 1002390.
2022-01-20 08:19:12,085 P2894 INFO Start training: 343 batches/epoch
2022-01-20 08:19:12,085 P2894 INFO ************ Epoch=1 start ************
2022-01-20 08:19:28,436 P2894 INFO [Metrics] AUC: 0.919298 - logloss: 0.316337
2022-01-20 08:19:28,437 P2894 INFO Save best model: monitor(max): 0.919298
2022-01-20 08:19:28,447 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:19:28,495 P2894 INFO Train loss: 0.448634
2022-01-20 08:19:28,495 P2894 INFO ************ Epoch=1 end ************
2022-01-20 08:19:44,807 P2894 INFO [Metrics] AUC: 0.922906 - logloss: 0.313155
2022-01-20 08:19:44,807 P2894 INFO Save best model: monitor(max): 0.922906
2022-01-20 08:19:44,815 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:19:44,865 P2894 INFO Train loss: 0.370033
2022-01-20 08:19:44,865 P2894 INFO ************ Epoch=2 end ************
2022-01-20 08:20:01,202 P2894 INFO [Metrics] AUC: 0.924106 - logloss: 0.311110
2022-01-20 08:20:01,203 P2894 INFO Save best model: monitor(max): 0.924106
2022-01-20 08:20:01,211 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:20:01,263 P2894 INFO Train loss: 0.362686
2022-01-20 08:20:01,263 P2894 INFO ************ Epoch=3 end ************
2022-01-20 08:20:17,627 P2894 INFO [Metrics] AUC: 0.925590 - logloss: 0.308131
2022-01-20 08:20:17,628 P2894 INFO Save best model: monitor(max): 0.925590
2022-01-20 08:20:17,636 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:20:17,692 P2894 INFO Train loss: 0.359155
2022-01-20 08:20:17,693 P2894 INFO ************ Epoch=4 end ************
2022-01-20 08:20:33,919 P2894 INFO [Metrics] AUC: 0.930822 - logloss: 0.295753
2022-01-20 08:20:33,920 P2894 INFO Save best model: monitor(max): 0.930822
2022-01-20 08:20:33,926 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:20:33,968 P2894 INFO Train loss: 0.356251
2022-01-20 08:20:33,968 P2894 INFO ************ Epoch=5 end ************
2022-01-20 08:20:50,290 P2894 INFO [Metrics] AUC: 0.936727 - logloss: 0.280760
2022-01-20 08:20:50,290 P2894 INFO Save best model: monitor(max): 0.936727
2022-01-20 08:20:50,296 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:20:50,351 P2894 INFO Train loss: 0.353547
2022-01-20 08:20:50,352 P2894 INFO ************ Epoch=6 end ************
2022-01-20 08:21:06,619 P2894 INFO [Metrics] AUC: 0.943390 - logloss: 0.264872
2022-01-20 08:21:06,620 P2894 INFO Save best model: monitor(max): 0.943390
2022-01-20 08:21:06,626 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:21:06,667 P2894 INFO Train loss: 0.349952
2022-01-20 08:21:06,667 P2894 INFO ************ Epoch=7 end ************
2022-01-20 08:21:22,921 P2894 INFO [Metrics] AUC: 0.949173 - logloss: 0.250694
2022-01-20 08:21:22,922 P2894 INFO Save best model: monitor(max): 0.949173
2022-01-20 08:21:22,930 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:21:22,981 P2894 INFO Train loss: 0.346874
2022-01-20 08:21:22,982 P2894 INFO ************ Epoch=8 end ************
2022-01-20 08:21:39,287 P2894 INFO [Metrics] AUC: 0.952457 - logloss: 0.243433
2022-01-20 08:21:39,288 P2894 INFO Save best model: monitor(max): 0.952457
2022-01-20 08:21:39,294 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:21:39,355 P2894 INFO Train loss: 0.342895
2022-01-20 08:21:39,355 P2894 INFO ************ Epoch=9 end ************
2022-01-20 08:21:55,719 P2894 INFO [Metrics] AUC: 0.953664 - logloss: 0.239521
2022-01-20 08:21:55,720 P2894 INFO Save best model: monitor(max): 0.953664
2022-01-20 08:21:55,727 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:21:55,791 P2894 INFO Train loss: 0.338629
2022-01-20 08:21:55,792 P2894 INFO ************ Epoch=10 end ************
2022-01-20 08:22:12,663 P2894 INFO [Metrics] AUC: 0.955220 - logloss: 0.236190
2022-01-20 08:22:12,665 P2894 INFO Save best model: monitor(max): 0.955220
2022-01-20 08:22:12,673 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:22:12,723 P2894 INFO Train loss: 0.334096
2022-01-20 08:22:12,723 P2894 INFO ************ Epoch=11 end ************
2022-01-20 08:22:28,947 P2894 INFO [Metrics] AUC: 0.955702 - logloss: 0.234898
2022-01-20 08:22:28,948 P2894 INFO Save best model: monitor(max): 0.955702
2022-01-20 08:22:28,955 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:22:29,017 P2894 INFO Train loss: 0.330885
2022-01-20 08:22:29,018 P2894 INFO ************ Epoch=12 end ************
2022-01-20 08:22:45,568 P2894 INFO [Metrics] AUC: 0.956666 - logloss: 0.232430
2022-01-20 08:22:45,569 P2894 INFO Save best model: monitor(max): 0.956666
2022-01-20 08:22:45,575 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:22:45,622 P2894 INFO Train loss: 0.328024
2022-01-20 08:22:45,623 P2894 INFO ************ Epoch=13 end ************
2022-01-20 08:23:01,792 P2894 INFO [Metrics] AUC: 0.957308 - logloss: 0.230661
2022-01-20 08:23:01,793 P2894 INFO Save best model: monitor(max): 0.957308
2022-01-20 08:23:01,799 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:23:01,841 P2894 INFO Train loss: 0.325466
2022-01-20 08:23:01,841 P2894 INFO ************ Epoch=14 end ************
2022-01-20 08:23:17,994 P2894 INFO [Metrics] AUC: 0.957438 - logloss: 0.231000
2022-01-20 08:23:17,995 P2894 INFO Save best model: monitor(max): 0.957438
2022-01-20 08:23:18,001 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:23:18,048 P2894 INFO Train loss: 0.324195
2022-01-20 08:23:18,048 P2894 INFO ************ Epoch=15 end ************
2022-01-20 08:23:34,542 P2894 INFO [Metrics] AUC: 0.957854 - logloss: 0.229508
2022-01-20 08:23:34,543 P2894 INFO Save best model: monitor(max): 0.957854
2022-01-20 08:23:34,550 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:23:34,599 P2894 INFO Train loss: 0.323339
2022-01-20 08:23:34,599 P2894 INFO ************ Epoch=16 end ************
2022-01-20 08:23:50,801 P2894 INFO [Metrics] AUC: 0.958591 - logloss: 0.228185
2022-01-20 08:23:50,802 P2894 INFO Save best model: monitor(max): 0.958591
2022-01-20 08:23:50,811 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:23:50,857 P2894 INFO Train loss: 0.324496
2022-01-20 08:23:50,857 P2894 INFO ************ Epoch=17 end ************
2022-01-20 08:24:06,996 P2894 INFO [Metrics] AUC: 0.958574 - logloss: 0.228368
2022-01-20 08:24:06,998 P2894 INFO Monitor(max) STOP: 0.958574 !
2022-01-20 08:24:06,998 P2894 INFO Reduce learning rate on plateau: 0.000100
2022-01-20 08:24:06,998 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:24:07,058 P2894 INFO Train loss: 0.324248
2022-01-20 08:24:07,058 P2894 INFO ************ Epoch=18 end ************
2022-01-20 08:24:23,128 P2894 INFO [Metrics] AUC: 0.966053 - logloss: 0.207699
2022-01-20 08:24:23,129 P2894 INFO Save best model: monitor(max): 0.966053
2022-01-20 08:24:23,135 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:24:23,213 P2894 INFO Train loss: 0.241276
2022-01-20 08:24:23,213 P2894 INFO ************ Epoch=19 end ************
2022-01-20 08:24:39,382 P2894 INFO [Metrics] AUC: 0.967570 - logloss: 0.207392
2022-01-20 08:24:39,383 P2894 INFO Save best model: monitor(max): 0.967570
2022-01-20 08:24:39,389 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:24:39,437 P2894 INFO Train loss: 0.185445
2022-01-20 08:24:39,437 P2894 INFO ************ Epoch=20 end ************
2022-01-20 08:24:55,533 P2894 INFO [Metrics] AUC: 0.967333 - logloss: 0.214755
2022-01-20 08:24:55,534 P2894 INFO Monitor(max) STOP: 0.967333 !
2022-01-20 08:24:55,534 P2894 INFO Reduce learning rate on plateau: 0.000010
2022-01-20 08:24:55,534 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:24:55,587 P2894 INFO Train loss: 0.156436
2022-01-20 08:24:55,587 P2894 INFO ************ Epoch=21 end ************
2022-01-20 08:25:11,723 P2894 INFO [Metrics] AUC: 0.967358 - logloss: 0.216456
2022-01-20 08:25:11,724 P2894 INFO Monitor(max) STOP: 0.967358 !
2022-01-20 08:25:11,724 P2894 INFO Reduce learning rate on plateau: 0.000001
2022-01-20 08:25:11,724 P2894 INFO Early stopping at epoch=22
2022-01-20 08:25:11,725 P2894 INFO --- 343/343 batches finished ---
2022-01-20 08:25:11,774 P2894 INFO Train loss: 0.131596
2022-01-20 08:25:11,774 P2894 INFO Training finished.
2022-01-20 08:25:11,774 P2894 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/xDeepFM_movielenslatest_x1/movielenslatest_x1_cd32d937/xDeepFM_movielenslatest_x1_001_0de9d14a.model
2022-01-20 08:25:11,785 P2894 INFO ****** Validation evaluation ******
2022-01-20 08:25:13,018 P2894 INFO [Metrics] AUC: 0.967570 - logloss: 0.207392
2022-01-20 08:25:13,064 P2894 INFO ******** Test evaluation ********
2022-01-20 08:25:13,065 P2894 INFO Loading data...
2022-01-20 08:25:13,065 P2894 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-20 08:25:13,071 P2894 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-20 08:25:13,071 P2894 INFO Loading test data done.
2022-01-20 08:25:13,747 P2894 INFO [Metrics] AUC: 0.967421 - logloss: 0.208490

```
