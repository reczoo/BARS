## AutoInt+_movielenslatest_x1

A hands-on guide to run the AutoInt model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt+_movielenslatest_x1_tuner_config_01](./AutoInt+_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt+_movielenslatest_x1
    nohup python run_expid.py --config ./AutoInt+_movielenslatest_x1_tuner_config_01 --expid AutoInt_movielenslatest_x1_008_47369e2c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.969291 | 0.216509  |
| 2 | 0.968015 | 0.216154  |
| 3 | 0.968559 | 0.215716  |
| 4 | 0.968710 | 0.218018  |
| 5 | 0.968226 | 0.215047  |
| | | | 
| Avg | 0.968560 | 0.216289 |
| Std | &#177;0.00043942 | &#177;0.00099267 |


### Logs
```python
2022-01-24 13:48:58,617 P44673 INFO {
    "attention_dim": "128",
    "attention_layers": "5",
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "True",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_movielenslatest_x1_008_47369e2c",
    "model_root": "./Movielens/AutoInt_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_heads": "1",
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
    "use_residual": "True",
    "use_scale": "True",
    "use_wide": "False",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-24 13:48:58,618 P44673 INFO Set up feature encoder...
2022-01-24 13:48:58,618 P44673 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-24 13:48:58,618 P44673 INFO Loading data...
2022-01-24 13:48:58,620 P44673 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-24 13:48:58,648 P44673 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-24 13:48:58,656 P44673 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-24 13:48:58,656 P44673 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-24 13:48:58,656 P44673 INFO Loading train data done.
2022-01-24 13:49:02,641 P44673 INFO Total number of parameters: 1441784.
2022-01-24 13:49:02,642 P44673 INFO Start training: 343 batches/epoch
2022-01-24 13:49:02,642 P44673 INFO ************ Epoch=1 start ************
2022-01-24 13:50:05,821 P44673 INFO [Metrics] AUC: 0.935528 - logloss: 0.289432
2022-01-24 13:50:05,822 P44673 INFO Save best model: monitor(max): 0.935528
2022-01-24 13:50:05,831 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:50:05,865 P44673 INFO Train loss: 0.381427
2022-01-24 13:50:05,865 P44673 INFO ************ Epoch=1 end ************
2022-01-24 13:51:08,979 P44673 INFO [Metrics] AUC: 0.945807 - logloss: 0.264319
2022-01-24 13:51:08,979 P44673 INFO Save best model: monitor(max): 0.945807
2022-01-24 13:51:08,991 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:51:09,030 P44673 INFO Train loss: 0.357559
2022-01-24 13:51:09,030 P44673 INFO ************ Epoch=2 end ************
2022-01-24 13:52:12,458 P44673 INFO [Metrics] AUC: 0.948922 - logloss: 0.253205
2022-01-24 13:52:12,459 P44673 INFO Save best model: monitor(max): 0.948922
2022-01-24 13:52:12,470 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:52:12,508 P44673 INFO Train loss: 0.361583
2022-01-24 13:52:12,508 P44673 INFO ************ Epoch=3 end ************
2022-01-24 13:53:15,847 P44673 INFO [Metrics] AUC: 0.951293 - logloss: 0.247246
2022-01-24 13:53:15,848 P44673 INFO Save best model: monitor(max): 0.951293
2022-01-24 13:53:15,860 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:53:15,897 P44673 INFO Train loss: 0.367173
2022-01-24 13:53:15,897 P44673 INFO ************ Epoch=4 end ************
2022-01-24 13:54:16,461 P44673 INFO [Metrics] AUC: 0.952467 - logloss: 0.248028
2022-01-24 13:54:16,462 P44673 INFO Save best model: monitor(max): 0.952467
2022-01-24 13:54:16,474 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:54:16,516 P44673 INFO Train loss: 0.370421
2022-01-24 13:54:16,517 P44673 INFO ************ Epoch=5 end ************
2022-01-24 13:55:19,715 P44673 INFO [Metrics] AUC: 0.952714 - logloss: 0.244469
2022-01-24 13:55:19,716 P44673 INFO Save best model: monitor(max): 0.952714
2022-01-24 13:55:19,729 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:55:19,765 P44673 INFO Train loss: 0.373596
2022-01-24 13:55:19,765 P44673 INFO ************ Epoch=6 end ************
2022-01-24 13:56:22,837 P44673 INFO [Metrics] AUC: 0.954175 - logloss: 0.239273
2022-01-24 13:56:22,838 P44673 INFO Save best model: monitor(max): 0.954175
2022-01-24 13:56:22,848 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:56:22,887 P44673 INFO Train loss: 0.376194
2022-01-24 13:56:22,887 P44673 INFO ************ Epoch=7 end ************
2022-01-24 13:57:25,704 P44673 INFO [Metrics] AUC: 0.954548 - logloss: 0.239112
2022-01-24 13:57:25,704 P44673 INFO Save best model: monitor(max): 0.954548
2022-01-24 13:57:25,715 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:57:25,753 P44673 INFO Train loss: 0.377089
2022-01-24 13:57:25,753 P44673 INFO ************ Epoch=8 end ************
2022-01-24 13:58:28,476 P44673 INFO [Metrics] AUC: 0.954995 - logloss: 0.236400
2022-01-24 13:58:28,476 P44673 INFO Save best model: monitor(max): 0.954995
2022-01-24 13:58:28,486 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:58:28,523 P44673 INFO Train loss: 0.377147
2022-01-24 13:58:28,523 P44673 INFO ************ Epoch=9 end ************
2022-01-24 13:59:31,589 P44673 INFO [Metrics] AUC: 0.955776 - logloss: 0.234188
2022-01-24 13:59:31,590 P44673 INFO Save best model: monitor(max): 0.955776
2022-01-24 13:59:31,601 P44673 INFO --- 343/343 batches finished ---
2022-01-24 13:59:31,640 P44673 INFO Train loss: 0.377498
2022-01-24 13:59:31,641 P44673 INFO ************ Epoch=10 end ************
2022-01-24 14:00:34,848 P44673 INFO [Metrics] AUC: 0.955859 - logloss: 0.233173
2022-01-24 14:00:34,849 P44673 INFO Save best model: monitor(max): 0.955859
2022-01-24 14:00:34,859 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:00:34,899 P44673 INFO Train loss: 0.377892
2022-01-24 14:00:34,900 P44673 INFO ************ Epoch=11 end ************
2022-01-24 14:01:37,806 P44673 INFO [Metrics] AUC: 0.956709 - logloss: 0.232684
2022-01-24 14:01:37,807 P44673 INFO Save best model: monitor(max): 0.956709
2022-01-24 14:01:37,817 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:01:37,855 P44673 INFO Train loss: 0.379856
2022-01-24 14:01:37,855 P44673 INFO ************ Epoch=12 end ************
2022-01-24 14:02:40,938 P44673 INFO [Metrics] AUC: 0.956988 - logloss: 0.231457
2022-01-24 14:02:40,939 P44673 INFO Save best model: monitor(max): 0.956988
2022-01-24 14:02:40,950 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:02:40,990 P44673 INFO Train loss: 0.379921
2022-01-24 14:02:40,990 P44673 INFO ************ Epoch=13 end ************
2022-01-24 14:03:43,830 P44673 INFO [Metrics] AUC: 0.957131 - logloss: 0.233748
2022-01-24 14:03:43,830 P44673 INFO Save best model: monitor(max): 0.957131
2022-01-24 14:03:43,841 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:03:43,880 P44673 INFO Train loss: 0.379892
2022-01-24 14:03:43,880 P44673 INFO ************ Epoch=14 end ************
2022-01-24 14:04:46,494 P44673 INFO [Metrics] AUC: 0.957507 - logloss: 0.230708
2022-01-24 14:04:46,494 P44673 INFO Save best model: monitor(max): 0.957507
2022-01-24 14:04:46,504 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:04:46,544 P44673 INFO Train loss: 0.379719
2022-01-24 14:04:46,544 P44673 INFO ************ Epoch=15 end ************
2022-01-24 14:05:44,888 P44673 INFO [Metrics] AUC: 0.957810 - logloss: 0.230643
2022-01-24 14:05:44,889 P44673 INFO Save best model: monitor(max): 0.957810
2022-01-24 14:05:44,899 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:05:44,938 P44673 INFO Train loss: 0.380646
2022-01-24 14:05:44,939 P44673 INFO ************ Epoch=16 end ************
2022-01-24 14:06:44,484 P44673 INFO [Metrics] AUC: 0.957410 - logloss: 0.230534
2022-01-24 14:06:44,484 P44673 INFO Monitor(max) STOP: 0.957410 !
2022-01-24 14:06:44,484 P44673 INFO Reduce learning rate on plateau: 0.000100
2022-01-24 14:06:44,484 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:06:44,526 P44673 INFO Train loss: 0.380014
2022-01-24 14:06:44,526 P44673 INFO ************ Epoch=17 end ************
2022-01-24 14:07:52,592 P44673 INFO [Metrics] AUC: 0.968349 - logloss: 0.205559
2022-01-24 14:07:52,592 P44673 INFO Save best model: monitor(max): 0.968349
2022-01-24 14:07:52,603 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:07:52,643 P44673 INFO Train loss: 0.274646
2022-01-24 14:07:52,643 P44673 INFO ************ Epoch=18 end ************
2022-01-24 14:09:00,436 P44673 INFO [Metrics] AUC: 0.969530 - logloss: 0.215277
2022-01-24 14:09:00,436 P44673 INFO Save best model: monitor(max): 0.969530
2022-01-24 14:09:00,447 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:09:00,486 P44673 INFO Train loss: 0.185666
2022-01-24 14:09:00,487 P44673 INFO ************ Epoch=19 end ************
2022-01-24 14:10:08,403 P44673 INFO [Metrics] AUC: 0.967866 - logloss: 0.243179
2022-01-24 14:10:08,404 P44673 INFO Monitor(max) STOP: 0.967866 !
2022-01-24 14:10:08,404 P44673 INFO Reduce learning rate on plateau: 0.000010
2022-01-24 14:10:08,404 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:10:08,441 P44673 INFO Train loss: 0.140991
2022-01-24 14:10:08,442 P44673 INFO ************ Epoch=20 end ************
2022-01-24 14:11:15,973 P44673 INFO [Metrics] AUC: 0.967725 - logloss: 0.257149
2022-01-24 14:11:15,974 P44673 INFO Monitor(max) STOP: 0.967725 !
2022-01-24 14:11:15,974 P44673 INFO Reduce learning rate on plateau: 0.000001
2022-01-24 14:11:15,974 P44673 INFO Early stopping at epoch=21
2022-01-24 14:11:15,974 P44673 INFO --- 343/343 batches finished ---
2022-01-24 14:11:16,015 P44673 INFO Train loss: 0.109575
2022-01-24 14:11:16,016 P44673 INFO Training finished.
2022-01-24 14:11:16,016 P44673 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/AutoInt_movielenslatest_x1/movielenslatest_x1_cd32d937/AutoInt_movielenslatest_x1_008_47369e2c.model
2022-01-24 14:11:21,821 P44673 INFO ****** Validation evaluation ******
2022-01-24 14:11:25,653 P44673 INFO [Metrics] AUC: 0.969530 - logloss: 0.215277
2022-01-24 14:11:25,694 P44673 INFO ******** Test evaluation ********
2022-01-24 14:11:25,694 P44673 INFO Loading data...
2022-01-24 14:11:25,694 P44673 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-24 14:11:25,698 P44673 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-24 14:11:25,698 P44673 INFO Loading test data done.
2022-01-24 14:11:27,490 P44673 INFO [Metrics] AUC: 0.969291 - logloss: 0.216509

```
