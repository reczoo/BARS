## HFM+_movielenslatest_x1

A hands-on guide to run the HFM model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_movielenslatest_x1_tuner_config_03](./HFM+_movielenslatest_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_movielenslatest_x1
    nohup python run_expid.py --config ./HFM+_movielenslatest_x1_tuner_config_03 --expid HFM_movielenslatest_x1_007_91dbbe82 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.956971 | 0.264447  |


### Logs
```python
2022-02-01 21:04:06,625 P37760 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HFM",
    "model_id": "HFM_movielenslatest_x1_007_91dbbe82",
    "model_root": "./Movielens/HFM_movielenslatest_x1/",
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
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-01 21:04:06,625 P37760 INFO Set up feature encoder...
2022-02-01 21:04:06,625 P37760 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-01 21:04:06,626 P37760 INFO Loading data...
2022-02-01 21:04:06,628 P37760 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-01 21:04:06,658 P37760 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-01 21:04:06,668 P37760 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-01 21:04:06,668 P37760 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-01 21:04:06,668 P37760 INFO Loading train data done.
2022-02-01 21:04:10,528 P37760 INFO Total number of parameters: 1328631.
2022-02-01 21:04:10,529 P37760 INFO Start training: 343 batches/epoch
2022-02-01 21:04:10,529 P37760 INFO ************ Epoch=1 start ************
2022-02-01 21:04:30,779 P37760 INFO [Metrics] AUC: 0.922170 - logloss: 1.496585
2022-02-01 21:04:30,780 P37760 INFO Save best model: monitor(max): 0.922170
2022-02-01 21:04:30,787 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:04:30,844 P37760 INFO Train loss: 0.398816
2022-02-01 21:04:30,844 P37760 INFO ************ Epoch=1 end ************
2022-02-01 21:04:50,775 P37760 INFO [Metrics] AUC: 0.930850 - logloss: 0.450947
2022-02-01 21:04:50,776 P37760 INFO Save best model: monitor(max): 0.930850
2022-02-01 21:04:50,785 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:04:50,832 P37760 INFO Train loss: 0.363642
2022-02-01 21:04:50,832 P37760 INFO ************ Epoch=2 end ************
2022-02-01 21:05:10,847 P37760 INFO [Metrics] AUC: 0.934527 - logloss: 0.319688
2022-02-01 21:05:10,847 P37760 INFO Save best model: monitor(max): 0.934527
2022-02-01 21:05:10,856 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:05:10,898 P37760 INFO Train loss: 0.371298
2022-02-01 21:05:10,898 P37760 INFO ************ Epoch=3 end ************
2022-02-01 21:05:30,721 P37760 INFO [Metrics] AUC: 0.928523 - logloss: 0.747766
2022-02-01 21:05:30,722 P37760 INFO Monitor(max) STOP: 0.928523 !
2022-02-01 21:05:30,722 P37760 INFO Reduce learning rate on plateau: 0.000100
2022-02-01 21:05:30,722 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:05:30,765 P37760 INFO Train loss: 0.379755
2022-02-01 21:05:30,765 P37760 INFO ************ Epoch=4 end ************
2022-02-01 21:05:50,609 P37760 INFO [Metrics] AUC: 0.955322 - logloss: 0.267231
2022-02-01 21:05:50,610 P37760 INFO Save best model: monitor(max): 0.955322
2022-02-01 21:05:50,619 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:05:50,662 P37760 INFO Train loss: 0.291268
2022-02-01 21:05:50,662 P37760 INFO ************ Epoch=5 end ************
2022-02-01 21:06:10,428 P37760 INFO [Metrics] AUC: 0.954717 - logloss: 0.311356
2022-02-01 21:06:10,429 P37760 INFO Monitor(max) STOP: 0.954717 !
2022-02-01 21:06:10,429 P37760 INFO Reduce learning rate on plateau: 0.000010
2022-02-01 21:06:10,429 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:06:10,470 P37760 INFO Train loss: 0.211864
2022-02-01 21:06:10,471 P37760 INFO ************ Epoch=6 end ************
2022-02-01 21:06:27,317 P37760 INFO [Metrics] AUC: 0.956160 - logloss: 0.267137
2022-02-01 21:06:27,317 P37760 INFO Save best model: monitor(max): 0.956160
2022-02-01 21:06:27,326 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:06:27,367 P37760 INFO Train loss: 0.159947
2022-02-01 21:06:27,367 P37760 INFO ************ Epoch=7 end ************
2022-02-01 21:06:41,298 P37760 INFO [Metrics] AUC: 0.955332 - logloss: 0.279518
2022-02-01 21:06:41,299 P37760 INFO Monitor(max) STOP: 0.955332 !
2022-02-01 21:06:41,299 P37760 INFO Reduce learning rate on plateau: 0.000001
2022-02-01 21:06:41,299 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:06:41,366 P37760 INFO Train loss: 0.152000
2022-02-01 21:06:41,367 P37760 INFO ************ Epoch=8 end ************
2022-02-01 21:06:59,985 P37760 INFO [Metrics] AUC: 0.955571 - logloss: 0.278921
2022-02-01 21:06:59,986 P37760 INFO Monitor(max) STOP: 0.955571 !
2022-02-01 21:06:59,986 P37760 INFO Reduce learning rate on plateau: 0.000001
2022-02-01 21:06:59,986 P37760 INFO Early stopping at epoch=9
2022-02-01 21:06:59,986 P37760 INFO --- 343/343 batches finished ---
2022-02-01 21:07:00,026 P37760 INFO Train loss: 0.146438
2022-02-01 21:07:00,026 P37760 INFO Training finished.
2022-02-01 21:07:00,027 P37760 INFO Load best model: /home/XXX/benchmarks/Movielens/HFM_movielenslatest_x1/movielenslatest_x1_cd32d937/HFM_movielenslatest_x1_007_91dbbe82.model
2022-02-01 21:07:00,044 P37760 INFO ****** Validation evaluation ******
2022-02-01 21:07:01,613 P37760 INFO [Metrics] AUC: 0.956160 - logloss: 0.267137
2022-02-01 21:07:01,652 P37760 INFO ******** Test evaluation ********
2022-02-01 21:07:01,652 P37760 INFO Loading data...
2022-02-01 21:07:01,652 P37760 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-01 21:07:01,657 P37760 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-01 21:07:01,657 P37760 INFO Loading test data done.
2022-02-01 21:07:02,423 P37760 INFO [Metrics] AUC: 0.956971 - logloss: 0.264447

```
