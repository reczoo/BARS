## AutoInt_movielenslatest_x1

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

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_movielenslatest_x1_tuner_config_02](./AutoInt_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_movielenslatest_x1
    nohup python run_expid.py --config ./AutoInt_movielenslatest_x1_tuner_config_02 --expid AutoInt_movielenslatest_x1_025_f1ca3a7a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.967161 | 0.224678  |


### Logs
```python
2022-01-25 15:07:37,101 P27992 INFO {
    "attention_dim": "128",
    "attention_layers": "6",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_movielenslatest_x1_025_f1ca3a7a",
    "model_root": "./Movielens/AutoInt_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "2",
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
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-25 15:07:37,102 P27992 INFO Set up feature encoder...
2022-01-25 15:07:37,102 P27992 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-25 15:07:37,102 P27992 INFO Loading data...
2022-01-25 15:07:37,105 P27992 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-25 15:07:37,133 P27992 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-25 15:07:37,141 P27992 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-25 15:07:37,141 P27992 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-25 15:07:37,141 P27992 INFO Loading train data done.
2022-01-25 15:07:42,941 P27992 INFO Total number of parameters: 1896439.
2022-01-25 15:07:42,942 P27992 INFO Start training: 343 batches/epoch
2022-01-25 15:07:42,942 P27992 INFO ************ Epoch=1 start ************
2022-01-25 15:09:57,609 P27992 INFO [Metrics] AUC: 0.930930 - logloss: 0.296814
2022-01-25 15:09:57,609 P27992 INFO Save best model: monitor(max): 0.930930
2022-01-25 15:09:57,618 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:09:57,659 P27992 INFO Train loss: 0.409041
2022-01-25 15:09:57,659 P27992 INFO ************ Epoch=1 end ************
2022-01-25 15:12:13,786 P27992 INFO [Metrics] AUC: 0.937707 - logloss: 0.283186
2022-01-25 15:12:13,787 P27992 INFO Save best model: monitor(max): 0.937707
2022-01-25 15:12:13,797 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:12:13,838 P27992 INFO Train loss: 0.360653
2022-01-25 15:12:13,838 P27992 INFO ************ Epoch=2 end ************
2022-01-25 15:14:26,669 P27992 INFO [Metrics] AUC: 0.940930 - logloss: 0.276955
2022-01-25 15:14:26,670 P27992 INFO Save best model: monitor(max): 0.940930
2022-01-25 15:14:26,680 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:14:26,720 P27992 INFO Train loss: 0.363341
2022-01-25 15:14:26,720 P27992 INFO ************ Epoch=3 end ************
2022-01-25 15:16:42,812 P27992 INFO [Metrics] AUC: 0.944349 - logloss: 0.267558
2022-01-25 15:16:42,813 P27992 INFO Save best model: monitor(max): 0.944349
2022-01-25 15:16:42,824 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:16:42,868 P27992 INFO Train loss: 0.368047
2022-01-25 15:16:42,868 P27992 INFO ************ Epoch=4 end ************
2022-01-25 15:18:57,310 P27992 INFO [Metrics] AUC: 0.946384 - logloss: 0.260087
2022-01-25 15:18:57,311 P27992 INFO Save best model: monitor(max): 0.946384
2022-01-25 15:18:57,322 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:18:57,364 P27992 INFO Train loss: 0.372034
2022-01-25 15:18:57,364 P27992 INFO ************ Epoch=5 end ************
2022-01-25 15:21:12,103 P27992 INFO [Metrics] AUC: 0.947952 - logloss: 0.256824
2022-01-25 15:21:12,104 P27992 INFO Save best model: monitor(max): 0.947952
2022-01-25 15:21:12,115 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:21:12,161 P27992 INFO Train loss: 0.374591
2022-01-25 15:21:12,161 P27992 INFO ************ Epoch=6 end ************
2022-01-25 15:23:27,843 P27992 INFO [Metrics] AUC: 0.949421 - logloss: 0.252406
2022-01-25 15:23:27,844 P27992 INFO Save best model: monitor(max): 0.949421
2022-01-25 15:23:27,854 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:23:27,895 P27992 INFO Train loss: 0.376642
2022-01-25 15:23:27,895 P27992 INFO ************ Epoch=7 end ************
2022-01-25 15:25:41,110 P27992 INFO [Metrics] AUC: 0.950922 - logloss: 0.249054
2022-01-25 15:25:41,111 P27992 INFO Save best model: monitor(max): 0.950922
2022-01-25 15:25:41,122 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:25:41,169 P27992 INFO Train loss: 0.378831
2022-01-25 15:25:41,169 P27992 INFO ************ Epoch=8 end ************
2022-01-25 15:27:55,835 P27992 INFO [Metrics] AUC: 0.951271 - logloss: 0.247405
2022-01-25 15:27:55,836 P27992 INFO Save best model: monitor(max): 0.951271
2022-01-25 15:27:55,846 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:27:55,891 P27992 INFO Train loss: 0.379660
2022-01-25 15:27:55,891 P27992 INFO ************ Epoch=9 end ************
2022-01-25 15:30:11,319 P27992 INFO [Metrics] AUC: 0.952158 - logloss: 0.245122
2022-01-25 15:30:11,319 P27992 INFO Save best model: monitor(max): 0.952158
2022-01-25 15:30:11,330 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:30:11,370 P27992 INFO Train loss: 0.380731
2022-01-25 15:30:11,370 P27992 INFO ************ Epoch=10 end ************
2022-01-25 15:32:25,702 P27992 INFO [Metrics] AUC: 0.953572 - logloss: 0.242195
2022-01-25 15:32:25,703 P27992 INFO Save best model: monitor(max): 0.953572
2022-01-25 15:32:25,714 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:32:25,757 P27992 INFO Train loss: 0.381514
2022-01-25 15:32:25,757 P27992 INFO ************ Epoch=11 end ************
2022-01-25 15:34:40,857 P27992 INFO [Metrics] AUC: 0.953654 - logloss: 0.241329
2022-01-25 15:34:40,858 P27992 INFO Save best model: monitor(max): 0.953654
2022-01-25 15:34:40,868 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:34:40,914 P27992 INFO Train loss: 0.382920
2022-01-25 15:34:40,914 P27992 INFO ************ Epoch=12 end ************
2022-01-25 15:36:57,181 P27992 INFO [Metrics] AUC: 0.954173 - logloss: 0.239480
2022-01-25 15:36:57,181 P27992 INFO Save best model: monitor(max): 0.954173
2022-01-25 15:36:57,192 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:36:57,231 P27992 INFO Train loss: 0.384751
2022-01-25 15:36:57,231 P27992 INFO ************ Epoch=13 end ************
2022-01-25 15:39:16,929 P27992 INFO [Metrics] AUC: 0.954670 - logloss: 0.238676
2022-01-25 15:39:16,929 P27992 INFO Save best model: monitor(max): 0.954670
2022-01-25 15:39:16,940 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:39:16,999 P27992 INFO Train loss: 0.384751
2022-01-25 15:39:16,999 P27992 INFO ************ Epoch=14 end ************
2022-01-25 15:41:36,003 P27992 INFO [Metrics] AUC: 0.955131 - logloss: 0.236756
2022-01-25 15:41:36,004 P27992 INFO Save best model: monitor(max): 0.955131
2022-01-25 15:41:36,015 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:41:36,053 P27992 INFO Train loss: 0.384712
2022-01-25 15:41:36,053 P27992 INFO ************ Epoch=15 end ************
2022-01-25 15:43:56,817 P27992 INFO [Metrics] AUC: 0.955068 - logloss: 0.235997
2022-01-25 15:43:56,817 P27992 INFO Monitor(max) STOP: 0.955068 !
2022-01-25 15:43:56,817 P27992 INFO Reduce learning rate on plateau: 0.000100
2022-01-25 15:43:56,817 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:43:56,861 P27992 INFO Train loss: 0.387121
2022-01-25 15:43:56,861 P27992 INFO ************ Epoch=16 end ************
2022-01-25 15:46:17,435 P27992 INFO [Metrics] AUC: 0.966311 - logloss: 0.210763
2022-01-25 15:46:17,435 P27992 INFO Save best model: monitor(max): 0.966311
2022-01-25 15:46:17,446 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:46:17,487 P27992 INFO Train loss: 0.280757
2022-01-25 15:46:17,487 P27992 INFO ************ Epoch=17 end ************
2022-01-25 15:48:35,911 P27992 INFO [Metrics] AUC: 0.967552 - logloss: 0.222824
2022-01-25 15:48:35,912 P27992 INFO Save best model: monitor(max): 0.967552
2022-01-25 15:48:35,923 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:48:35,964 P27992 INFO Train loss: 0.189390
2022-01-25 15:48:35,964 P27992 INFO ************ Epoch=18 end ************
2022-01-25 15:50:56,356 P27992 INFO [Metrics] AUC: 0.965432 - logloss: 0.257573
2022-01-25 15:50:56,356 P27992 INFO Monitor(max) STOP: 0.965432 !
2022-01-25 15:50:56,356 P27992 INFO Reduce learning rate on plateau: 0.000010
2022-01-25 15:50:56,356 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:50:56,402 P27992 INFO Train loss: 0.141802
2022-01-25 15:50:56,402 P27992 INFO ************ Epoch=19 end ************
2022-01-25 15:53:17,165 P27992 INFO [Metrics] AUC: 0.965246 - logloss: 0.308727
2022-01-25 15:53:17,166 P27992 INFO Monitor(max) STOP: 0.965246 !
2022-01-25 15:53:17,166 P27992 INFO Reduce learning rate on plateau: 0.000001
2022-01-25 15:53:17,166 P27992 INFO Early stopping at epoch=20
2022-01-25 15:53:17,166 P27992 INFO --- 343/343 batches finished ---
2022-01-25 15:53:17,207 P27992 INFO Train loss: 0.106130
2022-01-25 15:53:17,207 P27992 INFO Training finished.
2022-01-25 15:53:17,207 P27992 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/AutoInt_movielenslatest_x1/movielenslatest_x1_cd32d937/AutoInt_movielenslatest_x1_025_f1ca3a7a.model
2022-01-25 15:53:22,925 P27992 INFO ****** Validation evaluation ******
2022-01-25 15:53:30,685 P27992 INFO [Metrics] AUC: 0.967552 - logloss: 0.222824
2022-01-25 15:53:30,723 P27992 INFO ******** Test evaluation ********
2022-01-25 15:53:30,723 P27992 INFO Loading data...
2022-01-25 15:53:30,724 P27992 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-25 15:53:30,728 P27992 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-25 15:53:30,728 P27992 INFO Loading test data done.
2022-01-25 15:53:34,780 P27992 INFO [Metrics] AUC: 0.967161 - logloss: 0.224678

```
