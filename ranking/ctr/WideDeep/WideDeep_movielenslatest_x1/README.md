## WideDeep_movielenslatest_x1

A hands-on guide to run the WideDeep model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [WideDeep](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_movielenslatest_x1_tuner_config_01](./WideDeep_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_movielenslatest_x1
    nohup python run_expid.py --config ./WideDeep_movielenslatest_x1_tuner_config_01 --expid WideDeep_movielenslatest_x1_007_1b65cd45 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.968789 | 0.216132  |
| 2 | 0.968007 | 0.217266  |
| 3 | 0.968213 | 0.217506  |
| 4 | 0.968186 | 0.219195  |
| 5 | 0.968771 | 0.215594  |
| Avg | 0.968393 | 0.217139 |
| Std | &#177;0.00032371 | &#177;0.00124732 |


### Logs
```python
2022-02-07 17:16:53,834 P1731 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "WideDeep",
    "model_id": "WideDeep_movielenslatest_x1_007_1b65cd45",
    "model_root": "./Movielens/WideDeep_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
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
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-02-07 17:16:53,847 P1731 INFO Set up feature encoder...
2022-02-07 17:16:53,847 P1731 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-02-07 17:16:53,847 P1731 INFO Loading data...
2022-02-07 17:16:53,857 P1731 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-02-07 17:16:53,932 P1731 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-02-07 17:16:53,953 P1731 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-02-07 17:16:53,957 P1731 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-02-07 17:16:53,958 P1731 INFO Loading train data done.
2022-02-07 17:17:06,166 P1731 INFO Total number of parameters: 1328630.
2022-02-07 17:17:06,167 P1731 INFO Start training: 343 batches/epoch
2022-02-07 17:17:06,168 P1731 INFO ************ Epoch=1 start ************
2022-02-07 17:17:34,977 P1731 INFO [Metrics] AUC: 0.937398 - logloss: 0.285267
2022-02-07 17:17:34,986 P1731 INFO Save best model: monitor(max): 0.937398
2022-02-07 17:17:35,003 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:17:35,157 P1731 INFO Train loss: 0.381005
2022-02-07 17:17:35,157 P1731 INFO ************ Epoch=1 end ************
2022-02-07 17:18:03,002 P1731 INFO [Metrics] AUC: 0.947017 - logloss: 0.261755
2022-02-07 17:18:03,004 P1731 INFO Save best model: monitor(max): 0.947017
2022-02-07 17:18:03,042 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:18:03,161 P1731 INFO Train loss: 0.365650
2022-02-07 17:18:03,162 P1731 INFO ************ Epoch=2 end ************
2022-02-07 17:18:29,936 P1731 INFO [Metrics] AUC: 0.951037 - logloss: 0.253379
2022-02-07 17:18:29,936 P1731 INFO Save best model: monitor(max): 0.951037
2022-02-07 17:18:29,967 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:18:30,075 P1731 INFO Train loss: 0.368333
2022-02-07 17:18:30,075 P1731 INFO ************ Epoch=3 end ************
2022-02-07 17:18:57,458 P1731 INFO [Metrics] AUC: 0.951967 - logloss: 0.244816
2022-02-07 17:18:57,459 P1731 INFO Save best model: monitor(max): 0.951967
2022-02-07 17:18:57,489 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:18:57,640 P1731 INFO Train loss: 0.370959
2022-02-07 17:18:57,640 P1731 INFO ************ Epoch=4 end ************
2022-02-07 17:19:26,870 P1731 INFO [Metrics] AUC: 0.952698 - logloss: 0.245630
2022-02-07 17:19:26,871 P1731 INFO Save best model: monitor(max): 0.952698
2022-02-07 17:19:26,897 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:19:27,018 P1731 INFO Train loss: 0.374239
2022-02-07 17:19:27,019 P1731 INFO ************ Epoch=5 end ************
2022-02-07 17:19:51,275 P1731 INFO [Metrics] AUC: 0.954021 - logloss: 0.243898
2022-02-07 17:19:51,284 P1731 INFO Save best model: monitor(max): 0.954021
2022-02-07 17:19:51,308 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:19:51,492 P1731 INFO Train loss: 0.375670
2022-02-07 17:19:51,493 P1731 INFO ************ Epoch=6 end ************
2022-02-07 17:20:14,890 P1731 INFO [Metrics] AUC: 0.955197 - logloss: 0.236842
2022-02-07 17:20:14,891 P1731 INFO Save best model: monitor(max): 0.955197
2022-02-07 17:20:14,933 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:20:15,141 P1731 INFO Train loss: 0.377526
2022-02-07 17:20:15,141 P1731 INFO ************ Epoch=7 end ************
2022-02-07 17:20:47,070 P1731 INFO [Metrics] AUC: 0.955377 - logloss: 0.236294
2022-02-07 17:20:47,071 P1731 INFO Save best model: monitor(max): 0.955377
2022-02-07 17:20:47,099 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:20:47,388 P1731 INFO Train loss: 0.378962
2022-02-07 17:20:47,388 P1731 INFO ************ Epoch=8 end ************
2022-02-07 17:21:14,060 P1731 INFO [Metrics] AUC: 0.956254 - logloss: 0.235094
2022-02-07 17:21:14,068 P1731 INFO Save best model: monitor(max): 0.956254
2022-02-07 17:21:14,122 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:21:14,285 P1731 INFO Train loss: 0.378117
2022-02-07 17:21:14,286 P1731 INFO ************ Epoch=9 end ************
2022-02-07 17:21:40,544 P1731 INFO [Metrics] AUC: 0.956004 - logloss: 0.235710
2022-02-07 17:21:40,552 P1731 INFO Monitor(max) STOP: 0.956004 !
2022-02-07 17:21:40,552 P1731 INFO Reduce learning rate on plateau: 0.000100
2022-02-07 17:21:40,552 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:21:40,738 P1731 INFO Train loss: 0.378566
2022-02-07 17:21:40,738 P1731 INFO ************ Epoch=10 end ************
2022-02-07 17:22:10,032 P1731 INFO [Metrics] AUC: 0.968146 - logloss: 0.204386
2022-02-07 17:22:10,033 P1731 INFO Save best model: monitor(max): 0.968146
2022-02-07 17:22:10,046 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:22:10,229 P1731 INFO Train loss: 0.274977
2022-02-07 17:22:10,229 P1731 INFO ************ Epoch=11 end ************
2022-02-07 17:22:33,290 P1731 INFO [Metrics] AUC: 0.969056 - logloss: 0.214863
2022-02-07 17:22:33,291 P1731 INFO Save best model: monitor(max): 0.969056
2022-02-07 17:22:33,302 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:22:33,410 P1731 INFO Train loss: 0.183548
2022-02-07 17:22:33,410 P1731 INFO ************ Epoch=12 end ************
2022-02-07 17:22:55,667 P1731 INFO [Metrics] AUC: 0.967378 - logloss: 0.245376
2022-02-07 17:22:55,668 P1731 INFO Monitor(max) STOP: 0.967378 !
2022-02-07 17:22:55,669 P1731 INFO Reduce learning rate on plateau: 0.000010
2022-02-07 17:22:55,669 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:22:55,757 P1731 INFO Train loss: 0.136367
2022-02-07 17:22:55,758 P1731 INFO ************ Epoch=13 end ************
2022-02-07 17:23:17,984 P1731 INFO [Metrics] AUC: 0.967167 - logloss: 0.254152
2022-02-07 17:23:17,985 P1731 INFO Monitor(max) STOP: 0.967167 !
2022-02-07 17:23:17,985 P1731 INFO Reduce learning rate on plateau: 0.000001
2022-02-07 17:23:17,985 P1731 INFO Early stopping at epoch=14
2022-02-07 17:23:17,985 P1731 INFO --- 343/343 batches finished ---
2022-02-07 17:23:18,066 P1731 INFO Train loss: 0.103593
2022-02-07 17:23:18,067 P1731 INFO Training finished.
2022-02-07 17:23:18,067 P1731 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Movielens/WideDeep_movielenslatest_x1/movielenslatest_x1_cd32d937/WideDeep_movielenslatest_x1_007_1b65cd45.model
2022-02-07 17:23:18,105 P1731 INFO ****** Validation evaluation ******
2022-02-07 17:23:21,190 P1731 INFO [Metrics] AUC: 0.969056 - logloss: 0.214863
2022-02-07 17:23:21,280 P1731 INFO ******** Test evaluation ********
2022-02-07 17:23:21,280 P1731 INFO Loading data...
2022-02-07 17:23:21,281 P1731 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-02-07 17:23:21,287 P1731 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-02-07 17:23:21,287 P1731 INFO Loading test data done.
2022-02-07 17:23:22,879 P1731 INFO [Metrics] AUC: 0.968789 - logloss: 0.216132

```
