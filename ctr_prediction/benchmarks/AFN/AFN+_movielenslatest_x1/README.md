## AFN+_movielenslatest_x1

A hands-on guide to run the AFN model on the Movielenslatest_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN+_movielenslatest_x1_tuner_config_01](./AFN+_movielenslatest_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN+_movielenslatest_x1
    nohup python run_expid.py --config ./AFN+_movielenslatest_x1_tuner_config_01 --expid AFN_movielenslatest_x1_003_cc30c477 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.964208 | 0.303013  |
| 2 | 0.963902 | 0.318095  |
| 3 | 0.962763 | 0.290086  |
| 4 | 0.964087 | 0.306997  |
| 5 | 0.963066 | 0.294733  |
| | | | 
| Avg | 0.963605 | 0.302585 |
| Std | &#177;0.00058028 | &#177;0.00977842 |


### Logs
```python
2022-01-31 11:09:41,148 P46946 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.4",
    "afn_hidden_units": "[200]",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0.5",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "800",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_movielenslatest_x1_003_cc30c477",
    "model_root": "./Movielens/AFN_movielenslatest_x1/",
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
    "valid_data": "../data/Movielens/MovielensLatest_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-31 11:09:41,167 P46946 INFO Set up feature encoder...
2022-01-31 11:09:41,167 P46946 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-31 11:09:41,168 P46946 INFO Loading data...
2022-01-31 11:09:41,171 P46946 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-31 11:09:41,254 P46946 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-31 11:09:41,272 P46946 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-31 11:09:41,272 P46946 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-31 11:09:41,273 P46946 INFO Loading train data done.
2022-01-31 11:09:52,717 P46946 INFO Total number of parameters: 3742791.
2022-01-31 11:09:52,721 P46946 INFO Start training: 343 batches/epoch
2022-01-31 11:09:52,722 P46946 INFO ************ Epoch=1 start ************
2022-01-31 11:10:34,451 P46946 INFO [Metrics] AUC: 0.933560 - logloss: 0.294246
2022-01-31 11:10:34,451 P46946 INFO Save best model: monitor(max): 0.933560
2022-01-31 11:10:34,489 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:10:34,592 P46946 INFO Train loss: 0.398314
2022-01-31 11:10:34,592 P46946 INFO ************ Epoch=1 end ************
2022-01-31 11:11:16,401 P46946 INFO [Metrics] AUC: 0.940702 - logloss: 0.279099
2022-01-31 11:11:16,411 P46946 INFO Save best model: monitor(max): 0.940702
2022-01-31 11:11:16,498 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:11:16,660 P46946 INFO Train loss: 0.355998
2022-01-31 11:11:16,660 P46946 INFO ************ Epoch=2 end ************
2022-01-31 11:11:59,105 P46946 INFO [Metrics] AUC: 0.948216 - logloss: 0.260481
2022-01-31 11:11:59,106 P46946 INFO Save best model: monitor(max): 0.948216
2022-01-31 11:11:59,188 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:11:59,354 P46946 INFO Train loss: 0.354279
2022-01-31 11:11:59,354 P46946 INFO ************ Epoch=3 end ************
2022-01-31 11:12:41,070 P46946 INFO [Metrics] AUC: 0.951603 - logloss: 0.251751
2022-01-31 11:12:41,077 P46946 INFO Save best model: monitor(max): 0.951603
2022-01-31 11:12:41,130 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:12:41,231 P46946 INFO Train loss: 0.345981
2022-01-31 11:12:41,232 P46946 INFO ************ Epoch=4 end ************
2022-01-31 11:13:23,301 P46946 INFO [Metrics] AUC: 0.953311 - logloss: 0.248506
2022-01-31 11:13:23,301 P46946 INFO Save best model: monitor(max): 0.953311
2022-01-31 11:13:23,373 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:13:23,552 P46946 INFO Train loss: 0.341915
2022-01-31 11:13:23,552 P46946 INFO ************ Epoch=5 end ************
2022-01-31 11:14:05,889 P46946 INFO [Metrics] AUC: 0.954992 - logloss: 0.245611
2022-01-31 11:14:05,890 P46946 INFO Save best model: monitor(max): 0.954992
2022-01-31 11:14:05,961 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:14:06,145 P46946 INFO Train loss: 0.337735
2022-01-31 11:14:06,146 P46946 INFO ************ Epoch=6 end ************
2022-01-31 11:14:47,290 P46946 INFO [Metrics] AUC: 0.956404 - logloss: 0.242309
2022-01-31 11:14:47,291 P46946 INFO Save best model: monitor(max): 0.956404
2022-01-31 11:14:47,343 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:14:47,446 P46946 INFO Train loss: 0.333618
2022-01-31 11:14:47,446 P46946 INFO ************ Epoch=7 end ************
2022-01-31 11:15:29,094 P46946 INFO [Metrics] AUC: 0.957560 - logloss: 0.238802
2022-01-31 11:15:29,095 P46946 INFO Save best model: monitor(max): 0.957560
2022-01-31 11:15:29,171 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:15:29,357 P46946 INFO Train loss: 0.328551
2022-01-31 11:15:29,357 P46946 INFO ************ Epoch=8 end ************
2022-01-31 11:16:16,256 P46946 INFO [Metrics] AUC: 0.958903 - logloss: 0.237413
2022-01-31 11:16:16,257 P46946 INFO Save best model: monitor(max): 0.958903
2022-01-31 11:16:16,344 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:16:16,619 P46946 INFO Train loss: 0.322232
2022-01-31 11:16:16,620 P46946 INFO ************ Epoch=9 end ************
2022-01-31 11:17:04,300 P46946 INFO [Metrics] AUC: 0.959464 - logloss: 0.236080
2022-01-31 11:17:04,301 P46946 INFO Save best model: monitor(max): 0.959464
2022-01-31 11:17:04,402 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:17:04,702 P46946 INFO Train loss: 0.315644
2022-01-31 11:17:04,702 P46946 INFO ************ Epoch=10 end ************
2022-01-31 11:17:49,647 P46946 INFO [Metrics] AUC: 0.959987 - logloss: 0.234829
2022-01-31 11:17:49,653 P46946 INFO Save best model: monitor(max): 0.959987
2022-01-31 11:17:49,728 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:17:49,983 P46946 INFO Train loss: 0.308503
2022-01-31 11:17:49,984 P46946 INFO ************ Epoch=11 end ************
2022-01-31 11:18:33,805 P46946 INFO [Metrics] AUC: 0.960609 - logloss: 0.234423
2022-01-31 11:18:33,809 P46946 INFO Save best model: monitor(max): 0.960609
2022-01-31 11:18:33,881 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:18:34,052 P46946 INFO Train loss: 0.302716
2022-01-31 11:18:34,052 P46946 INFO ************ Epoch=12 end ************
2022-01-31 11:19:16,636 P46946 INFO [Metrics] AUC: 0.961140 - logloss: 0.235659
2022-01-31 11:19:16,637 P46946 INFO Save best model: monitor(max): 0.961140
2022-01-31 11:19:16,690 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:19:16,866 P46946 INFO Train loss: 0.296989
2022-01-31 11:19:16,867 P46946 INFO ************ Epoch=13 end ************
2022-01-31 11:19:57,335 P46946 INFO [Metrics] AUC: 0.961360 - logloss: 0.233485
2022-01-31 11:19:57,336 P46946 INFO Save best model: monitor(max): 0.961360
2022-01-31 11:19:57,392 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:19:57,527 P46946 INFO Train loss: 0.291879
2022-01-31 11:19:57,527 P46946 INFO ************ Epoch=14 end ************
2022-01-31 11:20:36,970 P46946 INFO [Metrics] AUC: 0.961569 - logloss: 0.233516
2022-01-31 11:20:36,971 P46946 INFO Save best model: monitor(max): 0.961569
2022-01-31 11:20:37,020 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:20:37,126 P46946 INFO Train loss: 0.286860
2022-01-31 11:20:37,126 P46946 INFO ************ Epoch=15 end ************
2022-01-31 11:21:15,967 P46946 INFO [Metrics] AUC: 0.961790 - logloss: 0.233591
2022-01-31 11:21:15,968 P46946 INFO Save best model: monitor(max): 0.961790
2022-01-31 11:21:15,997 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:21:16,084 P46946 INFO Train loss: 0.282833
2022-01-31 11:21:16,085 P46946 INFO ************ Epoch=16 end ************
2022-01-31 11:21:55,114 P46946 INFO [Metrics] AUC: 0.962338 - logloss: 0.234178
2022-01-31 11:21:55,115 P46946 INFO Save best model: monitor(max): 0.962338
2022-01-31 11:21:55,136 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:21:55,242 P46946 INFO Train loss: 0.279675
2022-01-31 11:21:55,242 P46946 INFO ************ Epoch=17 end ************
2022-01-31 11:22:30,293 P46946 INFO [Metrics] AUC: 0.962285 - logloss: 0.233613
2022-01-31 11:22:30,294 P46946 INFO Monitor(max) STOP: 0.962285 !
2022-01-31 11:22:30,294 P46946 INFO Reduce learning rate on plateau: 0.000100
2022-01-31 11:22:30,294 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:22:30,441 P46946 INFO Train loss: 0.275966
2022-01-31 11:22:30,441 P46946 INFO ************ Epoch=18 end ************
2022-01-31 11:23:04,084 P46946 INFO [Metrics] AUC: 0.964386 - logloss: 0.269489
2022-01-31 11:23:04,084 P46946 INFO Save best model: monitor(max): 0.964386
2022-01-31 11:23:04,180 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:23:04,407 P46946 INFO Train loss: 0.225758
2022-01-31 11:23:04,408 P46946 INFO ************ Epoch=19 end ************
2022-01-31 11:23:37,367 P46946 INFO [Metrics] AUC: 0.964429 - logloss: 0.301879
2022-01-31 11:23:37,375 P46946 INFO Save best model: monitor(max): 0.964429
2022-01-31 11:23:37,426 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:23:37,592 P46946 INFO Train loss: 0.184881
2022-01-31 11:23:37,592 P46946 INFO ************ Epoch=20 end ************
2022-01-31 11:24:10,789 P46946 INFO [Metrics] AUC: 0.964157 - logloss: 0.327650
2022-01-31 11:24:10,790 P46946 INFO Monitor(max) STOP: 0.964157 !
2022-01-31 11:24:10,790 P46946 INFO Reduce learning rate on plateau: 0.000010
2022-01-31 11:24:10,790 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:24:10,913 P46946 INFO Train loss: 0.157453
2022-01-31 11:24:10,913 P46946 INFO ************ Epoch=21 end ************
2022-01-31 11:24:49,625 P46946 INFO [Metrics] AUC: 0.964304 - logloss: 0.343298
2022-01-31 11:24:49,633 P46946 INFO Monitor(max) STOP: 0.964304 !
2022-01-31 11:24:49,633 P46946 INFO Reduce learning rate on plateau: 0.000001
2022-01-31 11:24:49,633 P46946 INFO Early stopping at epoch=22
2022-01-31 11:24:49,633 P46946 INFO --- 343/343 batches finished ---
2022-01-31 11:24:49,801 P46946 INFO Train loss: 0.139694
2022-01-31 11:24:49,801 P46946 INFO Training finished.
2022-01-31 11:24:49,802 P46946 INFO Load best model: /home/XXX/benchmarks/Movielens/AFN_movielenslatest_x1/movielenslatest_x1_cd32d937/AFN_movielenslatest_x1_003_cc30c477.model
2022-01-31 11:24:49,819 P46946 INFO ****** Validation evaluation ******
2022-01-31 11:24:56,559 P46946 INFO [Metrics] AUC: 0.964429 - logloss: 0.301879
2022-01-31 11:24:56,641 P46946 INFO ******** Test evaluation ********
2022-01-31 11:24:56,641 P46946 INFO Loading data...
2022-01-31 11:24:56,642 P46946 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-31 11:24:56,647 P46946 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-31 11:24:56,648 P46946 INFO Loading test data done.
2022-01-31 11:25:00,019 P46946 INFO [Metrics] AUC: 0.964208 - logloss: 0.303013

```
