## LR_movielenslatest_x1

A hands-on guide to run the LR model on the Movielenslatest_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 503G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.1
  scipy: 1.2.2
  sklearn: 0.19.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.28.1
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [MovielensLatest_x1](https://github.com/reczoo/Datasets/tree/main/MovieLens/MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [LR](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_movielenslatest_x1_tuner_config_02](./LR_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_movielenslatest_x1
    nohup python run_expid.py --config ./LR_movielenslatest_x1_tuner_config_02 --expid LR_movielenslatest_x1_001_7530c4ec --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.934189 | 0.320077  |


### Logs
```python
2022-01-25 13:57:02,454 P177236 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Movielens/",
    "dataset_id": "movielenslatest_x1_cd32d937",
    "debug": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "LR",
    "model_id": "LR_movielenslatest_x1_001_7530c4ec",
    "model_root": "./Movielens/LR_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
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
2022-01-25 13:57:02,455 P177236 INFO Set up feature encoder...
2022-01-25 13:57:02,455 P177236 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-25 13:57:02,455 P177236 INFO Loading data...
2022-01-25 13:57:02,458 P177236 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-25 13:57:02,484 P177236 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-25 13:57:02,493 P177236 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-25 13:57:02,493 P177236 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-25 13:57:02,493 P177236 INFO Loading train data done.
2022-01-25 13:57:05,450 P177236 INFO Total number of parameters: 90240.
2022-01-25 13:57:05,451 P177236 INFO Start training: 343 batches/epoch
2022-01-25 13:57:05,451 P177236 INFO ************ Epoch=1 start ************
2022-01-25 13:57:09,881 P177236 INFO [Metrics] AUC: 0.870551 - logloss: 0.601114
2022-01-25 13:57:09,882 P177236 INFO Save best model: monitor(max): 0.870551
2022-01-25 13:57:09,883 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:09,927 P177236 INFO Train loss: 0.639501
2022-01-25 13:57:09,927 P177236 INFO ************ Epoch=1 end ************
2022-01-25 13:57:14,345 P177236 INFO [Metrics] AUC: 0.898336 - logloss: 0.557670
2022-01-25 13:57:14,346 P177236 INFO Save best model: monitor(max): 0.898336
2022-01-25 13:57:14,347 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:14,388 P177236 INFO Train loss: 0.576933
2022-01-25 13:57:14,388 P177236 INFO ************ Epoch=2 end ************
2022-01-25 13:57:18,938 P177236 INFO [Metrics] AUC: 0.913738 - logloss: 0.524822
2022-01-25 13:57:18,939 P177236 INFO Save best model: monitor(max): 0.913738
2022-01-25 13:57:18,940 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:18,985 P177236 INFO Train loss: 0.539774
2022-01-25 13:57:18,985 P177236 INFO ************ Epoch=3 end ************
2022-01-25 13:57:23,470 P177236 INFO [Metrics] AUC: 0.920917 - logloss: 0.497322
2022-01-25 13:57:23,471 P177236 INFO Save best model: monitor(max): 0.920917
2022-01-25 13:57:23,473 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:23,515 P177236 INFO Train loss: 0.510510
2022-01-25 13:57:23,515 P177236 INFO ************ Epoch=4 end ************
2022-01-25 13:57:28,141 P177236 INFO [Metrics] AUC: 0.924259 - logloss: 0.473991
2022-01-25 13:57:28,142 P177236 INFO Save best model: monitor(max): 0.924259
2022-01-25 13:57:28,144 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:28,189 P177236 INFO Train loss: 0.486424
2022-01-25 13:57:28,189 P177236 INFO ************ Epoch=5 end ************
2022-01-25 13:57:32,505 P177236 INFO [Metrics] AUC: 0.926168 - logloss: 0.454079
2022-01-25 13:57:32,506 P177236 INFO Save best model: monitor(max): 0.926168
2022-01-25 13:57:32,508 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:32,549 P177236 INFO Train loss: 0.466432
2022-01-25 13:57:32,549 P177236 INFO ************ Epoch=6 end ************
2022-01-25 13:57:37,115 P177236 INFO [Metrics] AUC: 0.927308 - logloss: 0.437018
2022-01-25 13:57:37,115 P177236 INFO Save best model: monitor(max): 0.927308
2022-01-25 13:57:37,117 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:37,157 P177236 INFO Train loss: 0.449755
2022-01-25 13:57:37,157 P177236 INFO ************ Epoch=7 end ************
2022-01-25 13:57:41,859 P177236 INFO [Metrics] AUC: 0.928262 - logloss: 0.422358
2022-01-25 13:57:41,860 P177236 INFO Save best model: monitor(max): 0.928262
2022-01-25 13:57:41,861 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:41,909 P177236 INFO Train loss: 0.435780
2022-01-25 13:57:41,909 P177236 INFO ************ Epoch=8 end ************
2022-01-25 13:57:46,279 P177236 INFO [Metrics] AUC: 0.928927 - logloss: 0.409718
2022-01-25 13:57:46,279 P177236 INFO Save best model: monitor(max): 0.928927
2022-01-25 13:57:46,281 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:46,320 P177236 INFO Train loss: 0.424024
2022-01-25 13:57:46,321 P177236 INFO ************ Epoch=9 end ************
2022-01-25 13:57:50,736 P177236 INFO [Metrics] AUC: 0.929521 - logloss: 0.398793
2022-01-25 13:57:50,736 P177236 INFO Save best model: monitor(max): 0.929521
2022-01-25 13:57:50,738 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:50,777 P177236 INFO Train loss: 0.414094
2022-01-25 13:57:50,778 P177236 INFO ************ Epoch=10 end ************
2022-01-25 13:57:55,139 P177236 INFO [Metrics] AUC: 0.930025 - logloss: 0.389328
2022-01-25 13:57:55,140 P177236 INFO Save best model: monitor(max): 0.930025
2022-01-25 13:57:55,142 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:55,181 P177236 INFO Train loss: 0.405680
2022-01-25 13:57:55,181 P177236 INFO ************ Epoch=11 end ************
2022-01-25 13:57:59,707 P177236 INFO [Metrics] AUC: 0.930484 - logloss: 0.381115
2022-01-25 13:57:59,708 P177236 INFO Save best model: monitor(max): 0.930484
2022-01-25 13:57:59,710 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:57:59,752 P177236 INFO Train loss: 0.398531
2022-01-25 13:57:59,753 P177236 INFO ************ Epoch=12 end ************
2022-01-25 13:58:04,307 P177236 INFO [Metrics] AUC: 0.930877 - logloss: 0.373971
2022-01-25 13:58:04,307 P177236 INFO Save best model: monitor(max): 0.930877
2022-01-25 13:58:04,309 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:04,351 P177236 INFO Train loss: 0.392438
2022-01-25 13:58:04,351 P177236 INFO ************ Epoch=13 end ************
2022-01-25 13:58:08,805 P177236 INFO [Metrics] AUC: 0.931220 - logloss: 0.367745
2022-01-25 13:58:08,805 P177236 INFO Save best model: monitor(max): 0.931220
2022-01-25 13:58:08,807 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:08,852 P177236 INFO Train loss: 0.387245
2022-01-25 13:58:08,852 P177236 INFO ************ Epoch=14 end ************
2022-01-25 13:58:13,282 P177236 INFO [Metrics] AUC: 0.931553 - logloss: 0.362309
2022-01-25 13:58:13,283 P177236 INFO Save best model: monitor(max): 0.931553
2022-01-25 13:58:13,284 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:13,325 P177236 INFO Train loss: 0.382808
2022-01-25 13:58:13,325 P177236 INFO ************ Epoch=15 end ************
2022-01-25 13:58:17,705 P177236 INFO [Metrics] AUC: 0.931814 - logloss: 0.357567
2022-01-25 13:58:17,706 P177236 INFO Save best model: monitor(max): 0.931814
2022-01-25 13:58:17,708 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:17,747 P177236 INFO Train loss: 0.379025
2022-01-25 13:58:17,747 P177236 INFO ************ Epoch=16 end ************
2022-01-25 13:58:21,991 P177236 INFO [Metrics] AUC: 0.932066 - logloss: 0.353417
2022-01-25 13:58:21,992 P177236 INFO Save best model: monitor(max): 0.932066
2022-01-25 13:58:21,993 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:22,034 P177236 INFO Train loss: 0.375794
2022-01-25 13:58:22,034 P177236 INFO ************ Epoch=17 end ************
2022-01-25 13:58:26,418 P177236 INFO [Metrics] AUC: 0.932266 - logloss: 0.349785
2022-01-25 13:58:26,419 P177236 INFO Save best model: monitor(max): 0.932266
2022-01-25 13:58:26,420 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:26,460 P177236 INFO Train loss: 0.373043
2022-01-25 13:58:26,460 P177236 INFO ************ Epoch=18 end ************
2022-01-25 13:58:30,927 P177236 INFO [Metrics] AUC: 0.932464 - logloss: 0.346604
2022-01-25 13:58:30,928 P177236 INFO Save best model: monitor(max): 0.932464
2022-01-25 13:58:30,929 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:30,970 P177236 INFO Train loss: 0.370701
2022-01-25 13:58:30,970 P177236 INFO ************ Epoch=19 end ************
2022-01-25 13:58:35,607 P177236 INFO [Metrics] AUC: 0.932627 - logloss: 0.343808
2022-01-25 13:58:35,609 P177236 INFO Save best model: monitor(max): 0.932627
2022-01-25 13:58:35,611 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:35,650 P177236 INFO Train loss: 0.368711
2022-01-25 13:58:35,651 P177236 INFO ************ Epoch=20 end ************
2022-01-25 13:58:40,094 P177236 INFO [Metrics] AUC: 0.932787 - logloss: 0.341345
2022-01-25 13:58:40,095 P177236 INFO Save best model: monitor(max): 0.932787
2022-01-25 13:58:40,096 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:40,137 P177236 INFO Train loss: 0.367025
2022-01-25 13:58:40,137 P177236 INFO ************ Epoch=21 end ************
2022-01-25 13:58:44,832 P177236 INFO [Metrics] AUC: 0.932914 - logloss: 0.339181
2022-01-25 13:58:44,833 P177236 INFO Save best model: monitor(max): 0.932914
2022-01-25 13:58:44,834 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:44,875 P177236 INFO Train loss: 0.365595
2022-01-25 13:58:44,875 P177236 INFO ************ Epoch=22 end ************
2022-01-25 13:58:49,383 P177236 INFO [Metrics] AUC: 0.933031 - logloss: 0.337278
2022-01-25 13:58:49,383 P177236 INFO Save best model: monitor(max): 0.933031
2022-01-25 13:58:49,385 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:49,431 P177236 INFO Train loss: 0.364384
2022-01-25 13:58:49,431 P177236 INFO ************ Epoch=23 end ************
2022-01-25 13:58:54,043 P177236 INFO [Metrics] AUC: 0.933133 - logloss: 0.335595
2022-01-25 13:58:54,044 P177236 INFO Save best model: monitor(max): 0.933133
2022-01-25 13:58:54,045 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:54,093 P177236 INFO Train loss: 0.363355
2022-01-25 13:58:54,093 P177236 INFO ************ Epoch=24 end ************
2022-01-25 13:58:58,868 P177236 INFO [Metrics] AUC: 0.933235 - logloss: 0.334098
2022-01-25 13:58:58,869 P177236 INFO Save best model: monitor(max): 0.933235
2022-01-25 13:58:58,871 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:58:58,916 P177236 INFO Train loss: 0.362489
2022-01-25 13:58:58,916 P177236 INFO ************ Epoch=25 end ************
2022-01-25 13:59:03,355 P177236 INFO [Metrics] AUC: 0.933328 - logloss: 0.332777
2022-01-25 13:59:03,356 P177236 INFO Save best model: monitor(max): 0.933328
2022-01-25 13:59:03,358 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:03,402 P177236 INFO Train loss: 0.361753
2022-01-25 13:59:03,403 P177236 INFO ************ Epoch=26 end ************
2022-01-25 13:59:07,815 P177236 INFO [Metrics] AUC: 0.933404 - logloss: 0.331601
2022-01-25 13:59:07,816 P177236 INFO Save best model: monitor(max): 0.933404
2022-01-25 13:59:07,817 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:07,858 P177236 INFO Train loss: 0.361132
2022-01-25 13:59:07,858 P177236 INFO ************ Epoch=27 end ************
2022-01-25 13:59:12,425 P177236 INFO [Metrics] AUC: 0.933489 - logloss: 0.330551
2022-01-25 13:59:12,425 P177236 INFO Save best model: monitor(max): 0.933489
2022-01-25 13:59:12,427 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:12,472 P177236 INFO Train loss: 0.360600
2022-01-25 13:59:12,472 P177236 INFO ************ Epoch=28 end ************
2022-01-25 13:59:16,957 P177236 INFO [Metrics] AUC: 0.933564 - logloss: 0.329613
2022-01-25 13:59:16,958 P177236 INFO Save best model: monitor(max): 0.933564
2022-01-25 13:59:16,959 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:17,003 P177236 INFO Train loss: 0.360148
2022-01-25 13:59:17,003 P177236 INFO ************ Epoch=29 end ************
2022-01-25 13:59:21,535 P177236 INFO [Metrics] AUC: 0.933635 - logloss: 0.328769
2022-01-25 13:59:21,535 P177236 INFO Save best model: monitor(max): 0.933635
2022-01-25 13:59:21,537 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:21,584 P177236 INFO Train loss: 0.359762
2022-01-25 13:59:21,584 P177236 INFO ************ Epoch=30 end ************
2022-01-25 13:59:25,766 P177236 INFO [Metrics] AUC: 0.933690 - logloss: 0.328020
2022-01-25 13:59:25,766 P177236 INFO Save best model: monitor(max): 0.933690
2022-01-25 13:59:25,768 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:25,810 P177236 INFO Train loss: 0.359430
2022-01-25 13:59:25,810 P177236 INFO ************ Epoch=31 end ************
2022-01-25 13:59:30,012 P177236 INFO [Metrics] AUC: 0.933769 - logloss: 0.327331
2022-01-25 13:59:30,012 P177236 INFO Save best model: monitor(max): 0.933769
2022-01-25 13:59:30,014 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:30,056 P177236 INFO Train loss: 0.359146
2022-01-25 13:59:30,056 P177236 INFO ************ Epoch=32 end ************
2022-01-25 13:59:34,411 P177236 INFO [Metrics] AUC: 0.933824 - logloss: 0.326712
2022-01-25 13:59:34,411 P177236 INFO Save best model: monitor(max): 0.933824
2022-01-25 13:59:34,413 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:34,459 P177236 INFO Train loss: 0.358903
2022-01-25 13:59:34,459 P177236 INFO ************ Epoch=33 end ************
2022-01-25 13:59:39,118 P177236 INFO [Metrics] AUC: 0.933883 - logloss: 0.326151
2022-01-25 13:59:39,119 P177236 INFO Save best model: monitor(max): 0.933883
2022-01-25 13:59:39,120 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:39,160 P177236 INFO Train loss: 0.358689
2022-01-25 13:59:39,160 P177236 INFO ************ Epoch=34 end ************
2022-01-25 13:59:43,587 P177236 INFO [Metrics] AUC: 0.933929 - logloss: 0.325644
2022-01-25 13:59:43,587 P177236 INFO Save best model: monitor(max): 0.933929
2022-01-25 13:59:43,589 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:43,635 P177236 INFO Train loss: 0.358504
2022-01-25 13:59:43,635 P177236 INFO ************ Epoch=35 end ************
2022-01-25 13:59:48,049 P177236 INFO [Metrics] AUC: 0.933979 - logloss: 0.325178
2022-01-25 13:59:48,050 P177236 INFO Save best model: monitor(max): 0.933979
2022-01-25 13:59:48,051 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:48,096 P177236 INFO Train loss: 0.358346
2022-01-25 13:59:48,096 P177236 INFO ************ Epoch=36 end ************
2022-01-25 13:59:52,745 P177236 INFO [Metrics] AUC: 0.934025 - logloss: 0.324748
2022-01-25 13:59:52,745 P177236 INFO Save best model: monitor(max): 0.934025
2022-01-25 13:59:52,747 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:52,791 P177236 INFO Train loss: 0.358207
2022-01-25 13:59:52,792 P177236 INFO ************ Epoch=37 end ************
2022-01-25 13:59:57,477 P177236 INFO [Metrics] AUC: 0.934068 - logloss: 0.324359
2022-01-25 13:59:57,477 P177236 INFO Save best model: monitor(max): 0.934068
2022-01-25 13:59:57,479 P177236 INFO --- 343/343 batches finished ---
2022-01-25 13:59:57,523 P177236 INFO Train loss: 0.358086
2022-01-25 13:59:57,523 P177236 INFO ************ Epoch=38 end ************
2022-01-25 14:00:01,887 P177236 INFO [Metrics] AUC: 0.934115 - logloss: 0.324003
2022-01-25 14:00:01,887 P177236 INFO Save best model: monitor(max): 0.934115
2022-01-25 14:00:01,889 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:01,934 P177236 INFO Train loss: 0.357980
2022-01-25 14:00:01,934 P177236 INFO ************ Epoch=39 end ************
2022-01-25 14:00:06,541 P177236 INFO [Metrics] AUC: 0.934158 - logloss: 0.323667
2022-01-25 14:00:06,542 P177236 INFO Save best model: monitor(max): 0.934158
2022-01-25 14:00:06,543 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:06,586 P177236 INFO Train loss: 0.357887
2022-01-25 14:00:06,586 P177236 INFO ************ Epoch=40 end ************
2022-01-25 14:00:10,989 P177236 INFO [Metrics] AUC: 0.934199 - logloss: 0.323374
2022-01-25 14:00:10,990 P177236 INFO Save best model: monitor(max): 0.934199
2022-01-25 14:00:10,992 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:11,034 P177236 INFO Train loss: 0.357807
2022-01-25 14:00:11,035 P177236 INFO ************ Epoch=41 end ************
2022-01-25 14:00:15,389 P177236 INFO [Metrics] AUC: 0.934246 - logloss: 0.323090
2022-01-25 14:00:15,389 P177236 INFO Save best model: monitor(max): 0.934246
2022-01-25 14:00:15,391 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:15,433 P177236 INFO Train loss: 0.357736
2022-01-25 14:00:15,433 P177236 INFO ************ Epoch=42 end ************
2022-01-25 14:00:19,995 P177236 INFO [Metrics] AUC: 0.934273 - logloss: 0.322835
2022-01-25 14:00:19,996 P177236 INFO Save best model: monitor(max): 0.934273
2022-01-25 14:00:19,997 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:20,043 P177236 INFO Train loss: 0.357672
2022-01-25 14:00:20,043 P177236 INFO ************ Epoch=43 end ************
2022-01-25 14:00:24,544 P177236 INFO [Metrics] AUC: 0.934303 - logloss: 0.322598
2022-01-25 14:00:24,544 P177236 INFO Save best model: monitor(max): 0.934303
2022-01-25 14:00:24,546 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:24,587 P177236 INFO Train loss: 0.357619
2022-01-25 14:00:24,587 P177236 INFO ************ Epoch=44 end ************
2022-01-25 14:00:29,088 P177236 INFO [Metrics] AUC: 0.934329 - logloss: 0.322376
2022-01-25 14:00:29,088 P177236 INFO Save best model: monitor(max): 0.934329
2022-01-25 14:00:29,090 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:29,135 P177236 INFO Train loss: 0.357574
2022-01-25 14:00:29,135 P177236 INFO ************ Epoch=45 end ************
2022-01-25 14:00:33,708 P177236 INFO [Metrics] AUC: 0.934368 - logloss: 0.322177
2022-01-25 14:00:33,708 P177236 INFO Save best model: monitor(max): 0.934368
2022-01-25 14:00:33,710 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:33,750 P177236 INFO Train loss: 0.357528
2022-01-25 14:00:33,750 P177236 INFO ************ Epoch=46 end ************
2022-01-25 14:00:38,176 P177236 INFO [Metrics] AUC: 0.934389 - logloss: 0.321985
2022-01-25 14:00:38,176 P177236 INFO Save best model: monitor(max): 0.934389
2022-01-25 14:00:38,180 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:38,224 P177236 INFO Train loss: 0.357493
2022-01-25 14:00:38,224 P177236 INFO ************ Epoch=47 end ************
2022-01-25 14:00:42,775 P177236 INFO [Metrics] AUC: 0.934416 - logloss: 0.321808
2022-01-25 14:00:42,776 P177236 INFO Save best model: monitor(max): 0.934416
2022-01-25 14:00:42,777 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:42,824 P177236 INFO Train loss: 0.357460
2022-01-25 14:00:42,824 P177236 INFO ************ Epoch=48 end ************
2022-01-25 14:00:47,199 P177236 INFO [Metrics] AUC: 0.934445 - logloss: 0.321652
2022-01-25 14:00:47,201 P177236 INFO Save best model: monitor(max): 0.934445
2022-01-25 14:00:47,203 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:47,246 P177236 INFO Train loss: 0.357431
2022-01-25 14:00:47,246 P177236 INFO ************ Epoch=49 end ************
2022-01-25 14:00:51,636 P177236 INFO [Metrics] AUC: 0.934477 - logloss: 0.321498
2022-01-25 14:00:51,636 P177236 INFO Save best model: monitor(max): 0.934477
2022-01-25 14:00:51,638 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:51,692 P177236 INFO Train loss: 0.357406
2022-01-25 14:00:51,692 P177236 INFO ************ Epoch=50 end ************
2022-01-25 14:00:56,183 P177236 INFO [Metrics] AUC: 0.934501 - logloss: 0.321357
2022-01-25 14:00:56,183 P177236 INFO Save best model: monitor(max): 0.934501
2022-01-25 14:00:56,185 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:00:56,228 P177236 INFO Train loss: 0.357384
2022-01-25 14:00:56,228 P177236 INFO ************ Epoch=51 end ************
2022-01-25 14:01:00,707 P177236 INFO [Metrics] AUC: 0.934513 - logloss: 0.321232
2022-01-25 14:01:00,708 P177236 INFO Save best model: monitor(max): 0.934513
2022-01-25 14:01:00,710 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:00,756 P177236 INFO Train loss: 0.357363
2022-01-25 14:01:00,756 P177236 INFO ************ Epoch=52 end ************
2022-01-25 14:01:05,507 P177236 INFO [Metrics] AUC: 0.934541 - logloss: 0.321111
2022-01-25 14:01:05,507 P177236 INFO Save best model: monitor(max): 0.934541
2022-01-25 14:01:05,509 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:05,551 P177236 INFO Train loss: 0.357345
2022-01-25 14:01:05,551 P177236 INFO ************ Epoch=53 end ************
2022-01-25 14:01:10,213 P177236 INFO [Metrics] AUC: 0.934555 - logloss: 0.320989
2022-01-25 14:01:10,213 P177236 INFO Save best model: monitor(max): 0.934555
2022-01-25 14:01:10,215 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:10,261 P177236 INFO Train loss: 0.357331
2022-01-25 14:01:10,262 P177236 INFO ************ Epoch=54 end ************
2022-01-25 14:01:14,798 P177236 INFO [Metrics] AUC: 0.934570 - logloss: 0.320885
2022-01-25 14:01:14,798 P177236 INFO Save best model: monitor(max): 0.934570
2022-01-25 14:01:14,800 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:14,854 P177236 INFO Train loss: 0.357317
2022-01-25 14:01:14,854 P177236 INFO ************ Epoch=55 end ************
2022-01-25 14:01:19,633 P177236 INFO [Metrics] AUC: 0.934595 - logloss: 0.320795
2022-01-25 14:01:19,634 P177236 INFO Save best model: monitor(max): 0.934595
2022-01-25 14:01:19,635 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:19,679 P177236 INFO Train loss: 0.357306
2022-01-25 14:01:19,679 P177236 INFO ************ Epoch=56 end ************
2022-01-25 14:01:24,218 P177236 INFO [Metrics] AUC: 0.934609 - logloss: 0.320697
2022-01-25 14:01:24,219 P177236 INFO Save best model: monitor(max): 0.934609
2022-01-25 14:01:24,221 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:24,267 P177236 INFO Train loss: 0.357295
2022-01-25 14:01:24,267 P177236 INFO ************ Epoch=57 end ************
2022-01-25 14:01:28,858 P177236 INFO [Metrics] AUC: 0.934620 - logloss: 0.320615
2022-01-25 14:01:28,859 P177236 INFO Save best model: monitor(max): 0.934620
2022-01-25 14:01:28,861 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:28,903 P177236 INFO Train loss: 0.357285
2022-01-25 14:01:28,904 P177236 INFO ************ Epoch=58 end ************
2022-01-25 14:01:33,637 P177236 INFO [Metrics] AUC: 0.934633 - logloss: 0.320540
2022-01-25 14:01:33,638 P177236 INFO Save best model: monitor(max): 0.934633
2022-01-25 14:01:33,639 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:33,685 P177236 INFO Train loss: 0.357277
2022-01-25 14:01:33,685 P177236 INFO ************ Epoch=59 end ************
2022-01-25 14:01:38,181 P177236 INFO [Metrics] AUC: 0.934649 - logloss: 0.320465
2022-01-25 14:01:38,182 P177236 INFO Save best model: monitor(max): 0.934649
2022-01-25 14:01:38,183 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:38,228 P177236 INFO Train loss: 0.357271
2022-01-25 14:01:38,228 P177236 INFO ************ Epoch=60 end ************
2022-01-25 14:01:42,792 P177236 INFO [Metrics] AUC: 0.934656 - logloss: 0.320389
2022-01-25 14:01:42,793 P177236 INFO Save best model: monitor(max): 0.934656
2022-01-25 14:01:42,794 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:42,838 P177236 INFO Train loss: 0.357263
2022-01-25 14:01:42,838 P177236 INFO ************ Epoch=61 end ************
2022-01-25 14:01:47,248 P177236 INFO [Metrics] AUC: 0.934671 - logloss: 0.320333
2022-01-25 14:01:47,249 P177236 INFO Save best model: monitor(max): 0.934671
2022-01-25 14:01:47,251 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:47,291 P177236 INFO Train loss: 0.357257
2022-01-25 14:01:47,292 P177236 INFO ************ Epoch=62 end ************
2022-01-25 14:01:51,584 P177236 INFO [Metrics] AUC: 0.934693 - logloss: 0.320267
2022-01-25 14:01:51,584 P177236 INFO Save best model: monitor(max): 0.934693
2022-01-25 14:01:51,586 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:51,632 P177236 INFO Train loss: 0.357251
2022-01-25 14:01:51,632 P177236 INFO ************ Epoch=63 end ************
2022-01-25 14:01:56,490 P177236 INFO [Metrics] AUC: 0.934697 - logloss: 0.320211
2022-01-25 14:01:56,490 P177236 INFO Save best model: monitor(max): 0.934697
2022-01-25 14:01:56,491 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:01:56,534 P177236 INFO Train loss: 0.357246
2022-01-25 14:01:56,534 P177236 INFO ************ Epoch=64 end ************
2022-01-25 14:02:01,302 P177236 INFO [Metrics] AUC: 0.934703 - logloss: 0.320163
2022-01-25 14:02:01,303 P177236 INFO Save best model: monitor(max): 0.934703
2022-01-25 14:02:01,305 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:01,365 P177236 INFO Train loss: 0.357244
2022-01-25 14:02:01,365 P177236 INFO ************ Epoch=65 end ************
2022-01-25 14:02:06,164 P177236 INFO [Metrics] AUC: 0.934719 - logloss: 0.320110
2022-01-25 14:02:06,165 P177236 INFO Save best model: monitor(max): 0.934719
2022-01-25 14:02:06,166 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:06,212 P177236 INFO Train loss: 0.357238
2022-01-25 14:02:06,212 P177236 INFO ************ Epoch=66 end ************
2022-01-25 14:02:10,956 P177236 INFO [Metrics] AUC: 0.934721 - logloss: 0.320072
2022-01-25 14:02:10,957 P177236 INFO Save best model: monitor(max): 0.934721
2022-01-25 14:02:10,958 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:11,003 P177236 INFO Train loss: 0.357237
2022-01-25 14:02:11,003 P177236 INFO ************ Epoch=67 end ************
2022-01-25 14:02:15,522 P177236 INFO [Metrics] AUC: 0.934730 - logloss: 0.320039
2022-01-25 14:02:15,523 P177236 INFO Save best model: monitor(max): 0.934730
2022-01-25 14:02:15,525 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:15,571 P177236 INFO Train loss: 0.357232
2022-01-25 14:02:15,572 P177236 INFO ************ Epoch=68 end ************
2022-01-25 14:02:20,066 P177236 INFO [Metrics] AUC: 0.934736 - logloss: 0.319986
2022-01-25 14:02:20,066 P177236 INFO Save best model: monitor(max): 0.934736
2022-01-25 14:02:20,068 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:20,112 P177236 INFO Train loss: 0.357229
2022-01-25 14:02:20,112 P177236 INFO ************ Epoch=69 end ************
2022-01-25 14:02:24,489 P177236 INFO [Metrics] AUC: 0.934743 - logloss: 0.319944
2022-01-25 14:02:24,490 P177236 INFO Save best model: monitor(max): 0.934743
2022-01-25 14:02:24,492 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:24,531 P177236 INFO Train loss: 0.357228
2022-01-25 14:02:24,531 P177236 INFO ************ Epoch=70 end ************
2022-01-25 14:02:29,111 P177236 INFO [Metrics] AUC: 0.934744 - logloss: 0.319914
2022-01-25 14:02:29,111 P177236 INFO Monitor(max) STOP: 0.934744 !
2022-01-25 14:02:29,112 P177236 INFO Reduce learning rate on plateau: 0.000100
2022-01-25 14:02:29,112 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:29,151 P177236 INFO Train loss: 0.357224
2022-01-25 14:02:29,151 P177236 INFO ************ Epoch=71 end ************
2022-01-25 14:02:33,742 P177236 INFO [Metrics] AUC: 0.934747 - logloss: 0.319912
2022-01-25 14:02:33,743 P177236 INFO Save best model: monitor(max): 0.934747
2022-01-25 14:02:33,745 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:33,786 P177236 INFO Train loss: 0.356531
2022-01-25 14:02:33,786 P177236 INFO ************ Epoch=72 end ************
2022-01-25 14:02:38,317 P177236 INFO [Metrics] AUC: 0.934748 - logloss: 0.319909
2022-01-25 14:02:38,317 P177236 INFO Save best model: monitor(max): 0.934748
2022-01-25 14:02:38,319 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:38,363 P177236 INFO Train loss: 0.356531
2022-01-25 14:02:38,363 P177236 INFO ************ Epoch=73 end ************
2022-01-25 14:02:42,997 P177236 INFO [Metrics] AUC: 0.934750 - logloss: 0.319906
2022-01-25 14:02:42,998 P177236 INFO Save best model: monitor(max): 0.934750
2022-01-25 14:02:43,000 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:43,041 P177236 INFO Train loss: 0.356530
2022-01-25 14:02:43,041 P177236 INFO ************ Epoch=74 end ************
2022-01-25 14:02:47,500 P177236 INFO [Metrics] AUC: 0.934752 - logloss: 0.319903
2022-01-25 14:02:47,501 P177236 INFO Save best model: monitor(max): 0.934752
2022-01-25 14:02:47,503 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:47,548 P177236 INFO Train loss: 0.356530
2022-01-25 14:02:47,549 P177236 INFO ************ Epoch=75 end ************
2022-01-25 14:02:51,923 P177236 INFO [Metrics] AUC: 0.934752 - logloss: 0.319899
2022-01-25 14:02:51,924 P177236 INFO Monitor(max) STOP: 0.934752 !
2022-01-25 14:02:51,924 P177236 INFO Reduce learning rate on plateau: 0.000010
2022-01-25 14:02:51,924 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:51,966 P177236 INFO Train loss: 0.356530
2022-01-25 14:02:51,967 P177236 INFO ************ Epoch=76 end ************
2022-01-25 14:02:56,627 P177236 INFO [Metrics] AUC: 0.934752 - logloss: 0.319899
2022-01-25 14:02:56,627 P177236 INFO Monitor(max) STOP: 0.934752 !
2022-01-25 14:02:56,627 P177236 INFO Reduce learning rate on plateau: 0.000001
2022-01-25 14:02:56,627 P177236 INFO Early stopping at epoch=77
2022-01-25 14:02:56,628 P177236 INFO --- 343/343 batches finished ---
2022-01-25 14:02:56,671 P177236 INFO Train loss: 0.356459
2022-01-25 14:02:56,671 P177236 INFO Training finished.
2022-01-25 14:02:56,671 P177236 INFO Load best model: /home/ma-user/work/FuxiCTRv1.1/benchmarks/Movielens/LR_movielenslatest_x1/movielenslatest_x1_cd32d937/LR_movielenslatest_x1_001_7530c4ec.model
2022-01-25 14:02:56,673 P177236 INFO ****** Validation evaluation ******
2022-01-25 14:02:57,936 P177236 INFO [Metrics] AUC: 0.934752 - logloss: 0.319903
2022-01-25 14:02:57,984 P177236 INFO ******** Test evaluation ********
2022-01-25 14:02:57,985 P177236 INFO Loading data...
2022-01-25 14:02:57,985 P177236 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-25 14:02:57,990 P177236 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-25 14:02:57,990 P177236 INFO Loading test data done.
2022-01-25 14:02:58,659 P177236 INFO [Metrics] AUC: 0.934189 - logloss: 0.320077

```
