## AFN_movielenslatest_x1

A hands-on guide to run the AFN model on the Movielenslatest_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [MovielensLatest_x1](https://github.com/reczoo/Datasets/tree/main/MovieLens/MovielensLatest_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Movielens/MovielensLatest_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_movielenslatest_x1_tuner_config_02](./AFN_movielenslatest_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN_movielenslatest_x1
    nohup python run_expid.py --config ./AFN_movielenslatest_x1_tuner_config_02 --expid AFN_movielenslatest_x1_015_9d6fa874 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 27 runs:
| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.961073 | 0.296298  |
| 2 | 0.960930 | 0.267151  |
| 3 | 0.960886 | 0.307158  |
| 4 | 0.960614 | 0.276331  |
| 5 | 0.960298 | 0.278707  |
| 6 | 0.960120 | 0.263156  |
| 7 | 0.960081 | 0.298225  |
| 8 | 0.959445 | 0.299012  |
| 9 | 0.958548 | 0.268388  |
| 10 | 0.956918 | 0.317949  |
| 11 | 0.956788 | 0.306362  |
| 12 | 0.956542 | 0.312671  |
| 13 | 0.956504 | 0.302262  |
| 14 | 0.956364 | 0.306886  |
| 15 | 0.956235 | 0.323224  |
| 16 | 0.956139 | 0.285326  |
| 17 | 0.955893 | 0.357357  |
| 18 | 0.955851 | 0.344950  |
| 19 | 0.955360 | 0.383622  |
| 20 | 0.955341 | 0.384167  |
| 21 | 0.955171 | 0.305066  |
| 22 | 0.954615 | 0.372488  |
| 23 | 0.954612 | 0.359629  |
| 24 | 0.954218 | 0.359423  |
| 25 | 0.951321 | 0.319245  |
| 26 | 0.951189 | 0.271087  |
| 27 | 0.949524 | 0.273883  |
| Avg | 0.9566881481481482 | 0.31259344444444437 |
| Std | &#177;0.0030469135288232504 | &#177;0.036111582178465926 |


### Logs
```python
2022-01-30 00:38:07,270 P41243 INFO {
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
    "dnn_dropout": "0",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.001",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user_id', 'item_id', 'tag_id'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "800",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_movielenslatest_x1_015_9d6fa874",
    "model_root": "./Movielens/AFN_movielenslatest_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
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
2022-01-30 00:38:07,270 P41243 INFO Set up feature encoder...
2022-01-30 00:38:07,270 P41243 INFO Load feature_map from json: ../data/Movielens/movielenslatest_x1_cd32d937/feature_map.json
2022-01-30 00:38:07,271 P41243 INFO Loading data...
2022-01-30 00:38:07,273 P41243 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/train.h5
2022-01-30 00:38:07,303 P41243 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/valid.h5
2022-01-30 00:38:07,311 P41243 INFO Train samples: total/1404801, pos/467878, neg/936923, ratio/33.31%, blocks/1
2022-01-30 00:38:07,312 P41243 INFO Validation samples: total/401372, pos/134225, neg/267147, ratio/33.44%, blocks/1
2022-01-30 00:38:07,312 P41243 INFO Loading train data done.
2022-01-30 00:38:11,721 P41243 INFO Total number of parameters: 2506797.
2022-01-30 00:38:11,722 P41243 INFO Start training: 343 batches/epoch
2022-01-30 00:38:11,722 P41243 INFO ************ Epoch=1 start ************
2022-01-30 00:38:48,211 P41243 INFO [Metrics] AUC: 0.922777 - logloss: 0.312937
2022-01-30 00:38:48,211 P41243 INFO Save best model: monitor(max): 0.922777
2022-01-30 00:38:48,222 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:38:48,261 P41243 INFO Train loss: 0.438028
2022-01-30 00:38:48,261 P41243 INFO ************ Epoch=1 end ************
2022-01-30 00:39:25,069 P41243 INFO [Metrics] AUC: 0.935151 - logloss: 0.290454
2022-01-30 00:39:25,069 P41243 INFO Save best model: monitor(max): 0.935151
2022-01-30 00:39:25,083 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:39:25,128 P41243 INFO Train loss: 0.359103
2022-01-30 00:39:25,128 P41243 INFO ************ Epoch=2 end ************
2022-01-30 00:40:02,065 P41243 INFO [Metrics] AUC: 0.940676 - logloss: 0.277544
2022-01-30 00:40:02,066 P41243 INFO Save best model: monitor(max): 0.940676
2022-01-30 00:40:02,079 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:40:02,126 P41243 INFO Train loss: 0.351028
2022-01-30 00:40:02,127 P41243 INFO ************ Epoch=3 end ************
2022-01-30 00:40:38,949 P41243 INFO [Metrics] AUC: 0.943586 - logloss: 0.270905
2022-01-30 00:40:38,950 P41243 INFO Save best model: monitor(max): 0.943586
2022-01-30 00:40:38,963 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:40:39,001 P41243 INFO Train loss: 0.352679
2022-01-30 00:40:39,001 P41243 INFO ************ Epoch=4 end ************
2022-01-30 00:41:15,686 P41243 INFO [Metrics] AUC: 0.944989 - logloss: 0.267858
2022-01-30 00:41:15,687 P41243 INFO Save best model: monitor(max): 0.944989
2022-01-30 00:41:15,700 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:41:15,742 P41243 INFO Train loss: 0.354038
2022-01-30 00:41:15,742 P41243 INFO ************ Epoch=5 end ************
2022-01-30 00:41:52,507 P41243 INFO [Metrics] AUC: 0.947198 - logloss: 0.265365
2022-01-30 00:41:52,508 P41243 INFO Save best model: monitor(max): 0.947198
2022-01-30 00:41:52,521 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:41:52,563 P41243 INFO Train loss: 0.354768
2022-01-30 00:41:52,563 P41243 INFO ************ Epoch=6 end ************
2022-01-30 00:42:29,352 P41243 INFO [Metrics] AUC: 0.948121 - logloss: 0.261170
2022-01-30 00:42:29,353 P41243 INFO Save best model: monitor(max): 0.948121
2022-01-30 00:42:29,366 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:42:29,406 P41243 INFO Train loss: 0.353778
2022-01-30 00:42:29,406 P41243 INFO ************ Epoch=7 end ************
2022-01-30 00:43:06,469 P41243 INFO [Metrics] AUC: 0.950630 - logloss: 0.255236
2022-01-30 00:43:06,470 P41243 INFO Save best model: monitor(max): 0.950630
2022-01-30 00:43:06,483 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:43:06,522 P41243 INFO Train loss: 0.352875
2022-01-30 00:43:06,522 P41243 INFO ************ Epoch=8 end ************
2022-01-30 00:43:43,533 P41243 INFO [Metrics] AUC: 0.951419 - logloss: 0.253029
2022-01-30 00:43:43,533 P41243 INFO Save best model: monitor(max): 0.951419
2022-01-30 00:43:43,547 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:43:43,589 P41243 INFO Train loss: 0.350903
2022-01-30 00:43:43,590 P41243 INFO ************ Epoch=9 end ************
2022-01-30 00:44:20,663 P41243 INFO [Metrics] AUC: 0.952693 - logloss: 0.251171
2022-01-30 00:44:20,663 P41243 INFO Save best model: monitor(max): 0.952693
2022-01-30 00:44:20,677 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:44:20,718 P41243 INFO Train loss: 0.348767
2022-01-30 00:44:20,718 P41243 INFO ************ Epoch=10 end ************
2022-01-30 00:44:57,984 P41243 INFO [Metrics] AUC: 0.953389 - logloss: 0.248103
2022-01-30 00:44:57,984 P41243 INFO Save best model: monitor(max): 0.953389
2022-01-30 00:44:57,997 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:44:58,043 P41243 INFO Train loss: 0.345958
2022-01-30 00:44:58,043 P41243 INFO ************ Epoch=11 end ************
2022-01-30 00:45:33,717 P41243 INFO [Metrics] AUC: 0.954416 - logloss: 0.245438
2022-01-30 00:45:33,718 P41243 INFO Save best model: monitor(max): 0.954416
2022-01-30 00:45:33,731 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:45:33,785 P41243 INFO Train loss: 0.342470
2022-01-30 00:45:33,785 P41243 INFO ************ Epoch=12 end ************
2022-01-30 00:46:11,211 P41243 INFO [Metrics] AUC: 0.955081 - logloss: 0.246321
2022-01-30 00:46:11,211 P41243 INFO Save best model: monitor(max): 0.955081
2022-01-30 00:46:11,224 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:46:11,278 P41243 INFO Train loss: 0.338815
2022-01-30 00:46:11,278 P41243 INFO ************ Epoch=13 end ************
2022-01-30 00:46:57,451 P41243 INFO [Metrics] AUC: 0.956279 - logloss: 0.242704
2022-01-30 00:46:57,451 P41243 INFO Save best model: monitor(max): 0.956279
2022-01-30 00:46:57,464 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:46:57,504 P41243 INFO Train loss: 0.334712
2022-01-30 00:46:57,504 P41243 INFO ************ Epoch=14 end ************
2022-01-30 00:47:44,070 P41243 INFO [Metrics] AUC: 0.956665 - logloss: 0.242768
2022-01-30 00:47:44,070 P41243 INFO Save best model: monitor(max): 0.956665
2022-01-30 00:47:44,083 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:47:44,122 P41243 INFO Train loss: 0.330967
2022-01-30 00:47:44,123 P41243 INFO ************ Epoch=15 end ************
2022-01-30 00:48:30,454 P41243 INFO [Metrics] AUC: 0.957385 - logloss: 0.240455
2022-01-30 00:48:30,454 P41243 INFO Save best model: monitor(max): 0.957385
2022-01-30 00:48:30,468 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:48:30,510 P41243 INFO Train loss: 0.327170
2022-01-30 00:48:30,510 P41243 INFO ************ Epoch=16 end ************
2022-01-30 00:49:16,816 P41243 INFO [Metrics] AUC: 0.957847 - logloss: 0.239310
2022-01-30 00:49:16,816 P41243 INFO Save best model: monitor(max): 0.957847
2022-01-30 00:49:16,830 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:49:16,870 P41243 INFO Train loss: 0.324363
2022-01-30 00:49:16,870 P41243 INFO ************ Epoch=17 end ************
2022-01-30 00:50:02,922 P41243 INFO [Metrics] AUC: 0.958175 - logloss: 0.241587
2022-01-30 00:50:02,923 P41243 INFO Save best model: monitor(max): 0.958175
2022-01-30 00:50:02,936 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:50:02,974 P41243 INFO Train loss: 0.321523
2022-01-30 00:50:02,974 P41243 INFO ************ Epoch=18 end ************
2022-01-30 00:50:49,753 P41243 INFO [Metrics] AUC: 0.958295 - logloss: 0.239856
2022-01-30 00:50:49,753 P41243 INFO Save best model: monitor(max): 0.958295
2022-01-30 00:50:49,767 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:50:49,809 P41243 INFO Train loss: 0.319810
2022-01-30 00:50:49,809 P41243 INFO ************ Epoch=19 end ************
2022-01-30 00:51:31,375 P41243 INFO [Metrics] AUC: 0.959059 - logloss: 0.237365
2022-01-30 00:51:31,376 P41243 INFO Save best model: monitor(max): 0.959059
2022-01-30 00:51:31,389 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:51:31,431 P41243 INFO Train loss: 0.317080
2022-01-30 00:51:31,432 P41243 INFO ************ Epoch=20 end ************
2022-01-30 00:52:17,756 P41243 INFO [Metrics] AUC: 0.959489 - logloss: 0.237505
2022-01-30 00:52:17,757 P41243 INFO Save best model: monitor(max): 0.959489
2022-01-30 00:52:17,770 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:52:17,811 P41243 INFO Train loss: 0.315333
2022-01-30 00:52:17,811 P41243 INFO ************ Epoch=21 end ************
2022-01-30 00:53:04,026 P41243 INFO [Metrics] AUC: 0.959993 - logloss: 0.236892
2022-01-30 00:53:04,026 P41243 INFO Save best model: monitor(max): 0.959993
2022-01-30 00:53:04,040 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:53:04,081 P41243 INFO Train loss: 0.312758
2022-01-30 00:53:04,081 P41243 INFO ************ Epoch=22 end ************
2022-01-30 00:53:50,154 P41243 INFO [Metrics] AUC: 0.959670 - logloss: 0.239353
2022-01-30 00:53:50,155 P41243 INFO Monitor(max) STOP: 0.959670 !
2022-01-30 00:53:50,155 P41243 INFO Reduce learning rate on plateau: 0.000100
2022-01-30 00:53:50,155 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:53:50,195 P41243 INFO Train loss: 0.310585
2022-01-30 00:53:50,195 P41243 INFO ************ Epoch=23 end ************
2022-01-30 00:54:36,467 P41243 INFO [Metrics] AUC: 0.960780 - logloss: 0.266450
2022-01-30 00:54:36,468 P41243 INFO Save best model: monitor(max): 0.960780
2022-01-30 00:54:36,481 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:54:36,533 P41243 INFO Train loss: 0.267147
2022-01-30 00:54:36,533 P41243 INFO ************ Epoch=24 end ************
2022-01-30 00:55:22,592 P41243 INFO [Metrics] AUC: 0.961010 - logloss: 0.297078
2022-01-30 00:55:22,592 P41243 INFO Save best model: monitor(max): 0.961010
2022-01-30 00:55:22,605 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:55:22,644 P41243 INFO Train loss: 0.234620
2022-01-30 00:55:22,645 P41243 INFO ************ Epoch=25 end ************
2022-01-30 00:56:08,559 P41243 INFO [Metrics] AUC: 0.960620 - logloss: 0.329606
2022-01-30 00:56:08,559 P41243 INFO Monitor(max) STOP: 0.960620 !
2022-01-30 00:56:08,559 P41243 INFO Reduce learning rate on plateau: 0.000010
2022-01-30 00:56:08,559 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:56:08,597 P41243 INFO Train loss: 0.209719
2022-01-30 00:56:08,597 P41243 INFO ************ Epoch=26 end ************
2022-01-30 00:56:54,698 P41243 INFO [Metrics] AUC: 0.960353 - logloss: 0.349530
2022-01-30 00:56:54,699 P41243 INFO Monitor(max) STOP: 0.960353 !
2022-01-30 00:56:54,699 P41243 INFO Reduce learning rate on plateau: 0.000001
2022-01-30 00:56:54,699 P41243 INFO Early stopping at epoch=27
2022-01-30 00:56:54,699 P41243 INFO --- 343/343 batches finished ---
2022-01-30 00:56:54,738 P41243 INFO Train loss: 0.187876
2022-01-30 00:56:54,738 P41243 INFO Training finished.
2022-01-30 00:56:54,738 P41243 INFO Load best model: /home/XXXX/FuxiCTR/benchmarks/Movielens/AFN_movielenslatest_x1/movielenslatest_x1_cd32d937/AFN_movielenslatest_x1_015_9d6fa874.model
2022-01-30 00:56:59,725 P41243 INFO ****** Validation evaluation ******
2022-01-30 00:57:02,982 P41243 INFO [Metrics] AUC: 0.961010 - logloss: 0.297078
2022-01-30 00:57:03,025 P41243 INFO ******** Test evaluation ********
2022-01-30 00:57:03,025 P41243 INFO Loading data...
2022-01-30 00:57:03,025 P41243 INFO Loading data from h5: ../data/Movielens/movielenslatest_x1_cd32d937/test.h5
2022-01-30 00:57:03,030 P41243 INFO Test samples: total/200686, pos/66850, neg/133836, ratio/33.31%, blocks/1
2022-01-30 00:57:03,030 P41243 INFO Loading test data done.
2022-01-30 00:57:04,758 P41243 INFO [Metrics] AUC: 0.961073 - logloss: 0.296298

```
