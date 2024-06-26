## AFN_avazu_x1

A hands-on guide to run the AFN model on the Avazu_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.1.0

  ```

### Dataset
Dataset ID: [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AFN](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AFN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFN_avazu_x1_tuner_config_01](./AFN_avazu_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFN_avazu_x1
    nohup python run_expid.py --config ./AFN_avazu_x1_tuner_config_01 --expid AFN_avazu_x1_014_af89ab38 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.754655 | 0.372925  |


### Logs
```python
2022-01-21 10:15:02,373 P811 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.3",
    "afn_hidden_units": "[400]",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x1_3fb65689",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.01",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1500",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AFN",
    "model_id": "AFN_avazu_x1_014_af89ab38",
    "model_root": "./Avazu/AFN_avazu_x1/",
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
    "test_data": "../data/Avazu/Avazu_x1/test.csv",
    "train_data": "../data/Avazu/Avazu_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-21 10:15:02,374 P811 INFO Set up feature encoder...
2022-01-21 10:15:02,374 P811 INFO Load feature_map from json: ../data/Avazu/avazu_x1_3fb65689/feature_map.json
2022-01-21 10:15:02,374 P811 INFO Loading data...
2022-01-21 10:15:02,375 P811 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/train.h5
2022-01-21 10:15:05,191 P811 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/valid.h5
2022-01-21 10:15:05,596 P811 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2022-01-21 10:15:05,596 P811 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2022-01-21 10:15:05,596 P811 INFO Loading train data done.
2022-01-21 10:15:11,891 P811 INFO Total number of parameters: 19022835.
2022-01-21 10:15:11,891 P811 INFO Start training: 6910 batches/epoch
2022-01-21 10:15:11,891 P811 INFO ************ Epoch=1 start ************
2022-01-21 10:30:45,500 P811 INFO [Metrics] AUC: 0.706545 - logloss: 0.414640
2022-01-21 10:30:45,502 P811 INFO Save best model: monitor(max): 0.706545
2022-01-21 10:30:45,852 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 10:30:45,893 P811 INFO Train loss: 0.524678
2022-01-21 10:30:45,893 P811 INFO ************ Epoch=1 end ************
2022-01-21 10:46:18,834 P811 INFO [Metrics] AUC: 0.710342 - logloss: 0.500509
2022-01-21 10:46:18,837 P811 INFO Save best model: monitor(max): 0.710342
2022-01-21 10:46:18,939 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 10:46:18,984 P811 INFO Train loss: 0.485839
2022-01-21 10:46:18,984 P811 INFO ************ Epoch=2 end ************
2022-01-21 11:01:50,883 P811 INFO [Metrics] AUC: 0.711304 - logloss: 0.512217
2022-01-21 11:01:50,886 P811 INFO Save best model: monitor(max): 0.711304
2022-01-21 11:01:50,988 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 11:01:51,037 P811 INFO Train loss: 0.488231
2022-01-21 11:01:51,038 P811 INFO ************ Epoch=3 end ************
2022-01-21 11:16:53,722 P811 INFO [Metrics] AUC: 0.705958 - logloss: 0.540515
2022-01-21 11:16:53,725 P811 INFO Monitor(max) STOP: 0.705958 !
2022-01-21 11:16:53,726 P811 INFO Reduce learning rate on plateau: 0.000100
2022-01-21 11:16:53,726 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 11:16:53,772 P811 INFO Train loss: 0.502416
2022-01-21 11:16:53,772 P811 INFO ************ Epoch=4 end ************
2022-01-21 11:32:14,695 P811 INFO [Metrics] AUC: 0.713797 - logloss: 0.505290
2022-01-21 11:32:14,699 P811 INFO Save best model: monitor(max): 0.713797
2022-01-21 11:32:14,797 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 11:32:14,846 P811 INFO Train loss: 0.422496
2022-01-21 11:32:14,846 P811 INFO ************ Epoch=5 end ************
2022-01-21 11:47:35,388 P811 INFO [Metrics] AUC: 0.709671 - logloss: 0.553019
2022-01-21 11:47:35,390 P811 INFO Monitor(max) STOP: 0.709671 !
2022-01-21 11:47:35,390 P811 INFO Reduce learning rate on plateau: 0.000010
2022-01-21 11:47:35,390 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 11:47:35,443 P811 INFO Train loss: 0.406557
2022-01-21 11:47:35,443 P811 INFO ************ Epoch=6 end ************
2022-01-21 12:02:17,011 P811 INFO [Metrics] AUC: 0.719455 - logloss: 0.477549
2022-01-21 12:02:17,013 P811 INFO Save best model: monitor(max): 0.719455
2022-01-21 12:02:17,112 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 12:02:17,153 P811 INFO Train loss: 0.401866
2022-01-21 12:02:17,153 P811 INFO ************ Epoch=7 end ************
2022-01-21 12:17:17,194 P811 INFO [Metrics] AUC: 0.721626 - logloss: 0.451963
2022-01-21 12:17:17,197 P811 INFO Save best model: monitor(max): 0.721626
2022-01-21 12:17:17,290 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 12:17:17,336 P811 INFO Train loss: 0.397810
2022-01-21 12:17:17,336 P811 INFO ************ Epoch=8 end ************
2022-01-21 12:32:18,211 P811 INFO [Metrics] AUC: 0.723923 - logloss: 0.433900
2022-01-21 12:32:18,214 P811 INFO Save best model: monitor(max): 0.723923
2022-01-21 12:32:18,304 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 12:32:18,344 P811 INFO Train loss: 0.396589
2022-01-21 12:32:18,345 P811 INFO ************ Epoch=9 end ************
2022-01-21 12:47:18,838 P811 INFO [Metrics] AUC: 0.724275 - logloss: 0.430525
2022-01-21 12:47:18,840 P811 INFO Save best model: monitor(max): 0.724275
2022-01-21 12:47:18,925 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 12:47:18,966 P811 INFO Train loss: 0.395795
2022-01-21 12:47:18,966 P811 INFO ************ Epoch=10 end ************
2022-01-21 13:02:18,998 P811 INFO [Metrics] AUC: 0.725010 - logloss: 0.425413
2022-01-21 13:02:19,000 P811 INFO Save best model: monitor(max): 0.725010
2022-01-21 13:02:19,084 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:02:19,125 P811 INFO Train loss: 0.395167
2022-01-21 13:02:19,125 P811 INFO ************ Epoch=11 end ************
2022-01-21 13:11:49,791 P811 INFO [Metrics] AUC: 0.724172 - logloss: 0.427023
2022-01-21 13:11:49,794 P811 INFO Monitor(max) STOP: 0.724172 !
2022-01-21 13:11:49,794 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 13:11:49,794 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:11:49,843 P811 INFO Train loss: 0.394674
2022-01-21 13:11:49,844 P811 INFO ************ Epoch=12 end ************
2022-01-21 13:19:10,421 P811 INFO [Metrics] AUC: 0.726458 - logloss: 0.418822
2022-01-21 13:19:10,426 P811 INFO Save best model: monitor(max): 0.726458
2022-01-21 13:19:10,525 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:19:10,574 P811 INFO Train loss: 0.393261
2022-01-21 13:19:10,574 P811 INFO ************ Epoch=13 end ************
2022-01-21 13:26:29,712 P811 INFO [Metrics] AUC: 0.726527 - logloss: 0.416674
2022-01-21 13:26:29,714 P811 INFO Save best model: monitor(max): 0.726527
2022-01-21 13:26:29,799 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:26:29,841 P811 INFO Train loss: 0.392796
2022-01-21 13:26:29,841 P811 INFO ************ Epoch=14 end ************
2022-01-21 13:33:49,283 P811 INFO [Metrics] AUC: 0.727118 - logloss: 0.415022
2022-01-21 13:33:49,285 P811 INFO Save best model: monitor(max): 0.727118
2022-01-21 13:33:49,370 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:33:49,410 P811 INFO Train loss: 0.392551
2022-01-21 13:33:49,411 P811 INFO ************ Epoch=15 end ************
2022-01-21 13:41:08,914 P811 INFO [Metrics] AUC: 0.727478 - logloss: 0.412724
2022-01-21 13:41:08,916 P811 INFO Save best model: monitor(max): 0.727478
2022-01-21 13:41:09,008 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:41:09,048 P811 INFO Train loss: 0.392348
2022-01-21 13:41:09,048 P811 INFO ************ Epoch=16 end ************
2022-01-21 13:48:27,909 P811 INFO [Metrics] AUC: 0.728078 - logloss: 0.411513
2022-01-21 13:48:27,912 P811 INFO Save best model: monitor(max): 0.728078
2022-01-21 13:48:28,007 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:48:28,047 P811 INFO Train loss: 0.392194
2022-01-21 13:48:28,048 P811 INFO ************ Epoch=17 end ************
2022-01-21 13:55:47,098 P811 INFO [Metrics] AUC: 0.728113 - logloss: 0.410680
2022-01-21 13:55:47,102 P811 INFO Save best model: monitor(max): 0.728113
2022-01-21 13:55:47,201 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 13:55:47,242 P811 INFO Train loss: 0.392023
2022-01-21 13:55:47,242 P811 INFO ************ Epoch=18 end ************
2022-01-21 14:03:10,772 P811 INFO [Metrics] AUC: 0.728344 - logloss: 0.410018
2022-01-21 14:03:10,775 P811 INFO Save best model: monitor(max): 0.728344
2022-01-21 14:03:10,858 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:03:10,912 P811 INFO Train loss: 0.391936
2022-01-21 14:03:10,912 P811 INFO ************ Epoch=19 end ************
2022-01-21 14:10:31,205 P811 INFO [Metrics] AUC: 0.728628 - logloss: 0.409587
2022-01-21 14:10:31,208 P811 INFO Save best model: monitor(max): 0.728628
2022-01-21 14:10:31,301 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:10:31,348 P811 INFO Train loss: 0.391828
2022-01-21 14:10:31,348 P811 INFO ************ Epoch=20 end ************
2022-01-21 14:17:50,772 P811 INFO [Metrics] AUC: 0.728907 - logloss: 0.408836
2022-01-21 14:17:50,775 P811 INFO Save best model: monitor(max): 0.728907
2022-01-21 14:17:50,860 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:17:50,911 P811 INFO Train loss: 0.391733
2022-01-21 14:17:50,911 P811 INFO ************ Epoch=21 end ************
2022-01-21 14:25:14,047 P811 INFO [Metrics] AUC: 0.729400 - logloss: 0.408331
2022-01-21 14:25:14,051 P811 INFO Save best model: monitor(max): 0.729400
2022-01-21 14:25:14,142 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:25:14,194 P811 INFO Train loss: 0.391639
2022-01-21 14:25:14,194 P811 INFO ************ Epoch=22 end ************
2022-01-21 14:32:38,098 P811 INFO [Metrics] AUC: 0.729480 - logloss: 0.408168
2022-01-21 14:32:38,101 P811 INFO Save best model: monitor(max): 0.729480
2022-01-21 14:32:38,191 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:32:38,232 P811 INFO Train loss: 0.391560
2022-01-21 14:32:38,232 P811 INFO ************ Epoch=23 end ************
2022-01-21 14:39:59,905 P811 INFO [Metrics] AUC: 0.729380 - logloss: 0.408088
2022-01-21 14:39:59,907 P811 INFO Monitor(max) STOP: 0.729380 !
2022-01-21 14:39:59,907 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 14:39:59,908 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:39:59,958 P811 INFO Train loss: 0.391486
2022-01-21 14:39:59,958 P811 INFO ************ Epoch=24 end ************
2022-01-21 14:47:21,717 P811 INFO [Metrics] AUC: 0.729580 - logloss: 0.407679
2022-01-21 14:47:21,719 P811 INFO Save best model: monitor(max): 0.729580
2022-01-21 14:47:21,811 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:47:21,859 P811 INFO Train loss: 0.391394
2022-01-21 14:47:21,859 P811 INFO ************ Epoch=25 end ************
2022-01-21 14:54:42,398 P811 INFO [Metrics] AUC: 0.729747 - logloss: 0.407568
2022-01-21 14:54:42,401 P811 INFO Save best model: monitor(max): 0.729747
2022-01-21 14:54:42,503 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 14:54:42,541 P811 INFO Train loss: 0.391329
2022-01-21 14:54:42,541 P811 INFO ************ Epoch=26 end ************
2022-01-21 15:02:00,059 P811 INFO [Metrics] AUC: 0.729630 - logloss: 0.407413
2022-01-21 15:02:00,063 P811 INFO Monitor(max) STOP: 0.729630 !
2022-01-21 15:02:00,063 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 15:02:00,063 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:02:00,117 P811 INFO Train loss: 0.391288
2022-01-21 15:02:00,117 P811 INFO ************ Epoch=27 end ************
2022-01-21 15:09:17,133 P811 INFO [Metrics] AUC: 0.729929 - logloss: 0.407202
2022-01-21 15:09:17,138 P811 INFO Save best model: monitor(max): 0.729929
2022-01-21 15:09:17,239 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:09:17,286 P811 INFO Train loss: 0.391240
2022-01-21 15:09:17,287 P811 INFO ************ Epoch=28 end ************
2022-01-21 15:16:34,824 P811 INFO [Metrics] AUC: 0.729957 - logloss: 0.406989
2022-01-21 15:16:34,827 P811 INFO Save best model: monitor(max): 0.729957
2022-01-21 15:16:34,928 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:16:34,980 P811 INFO Train loss: 0.391187
2022-01-21 15:16:34,980 P811 INFO ************ Epoch=29 end ************
2022-01-21 15:23:52,206 P811 INFO [Metrics] AUC: 0.730043 - logloss: 0.406810
2022-01-21 15:23:52,209 P811 INFO Save best model: monitor(max): 0.730043
2022-01-21 15:23:52,320 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:23:52,367 P811 INFO Train loss: 0.391155
2022-01-21 15:23:52,367 P811 INFO ************ Epoch=30 end ************
2022-01-21 15:31:09,911 P811 INFO [Metrics] AUC: 0.730689 - logloss: 0.406587
2022-01-21 15:31:09,914 P811 INFO Save best model: monitor(max): 0.730689
2022-01-21 15:31:10,005 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:31:10,056 P811 INFO Train loss: 0.391121
2022-01-21 15:31:10,056 P811 INFO ************ Epoch=31 end ************
2022-01-21 15:38:28,965 P811 INFO [Metrics] AUC: 0.730474 - logloss: 0.406662
2022-01-21 15:38:28,968 P811 INFO Monitor(max) STOP: 0.730474 !
2022-01-21 15:38:28,968 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 15:38:28,968 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:38:29,014 P811 INFO Train loss: 0.391060
2022-01-21 15:38:29,015 P811 INFO ************ Epoch=32 end ************
2022-01-21 15:45:46,326 P811 INFO [Metrics] AUC: 0.730850 - logloss: 0.406499
2022-01-21 15:45:46,331 P811 INFO Save best model: monitor(max): 0.730850
2022-01-21 15:45:46,434 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:45:46,487 P811 INFO Train loss: 0.391016
2022-01-21 15:45:46,487 P811 INFO ************ Epoch=33 end ************
2022-01-21 15:53:04,650 P811 INFO [Metrics] AUC: 0.730951 - logloss: 0.406185
2022-01-21 15:53:04,654 P811 INFO Save best model: monitor(max): 0.730951
2022-01-21 15:53:04,756 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 15:53:04,799 P811 INFO Train loss: 0.390979
2022-01-21 15:53:04,799 P811 INFO ************ Epoch=34 end ************
2022-01-21 16:00:29,837 P811 INFO [Metrics] AUC: 0.731529 - logloss: 0.406079
2022-01-21 16:00:29,840 P811 INFO Save best model: monitor(max): 0.731529
2022-01-21 16:00:29,931 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 16:00:29,982 P811 INFO Train loss: 0.390957
2022-01-21 16:00:29,983 P811 INFO ************ Epoch=35 end ************
2022-01-21 16:07:48,119 P811 INFO [Metrics] AUC: 0.731021 - logloss: 0.406057
2022-01-21 16:07:48,123 P811 INFO Monitor(max) STOP: 0.731021 !
2022-01-21 16:07:48,123 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 16:07:48,123 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 16:07:48,173 P811 INFO Train loss: 0.390927
2022-01-21 16:07:48,174 P811 INFO ************ Epoch=36 end ************
2022-01-21 16:15:06,199 P811 INFO [Metrics] AUC: 0.731329 - logloss: 0.405795
2022-01-21 16:15:06,203 P811 INFO Monitor(max) STOP: 0.731329 !
2022-01-21 16:15:06,204 P811 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 16:15:06,204 P811 INFO Early stopping at epoch=37
2022-01-21 16:15:06,204 P811 INFO --- 6910/6910 batches finished ---
2022-01-21 16:15:06,247 P811 INFO Train loss: 0.390871
2022-01-21 16:15:06,247 P811 INFO Training finished.
2022-01-21 16:15:06,247 P811 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/AFN_avazu_x1/avazu_x1_3fb65689/AFN_avazu_x1_014_af89ab38.model
2022-01-21 16:15:08,920 P811 INFO ****** Validation evaluation ******
2022-01-21 16:15:32,139 P811 INFO [Metrics] AUC: 0.731529 - logloss: 0.406079
2022-01-21 16:15:32,202 P811 INFO ******** Test evaluation ********
2022-01-21 16:15:32,203 P811 INFO Loading data...
2022-01-21 16:15:32,203 P811 INFO Loading data from h5: ../data/Avazu/avazu_x1_3fb65689/test.h5
2022-01-21 16:15:33,101 P811 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2022-01-21 16:15:33,101 P811 INFO Loading test data done.
2022-01-21 16:16:20,275 P811 INFO [Metrics] AUC: 0.754655 - logloss: 0.372925

```
