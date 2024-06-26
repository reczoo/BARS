## IPNN_kkbox_x1

A hands-on guide to run the PNN model on the KKBox_x1 dataset.

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
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [PNN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/PNN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PNN_kkbox_x1_tuner_config_02](./PNN_kkbox_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd IPNN_kkbox_x1
    nohup python run_expid.py --config ./PNN_kkbox_x1_tuner_config_02 --expid PNN_kkbox_x1_015_1b4d837a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.479323 | 0.851470  |


### Logs
```python
2022-03-07 22:57:40,663 P83978 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "6",
    "hidden_activations": "relu",
    "hidden_units": "[2000, 2000]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "PNN",
    "model_id": "PNN_kkbox_x1_015_1b4d837a",
    "model_root": "./KKBox/PNN_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KKBox/KKBox_x1/test.csv",
    "train_data": "../data/KKBox/KKBox_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-07 22:57:40,663 P83978 INFO Set up feature encoder...
2022-03-07 22:57:40,664 P83978 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-07 22:57:42,357 P83978 INFO Total number of parameters: 19297617.
2022-03-07 22:57:42,357 P83978 INFO Loading data...
2022-03-07 22:57:42,357 P83978 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-07 22:57:42,824 P83978 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-07 22:57:43,107 P83978 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-07 22:57:43,127 P83978 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-07 22:57:43,127 P83978 INFO Loading train data done.
2022-03-07 22:57:49,393 P83978 INFO Start training: 591 batches/epoch
2022-03-07 22:57:49,393 P83978 INFO ************ Epoch=1 start ************
2022-03-07 23:00:29,791 P83978 INFO [Metrics] logloss: 0.556914 - AUC: 0.784949
2022-03-07 23:00:29,795 P83978 INFO Save best model: monitor(max): 0.228035
2022-03-07 23:00:30,589 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:00:30,623 P83978 INFO Train loss: 0.620746
2022-03-07 23:00:30,623 P83978 INFO ************ Epoch=1 end ************
2022-03-07 23:03:11,095 P83978 INFO [Metrics] logloss: 0.542254 - AUC: 0.798857
2022-03-07 23:03:11,096 P83978 INFO Save best model: monitor(max): 0.256602
2022-03-07 23:03:11,219 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:03:11,261 P83978 INFO Train loss: 0.605169
2022-03-07 23:03:11,261 P83978 INFO ************ Epoch=2 end ************
2022-03-07 23:05:51,992 P83978 INFO [Metrics] logloss: 0.532417 - AUC: 0.807791
2022-03-07 23:05:51,995 P83978 INFO Save best model: monitor(max): 0.275374
2022-03-07 23:05:52,109 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:05:52,145 P83978 INFO Train loss: 0.599968
2022-03-07 23:05:52,145 P83978 INFO ************ Epoch=3 end ************
2022-03-07 23:08:32,880 P83978 INFO [Metrics] logloss: 0.527426 - AUC: 0.812543
2022-03-07 23:08:32,884 P83978 INFO Save best model: monitor(max): 0.285118
2022-03-07 23:08:32,992 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:08:33,033 P83978 INFO Train loss: 0.600445
2022-03-07 23:08:33,033 P83978 INFO ************ Epoch=4 end ************
2022-03-07 23:11:13,823 P83978 INFO [Metrics] logloss: 0.521944 - AUC: 0.816970
2022-03-07 23:11:13,826 P83978 INFO Save best model: monitor(max): 0.295026
2022-03-07 23:11:13,914 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:11:13,951 P83978 INFO Train loss: 0.599511
2022-03-07 23:11:13,951 P83978 INFO ************ Epoch=5 end ************
2022-03-07 23:13:47,206 P83978 INFO [Metrics] logloss: 0.519594 - AUC: 0.818779
2022-03-07 23:13:47,209 P83978 INFO Save best model: monitor(max): 0.299185
2022-03-07 23:13:47,317 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:13:47,355 P83978 INFO Train loss: 0.597108
2022-03-07 23:13:47,356 P83978 INFO ************ Epoch=6 end ************
2022-03-07 23:15:47,223 P83978 INFO [Metrics] logloss: 0.516954 - AUC: 0.821172
2022-03-07 23:15:47,226 P83978 INFO Save best model: monitor(max): 0.304218
2022-03-07 23:15:47,322 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:15:47,379 P83978 INFO Train loss: 0.595444
2022-03-07 23:15:47,379 P83978 INFO ************ Epoch=7 end ************
2022-03-07 23:17:46,852 P83978 INFO [Metrics] logloss: 0.515826 - AUC: 0.822471
2022-03-07 23:17:46,855 P83978 INFO Save best model: monitor(max): 0.306645
2022-03-07 23:17:46,957 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:17:47,007 P83978 INFO Train loss: 0.594116
2022-03-07 23:17:47,007 P83978 INFO ************ Epoch=8 end ************
2022-03-07 23:19:46,706 P83978 INFO [Metrics] logloss: 0.513982 - AUC: 0.823686
2022-03-07 23:19:46,709 P83978 INFO Save best model: monitor(max): 0.309704
2022-03-07 23:19:46,818 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:19:46,855 P83978 INFO Train loss: 0.592783
2022-03-07 23:19:46,855 P83978 INFO ************ Epoch=9 end ************
2022-03-07 23:21:46,228 P83978 INFO [Metrics] logloss: 0.512930 - AUC: 0.824193
2022-03-07 23:21:46,230 P83978 INFO Save best model: monitor(max): 0.311263
2022-03-07 23:21:46,315 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:21:46,351 P83978 INFO Train loss: 0.592408
2022-03-07 23:21:46,351 P83978 INFO ************ Epoch=10 end ************
2022-03-07 23:23:45,684 P83978 INFO [Metrics] logloss: 0.511574 - AUC: 0.825521
2022-03-07 23:23:45,687 P83978 INFO Save best model: monitor(max): 0.313948
2022-03-07 23:23:45,800 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:23:45,837 P83978 INFO Train loss: 0.591254
2022-03-07 23:23:45,837 P83978 INFO ************ Epoch=11 end ************
2022-03-07 23:25:45,403 P83978 INFO [Metrics] logloss: 0.511324 - AUC: 0.825845
2022-03-07 23:25:45,406 P83978 INFO Save best model: monitor(max): 0.314521
2022-03-07 23:25:45,507 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:25:45,546 P83978 INFO Train loss: 0.591379
2022-03-07 23:25:45,546 P83978 INFO ************ Epoch=12 end ************
2022-03-07 23:27:45,060 P83978 INFO [Metrics] logloss: 0.509694 - AUC: 0.827103
2022-03-07 23:27:45,063 P83978 INFO Save best model: monitor(max): 0.317409
2022-03-07 23:27:45,159 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:27:45,201 P83978 INFO Train loss: 0.591142
2022-03-07 23:27:45,202 P83978 INFO ************ Epoch=13 end ************
2022-03-07 23:29:45,539 P83978 INFO [Metrics] logloss: 0.509176 - AUC: 0.827200
2022-03-07 23:29:45,542 P83978 INFO Save best model: monitor(max): 0.318024
2022-03-07 23:29:45,665 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:29:45,704 P83978 INFO Train loss: 0.590476
2022-03-07 23:29:45,704 P83978 INFO ************ Epoch=14 end ************
2022-03-07 23:31:45,683 P83978 INFO [Metrics] logloss: 0.508450 - AUC: 0.827863
2022-03-07 23:31:45,686 P83978 INFO Save best model: monitor(max): 0.319413
2022-03-07 23:31:45,772 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:31:45,813 P83978 INFO Train loss: 0.590467
2022-03-07 23:31:45,813 P83978 INFO ************ Epoch=15 end ************
2022-03-07 23:33:45,159 P83978 INFO [Metrics] logloss: 0.508070 - AUC: 0.828275
2022-03-07 23:33:45,162 P83978 INFO Save best model: monitor(max): 0.320206
2022-03-07 23:33:45,297 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:33:45,337 P83978 INFO Train loss: 0.590014
2022-03-07 23:33:45,337 P83978 INFO ************ Epoch=16 end ************
2022-03-07 23:35:45,001 P83978 INFO [Metrics] logloss: 0.507963 - AUC: 0.828106
2022-03-07 23:35:45,004 P83978 INFO Monitor(max) STOP: 0.320143 !
2022-03-07 23:35:45,004 P83978 INFO Reduce learning rate on plateau: 0.000100
2022-03-07 23:35:45,004 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:35:45,042 P83978 INFO Train loss: 0.589758
2022-03-07 23:35:45,042 P83978 INFO ************ Epoch=17 end ************
2022-03-07 23:37:44,268 P83978 INFO [Metrics] logloss: 0.483641 - AUC: 0.846992
2022-03-07 23:37:44,272 P83978 INFO Save best model: monitor(max): 0.363351
2022-03-07 23:37:44,360 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:37:44,397 P83978 INFO Train loss: 0.518607
2022-03-07 23:37:44,398 P83978 INFO ************ Epoch=18 end ************
2022-03-07 23:39:43,472 P83978 INFO [Metrics] logloss: 0.479736 - AUC: 0.850493
2022-03-07 23:39:43,474 P83978 INFO Save best model: monitor(max): 0.370757
2022-03-07 23:39:43,587 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:39:43,637 P83978 INFO Train loss: 0.485755
2022-03-07 23:39:43,637 P83978 INFO ************ Epoch=19 end ************
2022-03-07 23:41:42,838 P83978 INFO [Metrics] logloss: 0.479524 - AUC: 0.851428
2022-03-07 23:41:42,841 P83978 INFO Save best model: monitor(max): 0.371904
2022-03-07 23:41:42,926 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:41:42,964 P83978 INFO Train loss: 0.471771
2022-03-07 23:41:42,965 P83978 INFO ************ Epoch=20 end ************
2022-03-07 23:43:42,408 P83978 INFO [Metrics] logloss: 0.483106 - AUC: 0.850214
2022-03-07 23:43:42,411 P83978 INFO Monitor(max) STOP: 0.367108 !
2022-03-07 23:43:42,411 P83978 INFO Reduce learning rate on plateau: 0.000010
2022-03-07 23:43:42,411 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:43:42,448 P83978 INFO Train loss: 0.459693
2022-03-07 23:43:42,448 P83978 INFO ************ Epoch=21 end ************
2022-03-07 23:45:42,172 P83978 INFO [Metrics] logloss: 0.508886 - AUC: 0.845849
2022-03-07 23:45:42,176 P83978 INFO Monitor(max) STOP: 0.336963 !
2022-03-07 23:45:42,176 P83978 INFO Reduce learning rate on plateau: 0.000001
2022-03-07 23:45:42,176 P83978 INFO Early stopping at epoch=22
2022-03-07 23:45:42,177 P83978 INFO --- 591/591 batches finished ---
2022-03-07 23:45:42,214 P83978 INFO Train loss: 0.409497
2022-03-07 23:45:42,214 P83978 INFO Training finished.
2022-03-07 23:45:42,214 P83978 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/PNN_kkbox_x1/kkbox_x1_227d337d/PNN_kkbox_x1_015_1b4d837a_model.ckpt
2022-03-07 23:45:42,309 P83978 INFO ****** Validation evaluation ******
2022-03-07 23:45:47,360 P83978 INFO [Metrics] logloss: 0.479524 - AUC: 0.851428
2022-03-07 23:45:47,420 P83978 INFO ******** Test evaluation ********
2022-03-07 23:45:47,420 P83978 INFO Loading data...
2022-03-07 23:45:47,420 P83978 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-07 23:45:47,497 P83978 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-07 23:45:47,497 P83978 INFO Loading test data done.
2022-03-07 23:45:52,746 P83978 INFO [Metrics] logloss: 0.479323 - AUC: 0.851470

```
