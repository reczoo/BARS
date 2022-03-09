## AutoInt+_kkbox_x1

A hands-on guide to run the AutoInt model on the KKBox_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_kkbox_x1_tuner_config_06](./AutoInt_kkbox_x1_tuner_config_06). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt+_kkbox_x1
    nohup python run_expid.py --config ./AutoInt_kkbox_x1_tuner_config_06 --expid AutoInt_kkbox_x1_003_df9d4ed6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.477251 | 0.853443  |


### Logs
```python
2022-03-09 15:43:24,259 P61876 INFO {
    "attention_dim": "256",
    "attention_layers": "4",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[5000, 5000]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "2",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "AutoInt",
    "model_id": "AutoInt_kkbox_x1_003_df9d4ed6",
    "model_root": "./KKBox/AutoInt_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_heads": "1",
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
    "use_residual": "False",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-09 15:43:24,260 P61876 INFO Set up feature encoder...
2022-03-09 15:43:24,260 P61876 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-09 15:43:26,148 P61876 INFO Total number of parameters: 45959089.
2022-03-09 15:43:26,148 P61876 INFO Loading data...
2022-03-09 15:43:26,149 P61876 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-09 15:43:26,558 P61876 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-09 15:43:26,839 P61876 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-09 15:43:26,868 P61876 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 15:43:26,868 P61876 INFO Loading train data done.
2022-03-09 15:43:32,230 P61876 INFO Start training: 591 batches/epoch
2022-03-09 15:43:32,231 P61876 INFO ************ Epoch=1 start ************
2022-03-09 15:51:18,370 P61876 INFO [Metrics] logloss: 0.552795 - AUC: 0.788874
2022-03-09 15:51:18,371 P61876 INFO Save best model: monitor(max): 0.236079
2022-03-09 15:51:18,800 P61876 INFO --- 591/591 batches finished ---
2022-03-09 15:51:18,840 P61876 INFO Train loss: 0.607579
2022-03-09 15:51:18,840 P61876 INFO ************ Epoch=1 end ************
2022-03-09 15:59:04,815 P61876 INFO [Metrics] logloss: 0.540610 - AUC: 0.800924
2022-03-09 15:59:04,816 P61876 INFO Save best model: monitor(max): 0.260314
2022-03-09 15:59:05,051 P61876 INFO --- 591/591 batches finished ---
2022-03-09 15:59:05,092 P61876 INFO Train loss: 0.588530
2022-03-09 15:59:05,092 P61876 INFO ************ Epoch=2 end ************
2022-03-09 16:06:50,320 P61876 INFO [Metrics] logloss: 0.529565 - AUC: 0.809882
2022-03-09 16:06:50,320 P61876 INFO Save best model: monitor(max): 0.280317
2022-03-09 16:06:50,542 P61876 INFO --- 591/591 batches finished ---
2022-03-09 16:06:50,580 P61876 INFO Train loss: 0.580574
2022-03-09 16:06:50,580 P61876 INFO ************ Epoch=3 end ************
2022-03-09 16:14:36,066 P61876 INFO [Metrics] logloss: 0.526057 - AUC: 0.813616
2022-03-09 16:14:36,066 P61876 INFO Save best model: monitor(max): 0.287559
2022-03-09 16:14:36,271 P61876 INFO --- 591/591 batches finished ---
2022-03-09 16:14:36,306 P61876 INFO Train loss: 0.575939
2022-03-09 16:14:36,307 P61876 INFO ************ Epoch=4 end ************
2022-03-09 16:22:20,357 P61876 INFO [Metrics] logloss: 0.520769 - AUC: 0.817791
2022-03-09 16:22:20,358 P61876 INFO Save best model: monitor(max): 0.297022
2022-03-09 16:22:20,591 P61876 INFO --- 591/591 batches finished ---
2022-03-09 16:22:20,628 P61876 INFO Train loss: 0.573023
2022-03-09 16:22:20,628 P61876 INFO ************ Epoch=5 end ************
2022-03-09 16:30:06,140 P61876 INFO [Metrics] logloss: 0.517670 - AUC: 0.820090
2022-03-09 16:30:06,141 P61876 INFO Save best model: monitor(max): 0.302420
2022-03-09 16:30:06,350 P61876 INFO --- 591/591 batches finished ---
2022-03-09 16:30:06,386 P61876 INFO Train loss: 0.571028
2022-03-09 16:30:06,386 P61876 INFO ************ Epoch=6 end ************
2022-03-09 16:37:51,789 P61876 INFO [Metrics] logloss: 0.515325 - AUC: 0.822368
2022-03-09 16:37:51,789 P61876 INFO Save best model: monitor(max): 0.307043
2022-03-09 16:37:51,992 P61876 INFO --- 591/591 batches finished ---
2022-03-09 16:37:52,033 P61876 INFO Train loss: 0.569177
2022-03-09 16:37:52,033 P61876 INFO ************ Epoch=7 end ************
2022-03-09 16:45:37,424 P61876 INFO [Metrics] logloss: 0.512288 - AUC: 0.824651
2022-03-09 16:45:37,425 P61876 INFO Save best model: monitor(max): 0.312363
2022-03-09 16:45:37,649 P61876 INFO --- 591/591 batches finished ---
2022-03-09 16:45:37,685 P61876 INFO Train loss: 0.567642
2022-03-09 16:45:37,685 P61876 INFO ************ Epoch=8 end ************
2022-03-09 16:53:23,037 P61876 INFO [Metrics] logloss: 0.510271 - AUC: 0.826250
2022-03-09 16:53:23,038 P61876 INFO Save best model: monitor(max): 0.315979
2022-03-09 16:53:23,248 P61876 INFO --- 591/591 batches finished ---
2022-03-09 16:53:23,296 P61876 INFO Train loss: 0.566393
2022-03-09 16:53:23,297 P61876 INFO ************ Epoch=9 end ************
2022-03-09 17:01:08,817 P61876 INFO [Metrics] logloss: 0.508553 - AUC: 0.827523
2022-03-09 17:01:08,818 P61876 INFO Save best model: monitor(max): 0.318970
2022-03-09 17:01:09,024 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:01:09,061 P61876 INFO Train loss: 0.565111
2022-03-09 17:01:09,062 P61876 INFO ************ Epoch=10 end ************
2022-03-09 17:08:54,641 P61876 INFO [Metrics] logloss: 0.507281 - AUC: 0.828591
2022-03-09 17:08:54,642 P61876 INFO Save best model: monitor(max): 0.321309
2022-03-09 17:08:54,885 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:08:54,926 P61876 INFO Train loss: 0.564131
2022-03-09 17:08:54,926 P61876 INFO ************ Epoch=11 end ************
2022-03-09 17:16:39,882 P61876 INFO [Metrics] logloss: 0.505759 - AUC: 0.829700
2022-03-09 17:16:39,882 P61876 INFO Save best model: monitor(max): 0.323941
2022-03-09 17:16:40,101 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:16:40,143 P61876 INFO Train loss: 0.563110
2022-03-09 17:16:40,143 P61876 INFO ************ Epoch=12 end ************
2022-03-09 17:24:25,538 P61876 INFO [Metrics] logloss: 0.505432 - AUC: 0.830901
2022-03-09 17:24:25,539 P61876 INFO Save best model: monitor(max): 0.325469
2022-03-09 17:24:25,779 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:24:25,815 P61876 INFO Train loss: 0.562223
2022-03-09 17:24:25,815 P61876 INFO ************ Epoch=13 end ************
2022-03-09 17:32:11,167 P61876 INFO [Metrics] logloss: 0.502897 - AUC: 0.831963
2022-03-09 17:32:11,168 P61876 INFO Save best model: monitor(max): 0.329066
2022-03-09 17:32:11,372 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:32:11,413 P61876 INFO Train loss: 0.561455
2022-03-09 17:32:11,413 P61876 INFO ************ Epoch=14 end ************
2022-03-09 17:39:56,633 P61876 INFO [Metrics] logloss: 0.502594 - AUC: 0.832584
2022-03-09 17:39:56,634 P61876 INFO Save best model: monitor(max): 0.329990
2022-03-09 17:39:56,872 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:39:56,912 P61876 INFO Train loss: 0.560480
2022-03-09 17:39:56,913 P61876 INFO ************ Epoch=15 end ************
2022-03-09 17:47:42,320 P61876 INFO [Metrics] logloss: 0.501150 - AUC: 0.833446
2022-03-09 17:47:42,321 P61876 INFO Save best model: monitor(max): 0.332296
2022-03-09 17:47:42,527 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:47:42,565 P61876 INFO Train loss: 0.559930
2022-03-09 17:47:42,565 P61876 INFO ************ Epoch=16 end ************
2022-03-09 17:55:27,502 P61876 INFO [Metrics] logloss: 0.499701 - AUC: 0.834578
2022-03-09 17:55:27,503 P61876 INFO Save best model: monitor(max): 0.334878
2022-03-09 17:55:27,724 P61876 INFO --- 591/591 batches finished ---
2022-03-09 17:55:27,759 P61876 INFO Train loss: 0.559216
2022-03-09 17:55:27,760 P61876 INFO ************ Epoch=17 end ************
2022-03-09 18:03:13,021 P61876 INFO [Metrics] logloss: 0.498810 - AUC: 0.835198
2022-03-09 18:03:13,022 P61876 INFO Save best model: monitor(max): 0.336388
2022-03-09 18:03:13,221 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:03:13,265 P61876 INFO Train loss: 0.558398
2022-03-09 18:03:13,265 P61876 INFO ************ Epoch=18 end ************
2022-03-09 18:10:58,790 P61876 INFO [Metrics] logloss: 0.498322 - AUC: 0.835738
2022-03-09 18:10:58,791 P61876 INFO Save best model: monitor(max): 0.337416
2022-03-09 18:10:59,017 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:10:59,062 P61876 INFO Train loss: 0.557717
2022-03-09 18:10:59,062 P61876 INFO ************ Epoch=19 end ************
2022-03-09 18:18:44,268 P61876 INFO [Metrics] logloss: 0.496954 - AUC: 0.836706
2022-03-09 18:18:44,269 P61876 INFO Save best model: monitor(max): 0.339752
2022-03-09 18:18:44,513 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:18:44,549 P61876 INFO Train loss: 0.557003
2022-03-09 18:18:44,549 P61876 INFO ************ Epoch=20 end ************
2022-03-09 18:26:29,732 P61876 INFO [Metrics] logloss: 0.496633 - AUC: 0.836965
2022-03-09 18:26:29,732 P61876 INFO Save best model: monitor(max): 0.340333
2022-03-09 18:26:29,968 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:26:30,003 P61876 INFO Train loss: 0.556253
2022-03-09 18:26:30,003 P61876 INFO ************ Epoch=21 end ************
2022-03-09 18:34:14,904 P61876 INFO [Metrics] logloss: 0.496203 - AUC: 0.837383
2022-03-09 18:34:14,905 P61876 INFO Save best model: monitor(max): 0.341180
2022-03-09 18:34:15,128 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:34:15,163 P61876 INFO Train loss: 0.555427
2022-03-09 18:34:15,163 P61876 INFO ************ Epoch=22 end ************
2022-03-09 18:41:59,871 P61876 INFO [Metrics] logloss: 0.495612 - AUC: 0.837821
2022-03-09 18:41:59,872 P61876 INFO Save best model: monitor(max): 0.342209
2022-03-09 18:42:00,081 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:42:00,116 P61876 INFO Train loss: 0.555142
2022-03-09 18:42:00,116 P61876 INFO ************ Epoch=23 end ************
2022-03-09 18:49:45,401 P61876 INFO [Metrics] logloss: 0.494805 - AUC: 0.838481
2022-03-09 18:49:45,402 P61876 INFO Save best model: monitor(max): 0.343677
2022-03-09 18:49:45,630 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:49:45,667 P61876 INFO Train loss: 0.554462
2022-03-09 18:49:45,667 P61876 INFO ************ Epoch=24 end ************
2022-03-09 18:57:31,142 P61876 INFO [Metrics] logloss: 0.494395 - AUC: 0.838841
2022-03-09 18:57:31,143 P61876 INFO Save best model: monitor(max): 0.344446
2022-03-09 18:57:31,371 P61876 INFO --- 591/591 batches finished ---
2022-03-09 18:57:31,407 P61876 INFO Train loss: 0.553726
2022-03-09 18:57:31,408 P61876 INFO ************ Epoch=25 end ************
2022-03-09 19:05:16,570 P61876 INFO [Metrics] logloss: 0.494146 - AUC: 0.839109
2022-03-09 19:05:16,571 P61876 INFO Save best model: monitor(max): 0.344963
2022-03-09 19:05:16,802 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:05:16,838 P61876 INFO Train loss: 0.553084
2022-03-09 19:05:16,838 P61876 INFO ************ Epoch=26 end ************
2022-03-09 19:13:01,533 P61876 INFO [Metrics] logloss: 0.493506 - AUC: 0.839612
2022-03-09 19:13:01,534 P61876 INFO Save best model: monitor(max): 0.346106
2022-03-09 19:13:01,733 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:13:01,770 P61876 INFO Train loss: 0.552526
2022-03-09 19:13:01,770 P61876 INFO ************ Epoch=27 end ************
2022-03-09 19:20:47,124 P61876 INFO [Metrics] logloss: 0.493385 - AUC: 0.839825
2022-03-09 19:20:47,125 P61876 INFO Save best model: monitor(max): 0.346441
2022-03-09 19:20:47,337 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:20:47,371 P61876 INFO Train loss: 0.551838
2022-03-09 19:20:47,371 P61876 INFO ************ Epoch=28 end ************
2022-03-09 19:28:32,674 P61876 INFO [Metrics] logloss: 0.492520 - AUC: 0.840463
2022-03-09 19:28:32,675 P61876 INFO Save best model: monitor(max): 0.347943
2022-03-09 19:28:32,913 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:28:32,948 P61876 INFO Train loss: 0.551159
2022-03-09 19:28:32,948 P61876 INFO ************ Epoch=29 end ************
2022-03-09 19:36:18,231 P61876 INFO [Metrics] logloss: 0.493094 - AUC: 0.840309
2022-03-09 19:36:18,231 P61876 INFO Monitor(max) STOP: 0.347215 !
2022-03-09 19:36:18,232 P61876 INFO Reduce learning rate on plateau: 0.000100
2022-03-09 19:36:18,232 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:36:18,271 P61876 INFO Train loss: 0.550588
2022-03-09 19:36:18,271 P61876 INFO ************ Epoch=30 end ************
2022-03-09 19:44:03,201 P61876 INFO [Metrics] logloss: 0.480651 - AUC: 0.850301
2022-03-09 19:44:03,202 P61876 INFO Save best model: monitor(max): 0.369649
2022-03-09 19:44:03,428 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:44:03,462 P61876 INFO Train loss: 0.488372
2022-03-09 19:44:03,462 P61876 INFO ************ Epoch=31 end ************
2022-03-09 19:51:48,874 P61876 INFO [Metrics] logloss: 0.478446 - AUC: 0.852438
2022-03-09 19:51:48,874 P61876 INFO Save best model: monitor(max): 0.373993
2022-03-09 19:51:49,109 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:51:49,141 P61876 INFO Train loss: 0.458411
2022-03-09 19:51:49,141 P61876 INFO ************ Epoch=32 end ************
2022-03-09 19:59:34,500 P61876 INFO [Metrics] logloss: 0.478030 - AUC: 0.852982
2022-03-09 19:59:34,501 P61876 INFO Save best model: monitor(max): 0.374952
2022-03-09 19:59:34,765 P61876 INFO --- 591/591 batches finished ---
2022-03-09 19:59:34,797 P61876 INFO Train loss: 0.447719
2022-03-09 19:59:34,797 P61876 INFO ************ Epoch=33 end ************
2022-03-09 20:07:20,411 P61876 INFO [Metrics] logloss: 0.478809 - AUC: 0.853239
2022-03-09 20:07:20,412 P61876 INFO Monitor(max) STOP: 0.374430 !
2022-03-09 20:07:20,412 P61876 INFO Reduce learning rate on plateau: 0.000010
2022-03-09 20:07:20,412 P61876 INFO --- 591/591 batches finished ---
2022-03-09 20:07:20,444 P61876 INFO Train loss: 0.441009
2022-03-09 20:07:20,444 P61876 INFO ************ Epoch=34 end ************
2022-03-09 20:15:06,351 P61876 INFO [Metrics] logloss: 0.482409 - AUC: 0.852981
2022-03-09 20:15:06,352 P61876 INFO Monitor(max) STOP: 0.370572 !
2022-03-09 20:15:06,352 P61876 INFO Reduce learning rate on plateau: 0.000001
2022-03-09 20:15:06,352 P61876 INFO Early stopping at epoch=35
2022-03-09 20:15:06,352 P61876 INFO --- 591/591 batches finished ---
2022-03-09 20:15:06,385 P61876 INFO Train loss: 0.424059
2022-03-09 20:15:06,385 P61876 INFO Training finished.
2022-03-09 20:15:06,385 P61876 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/AutoInt_kkbox_x1/kkbox_x1_227d337d/AutoInt_kkbox_x1_003_df9d4ed6_model.ckpt
2022-03-09 20:15:06,642 P61876 INFO ****** Validation evaluation ******
2022-03-09 20:15:24,284 P61876 INFO [Metrics] logloss: 0.478030 - AUC: 0.852982
2022-03-09 20:15:24,333 P61876 INFO ******** Test evaluation ********
2022-03-09 20:15:24,333 P61876 INFO Loading data...
2022-03-09 20:15:24,333 P61876 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-09 20:15:24,406 P61876 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 20:15:24,406 P61876 INFO Loading test data done.
2022-03-09 20:15:41,997 P61876 INFO [Metrics] logloss: 0.477251 - AUC: 0.853443

```
