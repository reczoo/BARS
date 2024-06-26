## xDeepFM_kkbox_x1

A hands-on guide to run the xDeepFM model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_kkbox_x1_tuner_config_03](./xDeepFM_kkbox_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_kkbox_x1
    nohup python run_expid.py --config ./xDeepFM_kkbox_x1_tuner_config_03 --expid xDeepFM_kkbox_x1_019_49197bdf --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.477164 | 0.853498  |


### Logs
```python
2022-03-07 23:04:28,010 P739 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[16, 16]",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "dnn_hidden_units": "[5000, 5000]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "2",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "xDeepFM",
    "model_id": "xDeepFM_kkbox_x1_019_49197bdf",
    "model_root": "./KKBox/xDeepFM_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
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
2022-03-07 23:04:28,010 P739 INFO Set up feature encoder...
2022-03-07 23:04:28,010 P739 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-07 23:04:30,426 P739 INFO Total number of parameters: 45240961.
2022-03-07 23:04:30,426 P739 INFO Loading data...
2022-03-07 23:04:30,427 P739 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-07 23:04:30,862 P739 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-07 23:04:31,087 P739 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-07 23:04:31,112 P739 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-07 23:04:31,112 P739 INFO Loading train data done.
2022-03-07 23:04:37,187 P739 INFO Start training: 591 batches/epoch
2022-03-07 23:04:37,188 P739 INFO ************ Epoch=1 start ************
2022-03-07 23:32:22,084 P739 INFO [Metrics] logloss: 0.558959 - AUC: 0.786229
2022-03-07 23:32:22,085 P739 INFO Save best model: monitor(max): 0.227270
2022-03-07 23:32:22,721 P739 INFO --- 591/591 batches finished ---
2022-03-07 23:32:22,750 P739 INFO Train loss: 0.622283
2022-03-07 23:32:22,750 P739 INFO ************ Epoch=1 end ************
2022-03-08 00:00:01,972 P739 INFO [Metrics] logloss: 0.539579 - AUC: 0.801225
2022-03-08 00:00:01,973 P739 INFO Save best model: monitor(max): 0.261646
2022-03-08 00:00:02,236 P739 INFO --- 591/591 batches finished ---
2022-03-08 00:00:02,269 P739 INFO Train loss: 0.591130
2022-03-08 00:00:02,270 P739 INFO ************ Epoch=2 end ************
2022-03-08 00:27:41,983 P739 INFO [Metrics] logloss: 0.533481 - AUC: 0.808561
2022-03-08 00:27:41,984 P739 INFO Save best model: monitor(max): 0.275081
2022-03-08 00:27:42,337 P739 INFO --- 591/591 batches finished ---
2022-03-08 00:27:42,383 P739 INFO Train loss: 0.582633
2022-03-08 00:27:42,383 P739 INFO ************ Epoch=3 end ************
2022-03-08 00:55:21,825 P739 INFO [Metrics] logloss: 0.526002 - AUC: 0.813082
2022-03-08 00:55:21,826 P739 INFO Save best model: monitor(max): 0.287080
2022-03-08 00:55:22,185 P739 INFO --- 591/591 batches finished ---
2022-03-08 00:55:22,219 P739 INFO Train loss: 0.578150
2022-03-08 00:55:22,219 P739 INFO ************ Epoch=4 end ************
2022-03-08 01:23:07,021 P739 INFO [Metrics] logloss: 0.521693 - AUC: 0.816944
2022-03-08 01:23:07,022 P739 INFO Save best model: monitor(max): 0.295251
2022-03-08 01:23:07,284 P739 INFO --- 591/591 batches finished ---
2022-03-08 01:23:07,318 P739 INFO Train loss: 0.575274
2022-03-08 01:23:07,318 P739 INFO ************ Epoch=5 end ************
2022-03-08 01:50:46,728 P739 INFO [Metrics] logloss: 0.517654 - AUC: 0.820212
2022-03-08 01:50:46,729 P739 INFO Save best model: monitor(max): 0.302558
2022-03-08 01:50:47,074 P739 INFO --- 591/591 batches finished ---
2022-03-08 01:50:47,106 P739 INFO Train loss: 0.572756
2022-03-08 01:50:47,107 P739 INFO ************ Epoch=6 end ************
2022-03-08 02:18:26,857 P739 INFO [Metrics] logloss: 0.515106 - AUC: 0.822463
2022-03-08 02:18:26,857 P739 INFO Save best model: monitor(max): 0.307357
2022-03-08 02:18:27,232 P739 INFO --- 591/591 batches finished ---
2022-03-08 02:18:27,280 P739 INFO Train loss: 0.571374
2022-03-08 02:18:27,280 P739 INFO ************ Epoch=7 end ************
2022-03-08 02:46:07,267 P739 INFO [Metrics] logloss: 0.512798 - AUC: 0.824393
2022-03-08 02:46:07,267 P739 INFO Save best model: monitor(max): 0.311595
2022-03-08 02:46:07,587 P739 INFO --- 591/591 batches finished ---
2022-03-08 02:46:07,632 P739 INFO Train loss: 0.569367
2022-03-08 02:46:07,633 P739 INFO ************ Epoch=8 end ************
2022-03-08 03:13:47,337 P739 INFO [Metrics] logloss: 0.511204 - AUC: 0.825999
2022-03-08 03:13:47,338 P739 INFO Save best model: monitor(max): 0.314795
2022-03-08 03:13:47,606 P739 INFO --- 591/591 batches finished ---
2022-03-08 03:13:47,640 P739 INFO Train loss: 0.568263
2022-03-08 03:13:47,640 P739 INFO ************ Epoch=9 end ************
2022-03-08 03:41:32,566 P739 INFO [Metrics] logloss: 0.508502 - AUC: 0.827712
2022-03-08 03:41:32,567 P739 INFO Save best model: monitor(max): 0.319210
2022-03-08 03:41:32,883 P739 INFO --- 591/591 batches finished ---
2022-03-08 03:41:32,917 P739 INFO Train loss: 0.567648
2022-03-08 03:41:32,917 P739 INFO ************ Epoch=10 end ************
2022-03-08 04:09:12,290 P739 INFO [Metrics] logloss: 0.506667 - AUC: 0.829065
2022-03-08 04:09:12,291 P739 INFO Save best model: monitor(max): 0.322398
2022-03-08 04:09:12,594 P739 INFO --- 591/591 batches finished ---
2022-03-08 04:09:12,630 P739 INFO Train loss: 0.566489
2022-03-08 04:09:12,630 P739 INFO ************ Epoch=11 end ************
2022-03-08 04:36:51,998 P739 INFO [Metrics] logloss: 0.507068 - AUC: 0.830118
2022-03-08 04:36:51,998 P739 INFO Save best model: monitor(max): 0.323051
2022-03-08 04:36:52,208 P739 INFO --- 591/591 batches finished ---
2022-03-08 04:36:52,243 P739 INFO Train loss: 0.565602
2022-03-08 04:36:52,244 P739 INFO ************ Epoch=12 end ************
2022-03-08 05:04:31,561 P739 INFO [Metrics] logloss: 0.505372 - AUC: 0.831150
2022-03-08 05:04:31,562 P739 INFO Save best model: monitor(max): 0.325778
2022-03-08 05:04:31,945 P739 INFO --- 591/591 batches finished ---
2022-03-08 05:04:31,994 P739 INFO Train loss: 0.564694
2022-03-08 05:04:31,994 P739 INFO ************ Epoch=13 end ************
2022-03-08 05:32:16,602 P739 INFO [Metrics] logloss: 0.504292 - AUC: 0.832490
2022-03-08 05:32:16,602 P739 INFO Save best model: monitor(max): 0.328198
2022-03-08 05:32:16,797 P739 INFO --- 591/591 batches finished ---
2022-03-08 05:32:16,832 P739 INFO Train loss: 0.564182
2022-03-08 05:32:16,832 P739 INFO ************ Epoch=14 end ************
2022-03-08 05:59:55,866 P739 INFO [Metrics] logloss: 0.503031 - AUC: 0.833486
2022-03-08 05:59:55,866 P739 INFO Save best model: monitor(max): 0.330456
2022-03-08 05:59:56,106 P739 INFO --- 591/591 batches finished ---
2022-03-08 05:59:56,142 P739 INFO Train loss: 0.563696
2022-03-08 05:59:56,142 P739 INFO ************ Epoch=15 end ************
2022-03-08 06:11:59,552 P739 INFO [Metrics] logloss: 0.500163 - AUC: 0.834112
2022-03-08 06:11:59,553 P739 INFO Save best model: monitor(max): 0.333949
2022-03-08 06:11:59,765 P739 INFO --- 591/591 batches finished ---
2022-03-08 06:11:59,799 P739 INFO Train loss: 0.563183
2022-03-08 06:11:59,799 P739 INFO ************ Epoch=16 end ************
2022-03-08 06:20:19,282 P739 INFO [Metrics] logloss: 0.500334 - AUC: 0.834851
2022-03-08 06:20:19,283 P739 INFO Save best model: monitor(max): 0.334517
2022-03-08 06:20:19,476 P739 INFO --- 591/591 batches finished ---
2022-03-08 06:20:19,523 P739 INFO Train loss: 0.562410
2022-03-08 06:20:19,523 P739 INFO ************ Epoch=17 end ************
2022-03-08 06:28:39,080 P739 INFO [Metrics] logloss: 0.498578 - AUC: 0.835637
2022-03-08 06:28:39,080 P739 INFO Save best model: monitor(max): 0.337059
2022-03-08 06:28:39,269 P739 INFO --- 591/591 batches finished ---
2022-03-08 06:28:39,303 P739 INFO Train loss: 0.561856
2022-03-08 06:28:39,304 P739 INFO ************ Epoch=18 end ************
2022-03-08 06:36:59,120 P739 INFO [Metrics] logloss: 0.498117 - AUC: 0.836116
2022-03-08 06:36:59,121 P739 INFO Save best model: monitor(max): 0.337999
2022-03-08 06:36:59,327 P739 INFO --- 591/591 batches finished ---
2022-03-08 06:36:59,362 P739 INFO Train loss: 0.561534
2022-03-08 06:36:59,363 P739 INFO ************ Epoch=19 end ************
2022-03-08 06:45:19,103 P739 INFO [Metrics] logloss: 0.496880 - AUC: 0.836808
2022-03-08 06:45:19,104 P739 INFO Save best model: monitor(max): 0.339928
2022-03-08 06:45:19,311 P739 INFO --- 591/591 batches finished ---
2022-03-08 06:45:19,356 P739 INFO Train loss: 0.561064
2022-03-08 06:45:19,356 P739 INFO ************ Epoch=20 end ************
2022-03-08 06:53:39,159 P739 INFO [Metrics] logloss: 0.496280 - AUC: 0.837490
2022-03-08 06:53:39,160 P739 INFO Save best model: monitor(max): 0.341209
2022-03-08 06:53:39,355 P739 INFO --- 591/591 batches finished ---
2022-03-08 06:53:39,394 P739 INFO Train loss: 0.560300
2022-03-08 06:53:39,394 P739 INFO ************ Epoch=21 end ************
2022-03-08 07:01:59,332 P739 INFO [Metrics] logloss: 0.499009 - AUC: 0.837853
2022-03-08 07:01:59,333 P739 INFO Monitor(max) STOP: 0.338844 !
2022-03-08 07:01:59,333 P739 INFO Reduce learning rate on plateau: 0.000100
2022-03-08 07:01:59,333 P739 INFO --- 591/591 batches finished ---
2022-03-08 07:01:59,366 P739 INFO Train loss: 0.559728
2022-03-08 07:01:59,366 P739 INFO ************ Epoch=22 end ************
2022-03-08 07:10:19,047 P739 INFO [Metrics] logloss: 0.481111 - AUC: 0.849433
2022-03-08 07:10:19,048 P739 INFO Save best model: monitor(max): 0.368322
2022-03-08 07:10:19,232 P739 INFO --- 591/591 batches finished ---
2022-03-08 07:10:19,267 P739 INFO Train loss: 0.496601
2022-03-08 07:10:19,267 P739 INFO ************ Epoch=23 end ************
2022-03-08 07:18:39,155 P739 INFO [Metrics] logloss: 0.478168 - AUC: 0.851861
2022-03-08 07:18:39,156 P739 INFO Save best model: monitor(max): 0.373693
2022-03-08 07:18:39,345 P739 INFO --- 591/591 batches finished ---
2022-03-08 07:18:39,388 P739 INFO Train loss: 0.467108
2022-03-08 07:18:39,388 P739 INFO ************ Epoch=24 end ************
2022-03-08 07:26:59,059 P739 INFO [Metrics] logloss: 0.478436 - AUC: 0.852534
2022-03-08 07:26:59,060 P739 INFO Save best model: monitor(max): 0.374098
2022-03-08 07:26:59,275 P739 INFO --- 591/591 batches finished ---
2022-03-08 07:26:59,318 P739 INFO Train loss: 0.456161
2022-03-08 07:26:59,318 P739 INFO ************ Epoch=25 end ************
2022-03-08 07:35:19,013 P739 INFO [Metrics] logloss: 0.477735 - AUC: 0.853019
2022-03-08 07:35:19,013 P739 INFO Save best model: monitor(max): 0.375285
2022-03-08 07:35:19,204 P739 INFO --- 591/591 batches finished ---
2022-03-08 07:35:19,236 P739 INFO Train loss: 0.449156
2022-03-08 07:35:19,236 P739 INFO ************ Epoch=26 end ************
2022-03-08 07:43:38,804 P739 INFO [Metrics] logloss: 0.478912 - AUC: 0.853216
2022-03-08 07:43:38,805 P739 INFO Monitor(max) STOP: 0.374303 !
2022-03-08 07:43:38,805 P739 INFO Reduce learning rate on plateau: 0.000010
2022-03-08 07:43:38,805 P739 INFO --- 591/591 batches finished ---
2022-03-08 07:43:38,847 P739 INFO Train loss: 0.443388
2022-03-08 07:43:38,847 P739 INFO ************ Epoch=27 end ************
2022-03-08 07:51:58,587 P739 INFO [Metrics] logloss: 0.483465 - AUC: 0.852744
2022-03-08 07:51:58,588 P739 INFO Monitor(max) STOP: 0.369279 !
2022-03-08 07:51:58,588 P739 INFO Reduce learning rate on plateau: 0.000001
2022-03-08 07:51:58,588 P739 INFO Early stopping at epoch=28
2022-03-08 07:51:58,588 P739 INFO --- 591/591 batches finished ---
2022-03-08 07:51:58,619 P739 INFO Train loss: 0.422383
2022-03-08 07:51:58,619 P739 INFO Training finished.
2022-03-08 07:51:58,619 P739 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/xDeepFM_kkbox_x1/kkbox_x1_227d337d/xDeepFM_kkbox_x1_019_49197bdf_model.ckpt
2022-03-08 07:51:58,934 P739 INFO ****** Validation evaluation ******
2022-03-08 07:52:04,954 P739 INFO [Metrics] logloss: 0.477735 - AUC: 0.853019
2022-03-08 07:52:05,001 P739 INFO ******** Test evaluation ********
2022-03-08 07:52:05,002 P739 INFO Loading data...
2022-03-08 07:52:05,002 P739 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-08 07:52:05,075 P739 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-08 07:52:05,075 P739 INFO Loading test data done.
2022-03-08 07:52:11,003 P739 INFO [Metrics] logloss: 0.477164 - AUC: 0.853498

```
