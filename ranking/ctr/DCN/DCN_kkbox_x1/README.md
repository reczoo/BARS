## DCN_kkbox_x1

A hands-on guide to run the DCN model on the KKBox_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [KKBox_x1](https://github.com/reczoo/Datasets/tree/main/KKBox/KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [DCN_kkbox_x1_tuner_config_05](./DCN_kkbox_x1_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd DCN_kkbox_x1
    nohup python run_expid.py --config ./DCN_kkbox_x1_tuner_config_05 --expid DCN_kkbox_x1_005_362e6c13 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.476555 | 0.853114  |


### Logs
```python
2022-03-01 14:55:49,635 P30229 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "4",
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
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "DCN",
    "model_id": "DCN_kkbox_x1_005_362e6c13",
    "model_root": "./KKBox/DCN_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.4",
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
2022-03-01 14:55:49,635 P30229 INFO Set up feature encoder...
2022-03-01 14:55:49,635 P30229 INFO Reading file: ../data/KKBox/KKBox_x1/train.csv
2022-03-01 14:56:09,480 P30229 INFO Preprocess feature columns...
2022-03-01 14:56:26,044 P30229 INFO Fit feature encoder...
2022-03-01 14:56:26,045 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'msno', 'type': 'categorical'}
2022-03-01 14:56:27,463 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'song_id', 'type': 'categorical'}
2022-03-01 14:56:29,260 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'source_system_tab', 'type': 'categorical'}
2022-03-01 14:56:30,176 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'source_screen_name', 'type': 'categorical'}
2022-03-01 14:56:31,068 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'source_type', 'type': 'categorical'}
2022-03-01 14:56:31,987 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'city', 'type': 'categorical'}
2022-03-01 14:56:32,831 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'gender', 'type': 'categorical'}
2022-03-01 14:56:33,549 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'registered_via', 'type': 'categorical'}
2022-03-01 14:56:34,357 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'language', 'type': 'categorical'}
2022-03-01 14:56:35,218 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}
2022-03-01 14:56:45,163 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}
2022-03-01 14:56:55,444 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}
2022-03-01 14:56:56,635 P30229 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}
2022-03-01 14:56:57,305 P30229 INFO Set feature index...
2022-03-01 14:56:57,306 P30229 INFO Pickle feature_encode: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-01 14:56:57,342 P30229 INFO Save feature_map to json: ../data/KKBox/kkbox_x1_227d337d/feature_map.json
2022-03-01 14:56:57,343 P30229 INFO Set feature encoder done.
2022-03-01 14:56:59,560 P30229 INFO Total number of parameters: 45157593.
2022-03-01 14:56:59,560 P30229 INFO Loading data...
2022-03-01 14:56:59,563 P30229 INFO Reading file: ../data/KKBox/KKBox_x1/train.csv
2022-03-01 14:57:18,881 P30229 INFO Preprocess feature columns...
2022-03-01 14:57:35,322 P30229 INFO Transform feature columns...
2022-03-01 14:58:55,231 P30229 INFO Saving data to h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-01 14:58:56,228 P30229 INFO Reading file: ../data/KKBox/KKBox_x1/valid.csv
2022-03-01 14:58:58,606 P30229 INFO Preprocess feature columns...
2022-03-01 14:59:00,582 P30229 INFO Transform feature columns...
2022-03-01 14:59:10,020 P30229 INFO Saving data to h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-01 14:59:10,376 P30229 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-01 14:59:10,396 P30229 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-01 14:59:10,398 P30229 INFO Loading train data done.
2022-03-01 14:59:12,917 P30229 INFO Start training: 591 batches/epoch
2022-03-01 14:59:12,917 P30229 INFO ************ Epoch=1 start ************
2022-03-01 15:02:25,409 P30229 INFO [Metrics] logloss: 0.554382 - AUC: 0.787493
2022-03-01 15:02:25,409 P30229 INFO Save best model: monitor(max): 0.233111
2022-03-01 15:02:25,577 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:02:25,646 P30229 INFO Train loss: 0.607611
2022-03-01 15:02:25,646 P30229 INFO ************ Epoch=1 end ************
2022-03-01 15:05:37,834 P30229 INFO [Metrics] logloss: 0.541952 - AUC: 0.799543
2022-03-01 15:05:37,835 P30229 INFO Save best model: monitor(max): 0.257591
2022-03-01 15:05:38,178 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:05:38,256 P30229 INFO Train loss: 0.589109
2022-03-01 15:05:38,257 P30229 INFO ************ Epoch=2 end ************
2022-03-01 15:08:50,506 P30229 INFO [Metrics] logloss: 0.533545 - AUC: 0.807026
2022-03-01 15:08:50,506 P30229 INFO Save best model: monitor(max): 0.273481
2022-03-01 15:08:50,825 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:08:50,912 P30229 INFO Train loss: 0.583545
2022-03-01 15:08:50,912 P30229 INFO ************ Epoch=3 end ************
2022-03-01 15:12:03,105 P30229 INFO [Metrics] logloss: 0.527800 - AUC: 0.811652
2022-03-01 15:12:03,106 P30229 INFO Save best model: monitor(max): 0.283852
2022-03-01 15:12:03,422 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:12:03,507 P30229 INFO Train loss: 0.578663
2022-03-01 15:12:03,507 P30229 INFO ************ Epoch=4 end ************
2022-03-01 15:15:15,642 P30229 INFO [Metrics] logloss: 0.523498 - AUC: 0.815761
2022-03-01 15:15:15,643 P30229 INFO Save best model: monitor(max): 0.292264
2022-03-01 15:15:15,969 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:15:16,051 P30229 INFO Train loss: 0.575274
2022-03-01 15:15:16,051 P30229 INFO ************ Epoch=5 end ************
2022-03-01 15:18:28,035 P30229 INFO [Metrics] logloss: 0.520412 - AUC: 0.818221
2022-03-01 15:18:28,036 P30229 INFO Save best model: monitor(max): 0.297809
2022-03-01 15:18:28,367 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:18:28,450 P30229 INFO Train loss: 0.572853
2022-03-01 15:18:28,450 P30229 INFO ************ Epoch=6 end ************
2022-03-01 15:21:40,373 P30229 INFO [Metrics] logloss: 0.517348 - AUC: 0.820593
2022-03-01 15:21:40,374 P30229 INFO Save best model: monitor(max): 0.303245
2022-03-01 15:21:40,695 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:21:40,757 P30229 INFO Train loss: 0.571217
2022-03-01 15:21:40,757 P30229 INFO ************ Epoch=7 end ************
2022-03-01 15:24:52,684 P30229 INFO [Metrics] logloss: 0.514184 - AUC: 0.823019
2022-03-01 15:24:52,685 P30229 INFO Save best model: monitor(max): 0.308835
2022-03-01 15:24:53,007 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:24:53,078 P30229 INFO Train loss: 0.569733
2022-03-01 15:24:53,079 P30229 INFO ************ Epoch=8 end ************
2022-03-01 15:28:04,925 P30229 INFO [Metrics] logloss: 0.512940 - AUC: 0.824105
2022-03-01 15:28:04,926 P30229 INFO Save best model: monitor(max): 0.311165
2022-03-01 15:28:05,248 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:28:05,316 P30229 INFO Train loss: 0.568352
2022-03-01 15:28:05,316 P30229 INFO ************ Epoch=9 end ************
2022-03-01 15:31:17,175 P30229 INFO [Metrics] logloss: 0.511948 - AUC: 0.825831
2022-03-01 15:31:17,176 P30229 INFO Save best model: monitor(max): 0.313883
2022-03-01 15:31:17,510 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:31:17,568 P30229 INFO Train loss: 0.567793
2022-03-01 15:31:17,568 P30229 INFO ************ Epoch=10 end ************
2022-03-01 15:34:29,440 P30229 INFO [Metrics] logloss: 0.509772 - AUC: 0.826508
2022-03-01 15:34:29,441 P30229 INFO Save best model: monitor(max): 0.316736
2022-03-01 15:34:29,768 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:34:29,834 P30229 INFO Train loss: 0.566824
2022-03-01 15:34:29,834 P30229 INFO ************ Epoch=11 end ************
2022-03-01 15:37:41,633 P30229 INFO [Metrics] logloss: 0.507963 - AUC: 0.828204
2022-03-01 15:37:41,634 P30229 INFO Save best model: monitor(max): 0.320240
2022-03-01 15:37:41,962 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:37:42,030 P30229 INFO Train loss: 0.566042
2022-03-01 15:37:42,030 P30229 INFO ************ Epoch=12 end ************
2022-03-01 15:40:53,707 P30229 INFO [Metrics] logloss: 0.507060 - AUC: 0.828721
2022-03-01 15:40:53,708 P30229 INFO Save best model: monitor(max): 0.321662
2022-03-01 15:40:54,042 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:40:54,109 P30229 INFO Train loss: 0.565228
2022-03-01 15:40:54,109 P30229 INFO ************ Epoch=13 end ************
2022-03-01 15:44:05,893 P30229 INFO [Metrics] logloss: 0.505816 - AUC: 0.829752
2022-03-01 15:44:05,893 P30229 INFO Save best model: monitor(max): 0.323936
2022-03-01 15:44:06,219 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:44:06,287 P30229 INFO Train loss: 0.564551
2022-03-01 15:44:06,287 P30229 INFO ************ Epoch=14 end ************
2022-03-01 15:47:18,108 P30229 INFO [Metrics] logloss: 0.504040 - AUC: 0.831068
2022-03-01 15:47:18,109 P30229 INFO Save best model: monitor(max): 0.327028
2022-03-01 15:47:18,443 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:47:18,505 P30229 INFO Train loss: 0.564035
2022-03-01 15:47:18,505 P30229 INFO ************ Epoch=15 end ************
2022-03-01 15:50:30,403 P30229 INFO [Metrics] logloss: 0.504139 - AUC: 0.831387
2022-03-01 15:50:30,404 P30229 INFO Save best model: monitor(max): 0.327248
2022-03-01 15:50:31,026 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:50:31,093 P30229 INFO Train loss: 0.563480
2022-03-01 15:50:31,093 P30229 INFO ************ Epoch=16 end ************
2022-03-01 15:53:42,788 P30229 INFO [Metrics] logloss: 0.502425 - AUC: 0.832447
2022-03-01 15:53:42,788 P30229 INFO Save best model: monitor(max): 0.330022
2022-03-01 15:53:43,133 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:53:43,196 P30229 INFO Train loss: 0.563114
2022-03-01 15:53:43,196 P30229 INFO ************ Epoch=17 end ************
2022-03-01 15:56:55,091 P30229 INFO [Metrics] logloss: 0.501691 - AUC: 0.832845
2022-03-01 15:56:55,092 P30229 INFO Save best model: monitor(max): 0.331154
2022-03-01 15:56:55,432 P30229 INFO --- 591/591 batches finished ---
2022-03-01 15:56:55,502 P30229 INFO Train loss: 0.562719
2022-03-01 15:56:55,502 P30229 INFO ************ Epoch=18 end ************
2022-03-01 16:00:07,992 P30229 INFO [Metrics] logloss: 0.501307 - AUC: 0.833250
2022-03-01 16:00:07,992 P30229 INFO Save best model: monitor(max): 0.331943
2022-03-01 16:00:08,357 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:00:08,422 P30229 INFO Train loss: 0.562152
2022-03-01 16:00:08,424 P30229 INFO ************ Epoch=19 end ************
2022-03-01 16:03:20,365 P30229 INFO [Metrics] logloss: 0.499900 - AUC: 0.834338
2022-03-01 16:03:20,365 P30229 INFO Save best model: monitor(max): 0.334438
2022-03-01 16:03:20,687 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:03:20,758 P30229 INFO Train loss: 0.561888
2022-03-01 16:03:20,758 P30229 INFO ************ Epoch=20 end ************
2022-03-01 16:06:32,429 P30229 INFO [Metrics] logloss: 0.499997 - AUC: 0.834470
2022-03-01 16:06:32,430 P30229 INFO Save best model: monitor(max): 0.334473
2022-03-01 16:06:32,760 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:06:32,821 P30229 INFO Train loss: 0.561406
2022-03-01 16:06:32,821 P30229 INFO ************ Epoch=21 end ************
2022-03-01 16:09:44,765 P30229 INFO [Metrics] logloss: 0.499577 - AUC: 0.834811
2022-03-01 16:09:44,766 P30229 INFO Save best model: monitor(max): 0.335234
2022-03-01 16:09:45,101 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:09:45,163 P30229 INFO Train loss: 0.561064
2022-03-01 16:09:45,163 P30229 INFO ************ Epoch=22 end ************
2022-03-01 16:12:56,915 P30229 INFO [Metrics] logloss: 0.497946 - AUC: 0.835879
2022-03-01 16:12:56,916 P30229 INFO Save best model: monitor(max): 0.337933
2022-03-01 16:12:57,248 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:12:57,306 P30229 INFO Train loss: 0.560738
2022-03-01 16:12:57,306 P30229 INFO ************ Epoch=23 end ************
2022-03-01 16:16:08,992 P30229 INFO [Metrics] logloss: 0.497622 - AUC: 0.836163
2022-03-01 16:16:08,993 P30229 INFO Save best model: monitor(max): 0.338541
2022-03-01 16:16:09,327 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:16:09,388 P30229 INFO Train loss: 0.560372
2022-03-01 16:16:09,388 P30229 INFO ************ Epoch=24 end ************
2022-03-01 16:19:21,153 P30229 INFO [Metrics] logloss: 0.497501 - AUC: 0.836384
2022-03-01 16:19:21,154 P30229 INFO Save best model: monitor(max): 0.338883
2022-03-01 16:19:21,485 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:19:21,557 P30229 INFO Train loss: 0.560137
2022-03-01 16:19:21,557 P30229 INFO ************ Epoch=25 end ************
2022-03-01 16:22:33,369 P30229 INFO [Metrics] logloss: 0.496910 - AUC: 0.836766
2022-03-01 16:22:33,370 P30229 INFO Save best model: monitor(max): 0.339856
2022-03-01 16:22:33,708 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:22:33,768 P30229 INFO Train loss: 0.559929
2022-03-01 16:22:33,768 P30229 INFO ************ Epoch=26 end ************
2022-03-01 16:25:45,485 P30229 INFO [Metrics] logloss: 0.496366 - AUC: 0.837202
2022-03-01 16:25:45,486 P30229 INFO Save best model: monitor(max): 0.340836
2022-03-01 16:25:45,820 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:25:45,891 P30229 INFO Train loss: 0.559461
2022-03-01 16:25:45,891 P30229 INFO ************ Epoch=27 end ************
2022-03-01 16:28:57,783 P30229 INFO [Metrics] logloss: 0.496371 - AUC: 0.837372
2022-03-01 16:28:57,784 P30229 INFO Save best model: monitor(max): 0.341001
2022-03-01 16:28:58,126 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:28:58,184 P30229 INFO Train loss: 0.559289
2022-03-01 16:28:58,184 P30229 INFO ************ Epoch=28 end ************
2022-03-01 16:32:09,943 P30229 INFO [Metrics] logloss: 0.495896 - AUC: 0.837699
2022-03-01 16:32:09,944 P30229 INFO Save best model: monitor(max): 0.341804
2022-03-01 16:32:10,285 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:32:10,357 P30229 INFO Train loss: 0.559001
2022-03-01 16:32:10,357 P30229 INFO ************ Epoch=29 end ************
2022-03-01 16:35:22,114 P30229 INFO [Metrics] logloss: 0.494787 - AUC: 0.838529
2022-03-01 16:35:22,114 P30229 INFO Save best model: monitor(max): 0.343743
2022-03-01 16:35:22,436 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:35:22,506 P30229 INFO Train loss: 0.558540
2022-03-01 16:35:22,506 P30229 INFO ************ Epoch=30 end ************
2022-03-01 16:38:34,327 P30229 INFO [Metrics] logloss: 0.494936 - AUC: 0.838524
2022-03-01 16:38:34,328 P30229 INFO Monitor(max) STOP: 0.343588 !
2022-03-01 16:38:34,328 P30229 INFO Reduce learning rate on plateau: 0.000100
2022-03-01 16:38:34,328 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:38:34,396 P30229 INFO Train loss: 0.558250
2022-03-01 16:38:34,396 P30229 INFO ************ Epoch=31 end ************
2022-03-01 16:41:46,312 P30229 INFO [Metrics] logloss: 0.480445 - AUC: 0.849283
2022-03-01 16:41:46,313 P30229 INFO Save best model: monitor(max): 0.368838
2022-03-01 16:41:46,670 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:41:46,739 P30229 INFO Train loss: 0.500224
2022-03-01 16:41:46,739 P30229 INFO ************ Epoch=32 end ************
2022-03-01 16:44:58,517 P30229 INFO [Metrics] logloss: 0.477445 - AUC: 0.851704
2022-03-01 16:44:58,517 P30229 INFO Save best model: monitor(max): 0.374258
2022-03-01 16:44:58,875 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:44:58,946 P30229 INFO Train loss: 0.472289
2022-03-01 16:44:58,946 P30229 INFO ************ Epoch=33 end ************
2022-03-01 16:48:10,869 P30229 INFO [Metrics] logloss: 0.476608 - AUC: 0.852613
2022-03-01 16:48:10,869 P30229 INFO Save best model: monitor(max): 0.376005
2022-03-01 16:48:11,218 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:48:11,315 P30229 INFO Train loss: 0.461926
2022-03-01 16:48:11,315 P30229 INFO ************ Epoch=34 end ************
2022-03-01 16:51:23,103 P30229 INFO [Metrics] logloss: 0.476696 - AUC: 0.852906
2022-03-01 16:51:23,104 P30229 INFO Save best model: monitor(max): 0.376210
2022-03-01 16:51:23,443 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:51:23,513 P30229 INFO Train loss: 0.455410
2022-03-01 16:51:23,513 P30229 INFO ************ Epoch=35 end ************
2022-03-01 16:54:35,285 P30229 INFO [Metrics] logloss: 0.476966 - AUC: 0.852999
2022-03-01 16:54:35,285 P30229 INFO Monitor(max) STOP: 0.376033 !
2022-03-01 16:54:35,285 P30229 INFO Reduce learning rate on plateau: 0.000010
2022-03-01 16:54:35,285 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:54:35,354 P30229 INFO Train loss: 0.450437
2022-03-01 16:54:35,354 P30229 INFO ************ Epoch=36 end ************
2022-03-01 16:57:47,179 P30229 INFO [Metrics] logloss: 0.481409 - AUC: 0.852600
2022-03-01 16:57:47,179 P30229 INFO Monitor(max) STOP: 0.371191 !
2022-03-01 16:57:47,179 P30229 INFO Reduce learning rate on plateau: 0.000001
2022-03-01 16:57:47,179 P30229 INFO Early stopping at epoch=37
2022-03-01 16:57:47,179 P30229 INFO --- 591/591 batches finished ---
2022-03-01 16:57:47,248 P30229 INFO Train loss: 0.432247
2022-03-01 16:57:47,248 P30229 INFO Training finished.
2022-03-01 16:57:47,248 P30229 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/KKBox/DCN_kkbox_x1/kkbox_x1_227d337d/DCN_kkbox_x1_005_362e6c13_model.ckpt
2022-03-01 16:57:47,480 P30229 INFO ****** Validation evaluation ******
2022-03-01 16:57:55,421 P30229 INFO [Metrics] logloss: 0.476696 - AUC: 0.852906
2022-03-01 16:57:55,456 P30229 INFO ******** Test evaluation ********
2022-03-01 16:57:55,456 P30229 INFO Loading data...
2022-03-01 16:57:55,456 P30229 INFO Reading file: ../data/KKBox/KKBox_x1/test.csv
2022-03-01 16:57:57,770 P30229 INFO Preprocess feature columns...
2022-03-01 16:57:59,698 P30229 INFO Transform feature columns...
2022-03-01 16:58:09,732 P30229 INFO Saving data to h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-01 16:58:09,871 P30229 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-01 16:58:09,871 P30229 INFO Loading test data done.
2022-03-01 16:58:17,807 P30229 INFO [Metrics] logloss: 0.476555 - AUC: 0.853114

```
