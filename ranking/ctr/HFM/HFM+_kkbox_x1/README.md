## HFM+_kkbox_x1

A hands-on guide to run the HFM model on the KKBox_x1 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [HFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_kkbox_x1_tuner_config_05](./HFM+_kkbox_x1_tuner_config_05). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_kkbox_x1
    nohup python run_expid.py --config ./HFM+_kkbox_x1_tuner_config_05 --expid HFM_kkbox_x1_019_7e9801c4 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.478074 | 0.852120  |


### Logs
```python
2022-03-11 17:31:39,130 P44577 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'str', 'name': ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via', 'language'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'genre_ids', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'encoder': 'MaskedSumPooling', 'max_len': 3, 'name': 'artist_name', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'name': 'isrc', 'preprocess': 'extract_country_code', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'bd', 'preprocess': 'bucketize_age', 'type': 'categorical'}]",
    "gpu": "6",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000, 1000]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "10",
    "model": "HFM",
    "model_id": "HFM_kkbox_x1_019_7e9801c4",
    "model_root": "./KKBox/HFM_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.5",
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
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/KKBox/KKBox_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2022-03-11 17:31:39,131 P44577 INFO Set up feature encoder...
2022-03-11 17:31:39,131 P44577 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-11 17:31:39,621 P44577 INFO Total number of parameters: 24888865.
2022-03-11 17:31:39,622 P44577 INFO Loading data...
2022-03-11 17:31:39,622 P44577 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-11 17:31:40,061 P44577 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-11 17:31:40,245 P44577 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-11 17:31:40,261 P44577 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 17:31:40,261 P44577 INFO Loading train data done.
2022-03-11 17:31:42,992 P44577 INFO Start training: 591 batches/epoch
2022-03-11 17:31:42,993 P44577 INFO ************ Epoch=1 start ************
2022-03-11 17:35:01,757 P44577 INFO [Metrics] logloss: 0.554090 - AUC: 0.788881
2022-03-11 17:35:01,762 P44577 INFO Save best model: monitor(max): 0.234791
2022-03-11 17:35:02,059 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:35:02,101 P44577 INFO Train loss: 0.617069
2022-03-11 17:35:02,101 P44577 INFO ************ Epoch=1 end ************
2022-03-11 17:38:20,576 P44577 INFO [Metrics] logloss: 0.541516 - AUC: 0.800558
2022-03-11 17:38:20,577 P44577 INFO Save best model: monitor(max): 0.259042
2022-03-11 17:38:20,691 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:38:20,722 P44577 INFO Train loss: 0.607724
2022-03-11 17:38:20,723 P44577 INFO ************ Epoch=2 end ************
2022-03-11 17:41:38,994 P44577 INFO [Metrics] logloss: 0.532929 - AUC: 0.808472
2022-03-11 17:41:38,995 P44577 INFO Save best model: monitor(max): 0.275544
2022-03-11 17:41:39,092 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:41:39,125 P44577 INFO Train loss: 0.605417
2022-03-11 17:41:39,125 P44577 INFO ************ Epoch=3 end ************
2022-03-11 17:44:57,689 P44577 INFO [Metrics] logloss: 0.525739 - AUC: 0.813979
2022-03-11 17:44:57,689 P44577 INFO Save best model: monitor(max): 0.288240
2022-03-11 17:44:57,824 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:44:57,859 P44577 INFO Train loss: 0.601949
2022-03-11 17:44:57,860 P44577 INFO ************ Epoch=4 end ************
2022-03-11 17:48:16,034 P44577 INFO [Metrics] logloss: 0.520375 - AUC: 0.818582
2022-03-11 17:48:16,034 P44577 INFO Save best model: monitor(max): 0.298207
2022-03-11 17:48:16,135 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:48:16,167 P44577 INFO Train loss: 0.599752
2022-03-11 17:48:16,167 P44577 INFO ************ Epoch=5 end ************
2022-03-11 17:51:34,338 P44577 INFO [Metrics] logloss: 0.517378 - AUC: 0.821153
2022-03-11 17:51:34,339 P44577 INFO Save best model: monitor(max): 0.303776
2022-03-11 17:51:34,437 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:51:34,479 P44577 INFO Train loss: 0.598448
2022-03-11 17:51:34,479 P44577 INFO ************ Epoch=6 end ************
2022-03-11 17:54:52,674 P44577 INFO [Metrics] logloss: 0.515804 - AUC: 0.822623
2022-03-11 17:54:52,675 P44577 INFO Save best model: monitor(max): 0.306819
2022-03-11 17:54:52,788 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:54:52,824 P44577 INFO Train loss: 0.597395
2022-03-11 17:54:52,824 P44577 INFO ************ Epoch=7 end ************
2022-03-11 17:58:10,958 P44577 INFO [Metrics] logloss: 0.513495 - AUC: 0.824008
2022-03-11 17:58:10,959 P44577 INFO Save best model: monitor(max): 0.310513
2022-03-11 17:58:11,056 P44577 INFO --- 591/591 batches finished ---
2022-03-11 17:58:11,091 P44577 INFO Train loss: 0.596272
2022-03-11 17:58:11,091 P44577 INFO ************ Epoch=8 end ************
2022-03-11 18:01:29,053 P44577 INFO [Metrics] logloss: 0.511471 - AUC: 0.825705
2022-03-11 18:01:29,054 P44577 INFO Save best model: monitor(max): 0.314234
2022-03-11 18:01:29,153 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:01:29,188 P44577 INFO Train loss: 0.596139
2022-03-11 18:01:29,189 P44577 INFO ************ Epoch=9 end ************
2022-03-11 18:04:47,290 P44577 INFO [Metrics] logloss: 0.509960 - AUC: 0.827010
2022-03-11 18:04:47,291 P44577 INFO Save best model: monitor(max): 0.317050
2022-03-11 18:04:47,413 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:04:47,449 P44577 INFO Train loss: 0.595989
2022-03-11 18:04:47,449 P44577 INFO ************ Epoch=10 end ************
2022-03-11 18:08:05,511 P44577 INFO [Metrics] logloss: 0.508402 - AUC: 0.828092
2022-03-11 18:08:05,511 P44577 INFO Save best model: monitor(max): 0.319690
2022-03-11 18:08:05,628 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:08:05,670 P44577 INFO Train loss: 0.595989
2022-03-11 18:08:05,670 P44577 INFO ************ Epoch=11 end ************
2022-03-11 18:11:23,866 P44577 INFO [Metrics] logloss: 0.507803 - AUC: 0.828544
2022-03-11 18:11:23,866 P44577 INFO Save best model: monitor(max): 0.320741
2022-03-11 18:11:23,965 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:11:24,007 P44577 INFO Train loss: 0.595904
2022-03-11 18:11:24,008 P44577 INFO ************ Epoch=12 end ************
2022-03-11 18:14:42,260 P44577 INFO [Metrics] logloss: 0.507046 - AUC: 0.829243
2022-03-11 18:14:42,261 P44577 INFO Save best model: monitor(max): 0.322197
2022-03-11 18:14:42,378 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:14:42,415 P44577 INFO Train loss: 0.595488
2022-03-11 18:14:42,415 P44577 INFO ************ Epoch=13 end ************
2022-03-11 18:18:00,508 P44577 INFO [Metrics] logloss: 0.505992 - AUC: 0.830128
2022-03-11 18:18:00,509 P44577 INFO Save best model: monitor(max): 0.324136
2022-03-11 18:18:00,607 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:18:00,642 P44577 INFO Train loss: 0.595699
2022-03-11 18:18:00,642 P44577 INFO ************ Epoch=14 end ************
2022-03-11 18:21:18,994 P44577 INFO [Metrics] logloss: 0.505391 - AUC: 0.830644
2022-03-11 18:21:18,995 P44577 INFO Save best model: monitor(max): 0.325252
2022-03-11 18:21:19,094 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:21:19,137 P44577 INFO Train loss: 0.595763
2022-03-11 18:21:19,137 P44577 INFO ************ Epoch=15 end ************
2022-03-11 18:24:37,370 P44577 INFO [Metrics] logloss: 0.504608 - AUC: 0.831176
2022-03-11 18:24:37,371 P44577 INFO Save best model: monitor(max): 0.326568
2022-03-11 18:24:37,723 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:24:37,762 P44577 INFO Train loss: 0.595634
2022-03-11 18:24:37,762 P44577 INFO ************ Epoch=16 end ************
2022-03-11 18:27:55,842 P44577 INFO [Metrics] logloss: 0.503468 - AUC: 0.831925
2022-03-11 18:27:55,842 P44577 INFO Save best model: monitor(max): 0.328457
2022-03-11 18:27:55,952 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:27:55,992 P44577 INFO Train loss: 0.595303
2022-03-11 18:27:55,992 P44577 INFO ************ Epoch=17 end ************
2022-03-11 18:31:14,172 P44577 INFO [Metrics] logloss: 0.503435 - AUC: 0.832099
2022-03-11 18:31:14,173 P44577 INFO Save best model: monitor(max): 0.328663
2022-03-11 18:31:14,275 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:31:14,312 P44577 INFO Train loss: 0.595585
2022-03-11 18:31:14,312 P44577 INFO ************ Epoch=18 end ************
2022-03-11 18:34:32,530 P44577 INFO [Metrics] logloss: 0.502003 - AUC: 0.832943
2022-03-11 18:34:32,530 P44577 INFO Save best model: monitor(max): 0.330939
2022-03-11 18:34:32,661 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:34:32,710 P44577 INFO Train loss: 0.595192
2022-03-11 18:34:32,710 P44577 INFO ************ Epoch=19 end ************
2022-03-11 18:37:50,872 P44577 INFO [Metrics] logloss: 0.501994 - AUC: 0.833014
2022-03-11 18:37:50,873 P44577 INFO Save best model: monitor(max): 0.331020
2022-03-11 18:37:50,970 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:37:51,004 P44577 INFO Train loss: 0.595631
2022-03-11 18:37:51,004 P44577 INFO ************ Epoch=20 end ************
2022-03-11 18:41:09,226 P44577 INFO [Metrics] logloss: 0.501823 - AUC: 0.833445
2022-03-11 18:41:09,227 P44577 INFO Save best model: monitor(max): 0.331622
2022-03-11 18:41:09,345 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:41:09,388 P44577 INFO Train loss: 0.595687
2022-03-11 18:41:09,388 P44577 INFO ************ Epoch=21 end ************
2022-03-11 18:44:27,527 P44577 INFO [Metrics] logloss: 0.500957 - AUC: 0.833913
2022-03-11 18:44:27,528 P44577 INFO Save best model: monitor(max): 0.332956
2022-03-11 18:44:27,648 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:44:27,690 P44577 INFO Train loss: 0.596042
2022-03-11 18:44:27,690 P44577 INFO ************ Epoch=22 end ************
2022-03-11 18:47:46,126 P44577 INFO [Metrics] logloss: 0.499827 - AUC: 0.834557
2022-03-11 18:47:46,126 P44577 INFO Save best model: monitor(max): 0.334731
2022-03-11 18:47:46,221 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:47:46,255 P44577 INFO Train loss: 0.595846
2022-03-11 18:47:46,255 P44577 INFO ************ Epoch=23 end ************
2022-03-11 18:51:04,629 P44577 INFO [Metrics] logloss: 0.501228 - AUC: 0.834163
2022-03-11 18:51:04,630 P44577 INFO Monitor(max) STOP: 0.332935 !
2022-03-11 18:51:04,630 P44577 INFO Reduce learning rate on plateau: 0.000100
2022-03-11 18:51:04,630 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:51:04,664 P44577 INFO Train loss: 0.595880
2022-03-11 18:51:04,664 P44577 INFO ************ Epoch=24 end ************
2022-03-11 18:54:22,706 P44577 INFO [Metrics] logloss: 0.480745 - AUC: 0.849256
2022-03-11 18:54:22,707 P44577 INFO Save best model: monitor(max): 0.368512
2022-03-11 18:54:22,805 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:54:22,839 P44577 INFO Train loss: 0.520182
2022-03-11 18:54:22,840 P44577 INFO ************ Epoch=25 end ************
2022-03-11 18:57:40,878 P44577 INFO [Metrics] logloss: 0.478366 - AUC: 0.851885
2022-03-11 18:57:40,878 P44577 INFO Save best model: monitor(max): 0.373518
2022-03-11 18:57:40,978 P44577 INFO --- 591/591 batches finished ---
2022-03-11 18:57:41,012 P44577 INFO Train loss: 0.475684
2022-03-11 18:57:41,013 P44577 INFO ************ Epoch=26 end ************
2022-03-11 19:00:59,241 P44577 INFO [Metrics] logloss: 0.482255 - AUC: 0.851770
2022-03-11 19:00:59,241 P44577 INFO Monitor(max) STOP: 0.369515 !
2022-03-11 19:00:59,241 P44577 INFO Reduce learning rate on plateau: 0.000010
2022-03-11 19:00:59,242 P44577 INFO --- 591/591 batches finished ---
2022-03-11 19:00:59,283 P44577 INFO Train loss: 0.454506
2022-03-11 19:00:59,283 P44577 INFO ************ Epoch=27 end ************
2022-03-11 19:04:17,356 P44577 INFO [Metrics] logloss: 0.503366 - AUC: 0.849270
2022-03-11 19:04:17,357 P44577 INFO Monitor(max) STOP: 0.345904 !
2022-03-11 19:04:17,357 P44577 INFO Reduce learning rate on plateau: 0.000001
2022-03-11 19:04:17,357 P44577 INFO Early stopping at epoch=28
2022-03-11 19:04:17,357 P44577 INFO --- 591/591 batches finished ---
2022-03-11 19:04:17,393 P44577 INFO Train loss: 0.413939
2022-03-11 19:04:17,393 P44577 INFO Training finished.
2022-03-11 19:04:17,393 P44577 INFO Load best model: /cache/FuxiCTR/benchmarks/KKBox/HFM_kkbox_x1/kkbox_x1_227d337d/HFM_kkbox_x1_019_7e9801c4_model.ckpt
2022-03-11 19:04:17,516 P44577 INFO ****** Validation evaluation ******
2022-03-11 19:04:25,940 P44577 INFO [Metrics] logloss: 0.478366 - AUC: 0.851885
2022-03-11 19:04:25,992 P44577 INFO ******** Test evaluation ********
2022-03-11 19:04:25,992 P44577 INFO Loading data...
2022-03-11 19:04:25,992 P44577 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-11 19:04:26,053 P44577 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-11 19:04:26,053 P44577 INFO Loading test data done.
2022-03-11 19:04:34,498 P44577 INFO [Metrics] logloss: 0.478074 - AUC: 0.852120

```
