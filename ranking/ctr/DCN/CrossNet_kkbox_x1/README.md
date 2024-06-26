## CrossNet_kkbox_x1

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
Dataset ID: [KKBox_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/KKBox#KKBox_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/KKBox/KKBox_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_kkbox_x1_tuner_config_01](./CrossNet_kkbox_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_kkbox_x1
    nohup python run_expid.py --config ./CrossNet_kkbox_x1_tuner_config_01 --expid DCN_kkbox_x1_024_b363d48e --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.528343 | 0.811639  |


### Logs
```python
2022-03-09 05:12:37,612 P32969 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "6",
    "data_format": "csv",
    "data_root": "../data/KKBox/",
    "dataset_id": "kkbox_x1_227d337d",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
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
    "model_id": "DCN_kkbox_x1_024_b363d48e",
    "model_root": "./KKBox/CrossNet_kkbox_x1/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
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
2022-03-09 05:12:37,613 P32969 INFO Set up feature encoder...
2022-03-09 05:12:37,614 P32969 INFO Load feature_encoder from pickle: ../data/KKBox/kkbox_x1_227d337d/feature_encoder.pkl
2022-03-09 05:12:38,345 P32969 INFO Total number of parameters: 11829249.
2022-03-09 05:12:38,347 P32969 INFO Loading data...
2022-03-09 05:12:38,357 P32969 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/train.h5
2022-03-09 05:12:39,600 P32969 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/valid.h5
2022-03-09 05:12:40,598 P32969 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2022-03-09 05:12:40,713 P32969 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 05:12:40,713 P32969 INFO Loading train data done.
2022-03-09 05:12:48,467 P32969 INFO Start training: 591 batches/epoch
2022-03-09 05:12:48,468 P32969 INFO ************ Epoch=1 start ************
2022-03-09 05:15:23,672 P32969 INFO [Metrics] logloss: 0.581353 - AUC: 0.760158
2022-03-09 05:15:23,681 P32969 INFO Save best model: monitor(max): 0.178805
2022-03-09 05:15:23,754 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:15:23,917 P32969 INFO Train loss: 0.620553
2022-03-09 05:15:23,917 P32969 INFO ************ Epoch=1 end ************
2022-03-09 05:18:03,400 P32969 INFO [Metrics] logloss: 0.569043 - AUC: 0.773100
2022-03-09 05:18:03,401 P32969 INFO Save best model: monitor(max): 0.204057
2022-03-09 05:18:03,649 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:18:03,908 P32969 INFO Train loss: 0.602794
2022-03-09 05:18:03,909 P32969 INFO ************ Epoch=2 end ************
2022-03-09 05:20:44,657 P32969 INFO [Metrics] logloss: 0.564325 - AUC: 0.777658
2022-03-09 05:20:44,661 P32969 INFO Save best model: monitor(max): 0.213333
2022-03-09 05:20:44,892 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:20:45,135 P32969 INFO Train loss: 0.590934
2022-03-09 05:20:45,135 P32969 INFO ************ Epoch=3 end ************
2022-03-09 05:23:22,702 P32969 INFO [Metrics] logloss: 0.561274 - AUC: 0.780571
2022-03-09 05:23:22,703 P32969 INFO Save best model: monitor(max): 0.219297
2022-03-09 05:23:22,873 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:23:23,132 P32969 INFO Train loss: 0.584980
2022-03-09 05:23:23,133 P32969 INFO ************ Epoch=4 end ************
2022-03-09 05:26:08,430 P32969 INFO [Metrics] logloss: 0.559387 - AUC: 0.782368
2022-03-09 05:26:08,432 P32969 INFO Save best model: monitor(max): 0.222981
2022-03-09 05:26:08,560 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:26:08,749 P32969 INFO Train loss: 0.582083
2022-03-09 05:26:08,750 P32969 INFO ************ Epoch=5 end ************
2022-03-09 05:28:58,303 P32969 INFO [Metrics] logloss: 0.557472 - AUC: 0.784067
2022-03-09 05:28:58,305 P32969 INFO Save best model: monitor(max): 0.226594
2022-03-09 05:28:58,560 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:28:58,850 P32969 INFO Train loss: 0.580800
2022-03-09 05:28:58,851 P32969 INFO ************ Epoch=6 end ************
2022-03-09 05:31:48,981 P32969 INFO [Metrics] logloss: 0.556624 - AUC: 0.785517
2022-03-09 05:31:48,982 P32969 INFO Save best model: monitor(max): 0.228893
2022-03-09 05:31:49,147 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:31:49,278 P32969 INFO Train loss: 0.578550
2022-03-09 05:31:49,278 P32969 INFO ************ Epoch=7 end ************
2022-03-09 05:34:32,612 P32969 INFO [Metrics] logloss: 0.554745 - AUC: 0.786779
2022-03-09 05:34:32,616 P32969 INFO Save best model: monitor(max): 0.232034
2022-03-09 05:34:32,808 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:34:33,143 P32969 INFO Train loss: 0.576850
2022-03-09 05:34:33,144 P32969 INFO ************ Epoch=8 end ************
2022-03-09 05:37:14,940 P32969 INFO [Metrics] logloss: 0.553328 - AUC: 0.788220
2022-03-09 05:37:14,944 P32969 INFO Save best model: monitor(max): 0.234892
2022-03-09 05:37:15,174 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:37:15,383 P32969 INFO Train loss: 0.575746
2022-03-09 05:37:15,383 P32969 INFO ************ Epoch=9 end ************
2022-03-09 05:39:59,847 P32969 INFO [Metrics] logloss: 0.551727 - AUC: 0.789808
2022-03-09 05:39:59,849 P32969 INFO Save best model: monitor(max): 0.238081
2022-03-09 05:40:00,040 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:40:00,208 P32969 INFO Train loss: 0.575242
2022-03-09 05:40:00,209 P32969 INFO ************ Epoch=10 end ************
2022-03-09 05:42:36,353 P32969 INFO [Metrics] logloss: 0.550942 - AUC: 0.790673
2022-03-09 05:42:36,355 P32969 INFO Save best model: monitor(max): 0.239731
2022-03-09 05:42:36,469 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:42:36,784 P32969 INFO Train loss: 0.574379
2022-03-09 05:42:36,784 P32969 INFO ************ Epoch=11 end ************
2022-03-09 05:45:24,630 P32969 INFO [Metrics] logloss: 0.550386 - AUC: 0.791292
2022-03-09 05:45:24,632 P32969 INFO Save best model: monitor(max): 0.240906
2022-03-09 05:45:24,810 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:45:25,072 P32969 INFO Train loss: 0.573974
2022-03-09 05:45:25,073 P32969 INFO ************ Epoch=12 end ************
2022-03-09 05:48:01,382 P32969 INFO [Metrics] logloss: 0.549528 - AUC: 0.792090
2022-03-09 05:48:01,384 P32969 INFO Save best model: monitor(max): 0.242562
2022-03-09 05:48:01,502 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:48:01,648 P32969 INFO Train loss: 0.573863
2022-03-09 05:48:01,649 P32969 INFO ************ Epoch=13 end ************
2022-03-09 05:50:39,565 P32969 INFO [Metrics] logloss: 0.549029 - AUC: 0.792677
2022-03-09 05:50:39,567 P32969 INFO Save best model: monitor(max): 0.243648
2022-03-09 05:50:39,761 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:50:39,910 P32969 INFO Train loss: 0.574051
2022-03-09 05:50:39,910 P32969 INFO ************ Epoch=14 end ************
2022-03-09 05:53:15,702 P32969 INFO [Metrics] logloss: 0.548259 - AUC: 0.793734
2022-03-09 05:53:15,705 P32969 INFO Save best model: monitor(max): 0.245475
2022-03-09 05:53:16,017 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:53:16,186 P32969 INFO Train loss: 0.572930
2022-03-09 05:53:16,187 P32969 INFO ************ Epoch=15 end ************
2022-03-09 05:55:19,002 P32969 INFO [Metrics] logloss: 0.546964 - AUC: 0.794750
2022-03-09 05:55:19,005 P32969 INFO Save best model: monitor(max): 0.247786
2022-03-09 05:55:19,131 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:55:19,250 P32969 INFO Train loss: 0.571892
2022-03-09 05:55:19,251 P32969 INFO ************ Epoch=16 end ************
2022-03-09 05:56:55,960 P32969 INFO [Metrics] logloss: 0.545764 - AUC: 0.795639
2022-03-09 05:56:55,961 P32969 INFO Save best model: monitor(max): 0.249875
2022-03-09 05:56:56,047 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:56:56,342 P32969 INFO Train loss: 0.570857
2022-03-09 05:56:56,342 P32969 INFO ************ Epoch=17 end ************
2022-03-09 05:58:27,354 P32969 INFO [Metrics] logloss: 0.544783 - AUC: 0.796487
2022-03-09 05:58:27,356 P32969 INFO Save best model: monitor(max): 0.251703
2022-03-09 05:58:27,442 P32969 INFO --- 591/591 batches finished ---
2022-03-09 05:58:27,678 P32969 INFO Train loss: 0.569958
2022-03-09 05:58:27,679 P32969 INFO ************ Epoch=18 end ************
2022-03-09 06:00:38,581 P32969 INFO [Metrics] logloss: 0.544324 - AUC: 0.796943
2022-03-09 06:00:38,582 P32969 INFO Save best model: monitor(max): 0.252618
2022-03-09 06:00:38,682 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:00:39,096 P32969 INFO Train loss: 0.569010
2022-03-09 06:00:39,096 P32969 INFO ************ Epoch=19 end ************
2022-03-09 06:03:16,054 P32969 INFO [Metrics] logloss: 0.543899 - AUC: 0.797665
2022-03-09 06:03:16,055 P32969 INFO Save best model: monitor(max): 0.253765
2022-03-09 06:03:16,181 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:03:16,331 P32969 INFO Train loss: 0.568570
2022-03-09 06:03:16,331 P32969 INFO ************ Epoch=20 end ************
2022-03-09 06:06:06,904 P32969 INFO [Metrics] logloss: 0.542687 - AUC: 0.798370
2022-03-09 06:06:06,905 P32969 INFO Save best model: monitor(max): 0.255683
2022-03-09 06:06:07,002 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:06:07,203 P32969 INFO Train loss: 0.568186
2022-03-09 06:06:07,204 P32969 INFO ************ Epoch=21 end ************
2022-03-09 06:08:39,150 P32969 INFO [Metrics] logloss: 0.543281 - AUC: 0.798306
2022-03-09 06:08:39,154 P32969 INFO Monitor(max) STOP: 0.255025 !
2022-03-09 06:08:39,154 P32969 INFO Reduce learning rate on plateau: 0.000100
2022-03-09 06:08:39,155 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:08:39,385 P32969 INFO Train loss: 0.567387
2022-03-09 06:08:39,386 P32969 INFO ************ Epoch=22 end ************
2022-03-09 06:11:16,937 P32969 INFO [Metrics] logloss: 0.533738 - AUC: 0.806458
2022-03-09 06:11:16,942 P32969 INFO Save best model: monitor(max): 0.272719
2022-03-09 06:11:17,150 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:11:17,382 P32969 INFO Train loss: 0.543528
2022-03-09 06:11:17,383 P32969 INFO ************ Epoch=23 end ************
2022-03-09 06:14:00,366 P32969 INFO [Metrics] logloss: 0.532222 - AUC: 0.808145
2022-03-09 06:14:00,367 P32969 INFO Save best model: monitor(max): 0.275923
2022-03-09 06:14:00,489 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:14:00,750 P32969 INFO Train loss: 0.535834
2022-03-09 06:14:00,750 P32969 INFO ************ Epoch=24 end ************
2022-03-09 06:16:36,503 P32969 INFO [Metrics] logloss: 0.530663 - AUC: 0.809135
2022-03-09 06:16:36,506 P32969 INFO Save best model: monitor(max): 0.278472
2022-03-09 06:16:37,005 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:16:37,390 P32969 INFO Train loss: 0.532509
2022-03-09 06:16:37,390 P32969 INFO ************ Epoch=25 end ************
2022-03-09 06:19:09,059 P32969 INFO [Metrics] logloss: 0.530141 - AUC: 0.809683
2022-03-09 06:19:09,062 P32969 INFO Save best model: monitor(max): 0.279542
2022-03-09 06:19:09,362 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:19:09,591 P32969 INFO Train loss: 0.530443
2022-03-09 06:19:09,592 P32969 INFO ************ Epoch=26 end ************
2022-03-09 06:21:49,295 P32969 INFO [Metrics] logloss: 0.529839 - AUC: 0.809975
2022-03-09 06:21:49,297 P32969 INFO Save best model: monitor(max): 0.280136
2022-03-09 06:21:49,387 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:21:49,621 P32969 INFO Train loss: 0.528891
2022-03-09 06:21:49,622 P32969 INFO ************ Epoch=27 end ************
2022-03-09 06:24:29,690 P32969 INFO [Metrics] logloss: 0.529514 - AUC: 0.810158
2022-03-09 06:24:29,695 P32969 INFO Save best model: monitor(max): 0.280643
2022-03-09 06:24:30,093 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:24:30,251 P32969 INFO Train loss: 0.527651
2022-03-09 06:24:30,252 P32969 INFO ************ Epoch=28 end ************
2022-03-09 06:27:11,362 P32969 INFO [Metrics] logloss: 0.529336 - AUC: 0.810444
2022-03-09 06:27:11,363 P32969 INFO Save best model: monitor(max): 0.281108
2022-03-09 06:27:11,433 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:27:11,564 P32969 INFO Train loss: 0.526470
2022-03-09 06:27:11,565 P32969 INFO ************ Epoch=29 end ************
2022-03-09 06:29:53,777 P32969 INFO [Metrics] logloss: 0.529113 - AUC: 0.810575
2022-03-09 06:29:53,778 P32969 INFO Save best model: monitor(max): 0.281463
2022-03-09 06:29:53,870 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:29:54,068 P32969 INFO Train loss: 0.525580
2022-03-09 06:29:54,069 P32969 INFO ************ Epoch=30 end ************
2022-03-09 06:32:30,465 P32969 INFO [Metrics] logloss: 0.529139 - AUC: 0.810611
2022-03-09 06:32:30,466 P32969 INFO Save best model: monitor(max): 0.281472
2022-03-09 06:32:30,735 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:32:31,093 P32969 INFO Train loss: 0.524854
2022-03-09 06:32:31,095 P32969 INFO ************ Epoch=31 end ************
2022-03-09 06:35:12,818 P32969 INFO [Metrics] logloss: 0.528996 - AUC: 0.810753
2022-03-09 06:35:12,819 P32969 INFO Save best model: monitor(max): 0.281757
2022-03-09 06:35:12,946 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:35:13,035 P32969 INFO Train loss: 0.523999
2022-03-09 06:35:13,035 P32969 INFO ************ Epoch=32 end ************
2022-03-09 06:38:02,993 P32969 INFO [Metrics] logloss: 0.528867 - AUC: 0.810803
2022-03-09 06:38:02,995 P32969 INFO Save best model: monitor(max): 0.281936
2022-03-09 06:38:03,130 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:38:03,473 P32969 INFO Train loss: 0.523321
2022-03-09 06:38:03,473 P32969 INFO ************ Epoch=33 end ************
2022-03-09 06:40:45,029 P32969 INFO [Metrics] logloss: 0.529005 - AUC: 0.810639
2022-03-09 06:40:45,030 P32969 INFO Monitor(max) STOP: 0.281634 !
2022-03-09 06:40:45,030 P32969 INFO Reduce learning rate on plateau: 0.000010
2022-03-09 06:40:45,030 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:40:45,221 P32969 INFO Train loss: 0.522771
2022-03-09 06:40:45,222 P32969 INFO ************ Epoch=34 end ************
2022-03-09 06:43:33,667 P32969 INFO [Metrics] logloss: 0.528779 - AUC: 0.811152
2022-03-09 06:43:33,687 P32969 INFO Save best model: monitor(max): 0.282373
2022-03-09 06:43:33,831 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:43:34,230 P32969 INFO Train loss: 0.515691
2022-03-09 06:43:34,231 P32969 INFO ************ Epoch=35 end ************
2022-03-09 06:46:20,257 P32969 INFO [Metrics] logloss: 0.528826 - AUC: 0.811269
2022-03-09 06:46:20,268 P32969 INFO Save best model: monitor(max): 0.282443
2022-03-09 06:46:20,409 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:46:20,513 P32969 INFO Train loss: 0.515059
2022-03-09 06:46:20,513 P32969 INFO ************ Epoch=36 end ************
2022-03-09 06:49:04,040 P32969 INFO [Metrics] logloss: 0.528927 - AUC: 0.811273
2022-03-09 06:49:04,045 P32969 INFO Monitor(max) STOP: 0.282345 !
2022-03-09 06:49:04,045 P32969 INFO Reduce learning rate on plateau: 0.000001
2022-03-09 06:49:04,046 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:49:04,202 P32969 INFO Train loss: 0.514633
2022-03-09 06:49:04,202 P32969 INFO ************ Epoch=37 end ************
2022-03-09 06:51:47,372 P32969 INFO [Metrics] logloss: 0.528932 - AUC: 0.811292
2022-03-09 06:51:47,400 P32969 INFO Monitor(max) STOP: 0.282360 !
2022-03-09 06:51:47,402 P32969 INFO Reduce learning rate on plateau: 0.000001
2022-03-09 06:51:47,402 P32969 INFO Early stopping at epoch=38
2022-03-09 06:51:47,403 P32969 INFO --- 591/591 batches finished ---
2022-03-09 06:51:47,575 P32969 INFO Train loss: 0.513601
2022-03-09 06:51:47,575 P32969 INFO Training finished.
2022-03-09 06:51:47,575 P32969 INFO Load best model: /home/XXX/FuxiCTR_v1.0/benchmarks/KKBox/CrossNet_kkbox_x1/kkbox_x1_227d337d/DCN_kkbox_x1_024_b363d48e_model.ckpt
2022-03-09 06:51:47,666 P32969 INFO ****** Validation evaluation ******
2022-03-09 06:52:07,306 P32969 INFO [Metrics] logloss: 0.528826 - AUC: 0.811269
2022-03-09 06:52:07,357 P32969 INFO ******** Test evaluation ********
2022-03-09 06:52:07,357 P32969 INFO Loading data...
2022-03-09 06:52:07,357 P32969 INFO Loading data from h5: ../data/KKBox/kkbox_x1_227d337d/test.h5
2022-03-09 06:52:07,472 P32969 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2022-03-09 06:52:07,472 P32969 INFO Loading test data done.
2022-03-09 06:52:24,850 P32969 INFO [Metrics] logloss: 0.528343 - AUC: 0.811639

```
