## FwFM_frappe_x1

A hands-on guide to run the FwFM model on the Frappe_x1 dataset.

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
Dataset ID: [Frappe_x1](https://github.com/reczoo/Datasets/tree/main/Frappe/Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FwFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FwFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FwFM_frappe_x1_tuner_config_02](./FwFM_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FwFM_frappe_x1
    nohup python run_expid.py --config ./FwFM_frappe_x1_tuner_config_02 --expid FwFM_frappe_x1_005_95cf3ccd --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.977585 | 0.202971  |


### Logs
```python
2022-01-25 13:02:57,678 P151130 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "linear_type": "FeLV",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FwFM",
    "model_id": "FwFM_frappe_x1_005_95cf3ccd",
    "model_root": "./Frappe/FwFM_frappe_x1/",
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
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-25 13:02:57,679 P151130 INFO Set up feature encoder...
2022-01-25 13:02:57,746 P151130 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-25 13:02:57,747 P151130 INFO Loading data...
2022-01-25 13:02:57,751 P151130 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-25 13:02:57,764 P151130 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-25 13:02:57,769 P151130 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-25 13:02:57,769 P151130 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-25 13:02:57,769 P151130 INFO Loading train data done.
2022-01-25 13:03:00,978 P151130 INFO Total number of parameters: 107826.
2022-01-25 13:03:00,979 P151130 INFO Start training: 50 batches/epoch
2022-01-25 13:03:00,979 P151130 INFO ************ Epoch=1 start ************
2022-01-25 13:03:06,426 P151130 INFO [Metrics] AUC: 0.825558 - logloss: 0.642467
2022-01-25 13:03:06,426 P151130 INFO Save best model: monitor(max): 0.825558
2022-01-25 13:03:06,429 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:06,468 P151130 INFO Train loss: 0.676589
2022-01-25 13:03:06,469 P151130 INFO ************ Epoch=1 end ************
2022-01-25 13:03:11,825 P151130 INFO [Metrics] AUC: 0.912322 - logloss: 0.585322
2022-01-25 13:03:11,825 P151130 INFO Save best model: monitor(max): 0.912322
2022-01-25 13:03:11,829 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:11,875 P151130 INFO Train loss: 0.615467
2022-01-25 13:03:11,875 P151130 INFO ************ Epoch=2 end ************
2022-01-25 13:03:17,300 P151130 INFO [Metrics] AUC: 0.930125 - logloss: 0.488029
2022-01-25 13:03:17,300 P151130 INFO Save best model: monitor(max): 0.930125
2022-01-25 13:03:17,303 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:17,347 P151130 INFO Train loss: 0.544677
2022-01-25 13:03:17,347 P151130 INFO ************ Epoch=3 end ************
2022-01-25 13:03:22,707 P151130 INFO [Metrics] AUC: 0.932953 - logloss: 0.362171
2022-01-25 13:03:22,708 P151130 INFO Save best model: monitor(max): 0.932953
2022-01-25 13:03:22,710 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:22,753 P151130 INFO Train loss: 0.424241
2022-01-25 13:03:22,754 P151130 INFO ************ Epoch=4 end ************
2022-01-25 13:03:28,032 P151130 INFO [Metrics] AUC: 0.936262 - logloss: 0.299799
2022-01-25 13:03:28,033 P151130 INFO Save best model: monitor(max): 0.936262
2022-01-25 13:03:28,037 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:28,081 P151130 INFO Train loss: 0.322649
2022-01-25 13:03:28,082 P151130 INFO ************ Epoch=5 end ************
2022-01-25 13:03:33,263 P151130 INFO [Metrics] AUC: 0.939015 - logloss: 0.282734
2022-01-25 13:03:33,264 P151130 INFO Save best model: monitor(max): 0.939015
2022-01-25 13:03:33,267 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:33,311 P151130 INFO Train loss: 0.284470
2022-01-25 13:03:33,311 P151130 INFO ************ Epoch=6 end ************
2022-01-25 13:03:38,508 P151130 INFO [Metrics] AUC: 0.940831 - logloss: 0.276600
2022-01-25 13:03:38,509 P151130 INFO Save best model: monitor(max): 0.940831
2022-01-25 13:03:38,513 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:38,558 P151130 INFO Train loss: 0.270986
2022-01-25 13:03:38,558 P151130 INFO ************ Epoch=7 end ************
2022-01-25 13:03:43,762 P151130 INFO [Metrics] AUC: 0.941994 - logloss: 0.273055
2022-01-25 13:03:43,763 P151130 INFO Save best model: monitor(max): 0.941994
2022-01-25 13:03:43,767 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:43,812 P151130 INFO Train loss: 0.264837
2022-01-25 13:03:43,813 P151130 INFO ************ Epoch=8 end ************
2022-01-25 13:03:48,988 P151130 INFO [Metrics] AUC: 0.943004 - logloss: 0.270376
2022-01-25 13:03:48,989 P151130 INFO Save best model: monitor(max): 0.943004
2022-01-25 13:03:48,991 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:49,032 P151130 INFO Train loss: 0.259572
2022-01-25 13:03:49,032 P151130 INFO ************ Epoch=9 end ************
2022-01-25 13:03:54,120 P151130 INFO [Metrics] AUC: 0.944096 - logloss: 0.267543
2022-01-25 13:03:54,121 P151130 INFO Save best model: monitor(max): 0.944096
2022-01-25 13:03:54,124 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:54,170 P151130 INFO Train loss: 0.254982
2022-01-25 13:03:54,170 P151130 INFO ************ Epoch=10 end ************
2022-01-25 13:03:59,267 P151130 INFO [Metrics] AUC: 0.945223 - logloss: 0.264601
2022-01-25 13:03:59,267 P151130 INFO Save best model: monitor(max): 0.945223
2022-01-25 13:03:59,270 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:03:59,310 P151130 INFO Train loss: 0.251286
2022-01-25 13:03:59,310 P151130 INFO ************ Epoch=11 end ************
2022-01-25 13:04:04,470 P151130 INFO [Metrics] AUC: 0.946552 - logloss: 0.261291
2022-01-25 13:04:04,470 P151130 INFO Save best model: monitor(max): 0.946552
2022-01-25 13:04:04,473 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:04,518 P151130 INFO Train loss: 0.246314
2022-01-25 13:04:04,519 P151130 INFO ************ Epoch=12 end ************
2022-01-25 13:04:09,680 P151130 INFO [Metrics] AUC: 0.947818 - logloss: 0.258079
2022-01-25 13:04:09,681 P151130 INFO Save best model: monitor(max): 0.947818
2022-01-25 13:04:09,684 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:09,729 P151130 INFO Train loss: 0.241710
2022-01-25 13:04:09,729 P151130 INFO ************ Epoch=13 end ************
2022-01-25 13:04:14,901 P151130 INFO [Metrics] AUC: 0.949313 - logloss: 0.254250
2022-01-25 13:04:14,902 P151130 INFO Save best model: monitor(max): 0.949313
2022-01-25 13:04:14,905 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:14,942 P151130 INFO Train loss: 0.236350
2022-01-25 13:04:14,943 P151130 INFO ************ Epoch=14 end ************
2022-01-25 13:04:20,034 P151130 INFO [Metrics] AUC: 0.950788 - logloss: 0.250517
2022-01-25 13:04:20,034 P151130 INFO Save best model: monitor(max): 0.950788
2022-01-25 13:04:20,037 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:20,075 P151130 INFO Train loss: 0.231681
2022-01-25 13:04:20,075 P151130 INFO ************ Epoch=15 end ************
2022-01-25 13:04:25,191 P151130 INFO [Metrics] AUC: 0.952359 - logloss: 0.246642
2022-01-25 13:04:25,192 P151130 INFO Save best model: monitor(max): 0.952359
2022-01-25 13:04:25,195 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:25,232 P151130 INFO Train loss: 0.226333
2022-01-25 13:04:25,233 P151130 INFO ************ Epoch=16 end ************
2022-01-25 13:04:30,134 P151130 INFO [Metrics] AUC: 0.953701 - logloss: 0.243073
2022-01-25 13:04:30,134 P151130 INFO Save best model: monitor(max): 0.953701
2022-01-25 13:04:30,137 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:30,175 P151130 INFO Train loss: 0.221282
2022-01-25 13:04:30,176 P151130 INFO ************ Epoch=17 end ************
2022-01-25 13:04:34,766 P151130 INFO [Metrics] AUC: 0.955059 - logloss: 0.239597
2022-01-25 13:04:34,766 P151130 INFO Save best model: monitor(max): 0.955059
2022-01-25 13:04:34,769 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:34,809 P151130 INFO Train loss: 0.215794
2022-01-25 13:04:34,809 P151130 INFO ************ Epoch=18 end ************
2022-01-25 13:04:39,417 P151130 INFO [Metrics] AUC: 0.956343 - logloss: 0.236236
2022-01-25 13:04:39,418 P151130 INFO Save best model: monitor(max): 0.956343
2022-01-25 13:04:39,421 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:39,463 P151130 INFO Train loss: 0.211311
2022-01-25 13:04:39,463 P151130 INFO ************ Epoch=19 end ************
2022-01-25 13:04:44,096 P151130 INFO [Metrics] AUC: 0.957586 - logloss: 0.232878
2022-01-25 13:04:44,097 P151130 INFO Save best model: monitor(max): 0.957586
2022-01-25 13:04:44,100 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:44,144 P151130 INFO Train loss: 0.206239
2022-01-25 13:04:44,144 P151130 INFO ************ Epoch=20 end ************
2022-01-25 13:04:48,832 P151130 INFO [Metrics] AUC: 0.958806 - logloss: 0.229916
2022-01-25 13:04:48,833 P151130 INFO Save best model: monitor(max): 0.958806
2022-01-25 13:04:48,836 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:48,881 P151130 INFO Train loss: 0.201945
2022-01-25 13:04:48,881 P151130 INFO ************ Epoch=21 end ************
2022-01-25 13:04:53,561 P151130 INFO [Metrics] AUC: 0.959829 - logloss: 0.227099
2022-01-25 13:04:53,561 P151130 INFO Save best model: monitor(max): 0.959829
2022-01-25 13:04:53,564 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:53,602 P151130 INFO Train loss: 0.197302
2022-01-25 13:04:53,603 P151130 INFO ************ Epoch=22 end ************
2022-01-25 13:04:58,294 P151130 INFO [Metrics] AUC: 0.960831 - logloss: 0.224389
2022-01-25 13:04:58,294 P151130 INFO Save best model: monitor(max): 0.960831
2022-01-25 13:04:58,297 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:04:58,339 P151130 INFO Train loss: 0.193032
2022-01-25 13:04:58,339 P151130 INFO ************ Epoch=23 end ************
2022-01-25 13:05:02,933 P151130 INFO [Metrics] AUC: 0.961774 - logloss: 0.221851
2022-01-25 13:05:02,934 P151130 INFO Save best model: monitor(max): 0.961774
2022-01-25 13:05:02,936 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:02,978 P151130 INFO Train loss: 0.189028
2022-01-25 13:05:02,978 P151130 INFO ************ Epoch=24 end ************
2022-01-25 13:05:07,720 P151130 INFO [Metrics] AUC: 0.962625 - logloss: 0.219540
2022-01-25 13:05:07,720 P151130 INFO Save best model: monitor(max): 0.962625
2022-01-25 13:05:07,723 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:07,768 P151130 INFO Train loss: 0.185187
2022-01-25 13:05:07,768 P151130 INFO ************ Epoch=25 end ************
2022-01-25 13:05:12,761 P151130 INFO [Metrics] AUC: 0.963505 - logloss: 0.217245
2022-01-25 13:05:12,762 P151130 INFO Save best model: monitor(max): 0.963505
2022-01-25 13:05:12,764 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:12,805 P151130 INFO Train loss: 0.181264
2022-01-25 13:05:12,805 P151130 INFO ************ Epoch=26 end ************
2022-01-25 13:05:18,062 P151130 INFO [Metrics] AUC: 0.964298 - logloss: 0.215106
2022-01-25 13:05:18,063 P151130 INFO Save best model: monitor(max): 0.964298
2022-01-25 13:05:18,065 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:18,122 P151130 INFO Train loss: 0.177636
2022-01-25 13:05:18,122 P151130 INFO ************ Epoch=27 end ************
2022-01-25 13:05:23,465 P151130 INFO [Metrics] AUC: 0.965036 - logloss: 0.213073
2022-01-25 13:05:23,465 P151130 INFO Save best model: monitor(max): 0.965036
2022-01-25 13:05:23,468 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:23,514 P151130 INFO Train loss: 0.174316
2022-01-25 13:05:23,515 P151130 INFO ************ Epoch=28 end ************
2022-01-25 13:05:28,959 P151130 INFO [Metrics] AUC: 0.965753 - logloss: 0.211255
2022-01-25 13:05:28,960 P151130 INFO Save best model: monitor(max): 0.965753
2022-01-25 13:05:28,963 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:29,004 P151130 INFO Train loss: 0.170571
2022-01-25 13:05:29,004 P151130 INFO ************ Epoch=29 end ************
2022-01-25 13:05:34,368 P151130 INFO [Metrics] AUC: 0.966380 - logloss: 0.209511
2022-01-25 13:05:34,369 P151130 INFO Save best model: monitor(max): 0.966380
2022-01-25 13:05:34,372 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:34,410 P151130 INFO Train loss: 0.167450
2022-01-25 13:05:34,410 P151130 INFO ************ Epoch=30 end ************
2022-01-25 13:05:39,809 P151130 INFO [Metrics] AUC: 0.967011 - logloss: 0.207716
2022-01-25 13:05:39,809 P151130 INFO Save best model: monitor(max): 0.967011
2022-01-25 13:05:39,812 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:39,852 P151130 INFO Train loss: 0.164642
2022-01-25 13:05:39,852 P151130 INFO ************ Epoch=31 end ************
2022-01-25 13:05:45,193 P151130 INFO [Metrics] AUC: 0.967550 - logloss: 0.206297
2022-01-25 13:05:45,194 P151130 INFO Save best model: monitor(max): 0.967550
2022-01-25 13:05:45,197 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:45,238 P151130 INFO Train loss: 0.161473
2022-01-25 13:05:45,239 P151130 INFO ************ Epoch=32 end ************
2022-01-25 13:05:50,473 P151130 INFO [Metrics] AUC: 0.968078 - logloss: 0.204966
2022-01-25 13:05:50,474 P151130 INFO Save best model: monitor(max): 0.968078
2022-01-25 13:05:50,477 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:50,516 P151130 INFO Train loss: 0.158590
2022-01-25 13:05:50,516 P151130 INFO ************ Epoch=33 end ************
2022-01-25 13:05:55,674 P151130 INFO [Metrics] AUC: 0.968542 - logloss: 0.203658
2022-01-25 13:05:55,674 P151130 INFO Save best model: monitor(max): 0.968542
2022-01-25 13:05:55,677 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:05:55,720 P151130 INFO Train loss: 0.155808
2022-01-25 13:05:55,720 P151130 INFO ************ Epoch=34 end ************
2022-01-25 13:06:00,874 P151130 INFO [Metrics] AUC: 0.969045 - logloss: 0.202377
2022-01-25 13:06:00,875 P151130 INFO Save best model: monitor(max): 0.969045
2022-01-25 13:06:00,878 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:00,915 P151130 INFO Train loss: 0.152936
2022-01-25 13:06:00,915 P151130 INFO ************ Epoch=35 end ************
2022-01-25 13:06:06,043 P151130 INFO [Metrics] AUC: 0.969464 - logloss: 0.201278
2022-01-25 13:06:06,044 P151130 INFO Save best model: monitor(max): 0.969464
2022-01-25 13:06:06,047 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:06,088 P151130 INFO Train loss: 0.149958
2022-01-25 13:06:06,088 P151130 INFO ************ Epoch=36 end ************
2022-01-25 13:06:11,235 P151130 INFO [Metrics] AUC: 0.969915 - logloss: 0.200242
2022-01-25 13:06:11,235 P151130 INFO Save best model: monitor(max): 0.969915
2022-01-25 13:06:11,238 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:11,281 P151130 INFO Train loss: 0.147619
2022-01-25 13:06:11,281 P151130 INFO ************ Epoch=37 end ************
2022-01-25 13:06:16,419 P151130 INFO [Metrics] AUC: 0.970298 - logloss: 0.199216
2022-01-25 13:06:16,420 P151130 INFO Save best model: monitor(max): 0.970298
2022-01-25 13:06:16,423 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:16,464 P151130 INFO Train loss: 0.145333
2022-01-25 13:06:16,464 P151130 INFO ************ Epoch=38 end ************
2022-01-25 13:06:21,638 P151130 INFO [Metrics] AUC: 0.970661 - logloss: 0.198269
2022-01-25 13:06:21,639 P151130 INFO Save best model: monitor(max): 0.970661
2022-01-25 13:06:21,642 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:21,684 P151130 INFO Train loss: 0.142817
2022-01-25 13:06:21,685 P151130 INFO ************ Epoch=39 end ************
2022-01-25 13:06:26,790 P151130 INFO [Metrics] AUC: 0.971045 - logloss: 0.197306
2022-01-25 13:06:26,791 P151130 INFO Save best model: monitor(max): 0.971045
2022-01-25 13:06:26,793 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:26,839 P151130 INFO Train loss: 0.140722
2022-01-25 13:06:26,839 P151130 INFO ************ Epoch=40 end ************
2022-01-25 13:06:31,985 P151130 INFO [Metrics] AUC: 0.971370 - logloss: 0.196588
2022-01-25 13:06:31,985 P151130 INFO Save best model: monitor(max): 0.971370
2022-01-25 13:06:31,988 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:32,032 P151130 INFO Train loss: 0.138491
2022-01-25 13:06:32,032 P151130 INFO ************ Epoch=41 end ************
2022-01-25 13:06:37,089 P151130 INFO [Metrics] AUC: 0.971716 - logloss: 0.195699
2022-01-25 13:06:37,089 P151130 INFO Save best model: monitor(max): 0.971716
2022-01-25 13:06:37,092 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:37,138 P151130 INFO Train loss: 0.136320
2022-01-25 13:06:37,138 P151130 INFO ************ Epoch=42 end ************
2022-01-25 13:06:42,226 P151130 INFO [Metrics] AUC: 0.972011 - logloss: 0.195044
2022-01-25 13:06:42,226 P151130 INFO Save best model: monitor(max): 0.972011
2022-01-25 13:06:42,229 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:42,272 P151130 INFO Train loss: 0.133843
2022-01-25 13:06:42,272 P151130 INFO ************ Epoch=43 end ************
2022-01-25 13:06:47,157 P151130 INFO [Metrics] AUC: 0.972313 - logloss: 0.194276
2022-01-25 13:06:47,158 P151130 INFO Save best model: monitor(max): 0.972313
2022-01-25 13:06:47,161 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:47,199 P151130 INFO Train loss: 0.131882
2022-01-25 13:06:47,199 P151130 INFO ************ Epoch=44 end ************
2022-01-25 13:06:51,916 P151130 INFO [Metrics] AUC: 0.972577 - logloss: 0.193937
2022-01-25 13:06:51,916 P151130 INFO Save best model: monitor(max): 0.972577
2022-01-25 13:06:51,919 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:51,961 P151130 INFO Train loss: 0.129808
2022-01-25 13:06:51,961 P151130 INFO ************ Epoch=45 end ************
2022-01-25 13:06:56,686 P151130 INFO [Metrics] AUC: 0.972899 - logloss: 0.193113
2022-01-25 13:06:56,686 P151130 INFO Save best model: monitor(max): 0.972899
2022-01-25 13:06:56,690 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:06:56,733 P151130 INFO Train loss: 0.127713
2022-01-25 13:06:56,733 P151130 INFO ************ Epoch=46 end ************
2022-01-25 13:07:01,572 P151130 INFO [Metrics] AUC: 0.973154 - logloss: 0.192513
2022-01-25 13:07:01,573 P151130 INFO Save best model: monitor(max): 0.973154
2022-01-25 13:07:01,575 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:01,618 P151130 INFO Train loss: 0.125912
2022-01-25 13:07:01,618 P151130 INFO ************ Epoch=47 end ************
2022-01-25 13:07:06,449 P151130 INFO [Metrics] AUC: 0.973380 - logloss: 0.192186
2022-01-25 13:07:06,450 P151130 INFO Save best model: monitor(max): 0.973380
2022-01-25 13:07:06,452 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:06,495 P151130 INFO Train loss: 0.124154
2022-01-25 13:07:06,495 P151130 INFO ************ Epoch=48 end ************
2022-01-25 13:07:11,273 P151130 INFO [Metrics] AUC: 0.973626 - logloss: 0.191674
2022-01-25 13:07:11,274 P151130 INFO Save best model: monitor(max): 0.973626
2022-01-25 13:07:11,277 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:11,315 P151130 INFO Train loss: 0.122062
2022-01-25 13:07:11,315 P151130 INFO ************ Epoch=49 end ************
2022-01-25 13:07:16,099 P151130 INFO [Metrics] AUC: 0.973874 - logloss: 0.191300
2022-01-25 13:07:16,099 P151130 INFO Save best model: monitor(max): 0.973874
2022-01-25 13:07:16,102 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:16,140 P151130 INFO Train loss: 0.120345
2022-01-25 13:07:16,140 P151130 INFO ************ Epoch=50 end ************
2022-01-25 13:07:20,885 P151130 INFO [Metrics] AUC: 0.974098 - logloss: 0.190934
2022-01-25 13:07:20,885 P151130 INFO Save best model: monitor(max): 0.974098
2022-01-25 13:07:20,888 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:20,931 P151130 INFO Train loss: 0.118131
2022-01-25 13:07:20,931 P151130 INFO ************ Epoch=51 end ************
2022-01-25 13:07:25,564 P151130 INFO [Metrics] AUC: 0.974294 - logloss: 0.190396
2022-01-25 13:07:25,564 P151130 INFO Save best model: monitor(max): 0.974294
2022-01-25 13:07:25,567 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:25,614 P151130 INFO Train loss: 0.116452
2022-01-25 13:07:25,614 P151130 INFO ************ Epoch=52 end ************
2022-01-25 13:07:30,286 P151130 INFO [Metrics] AUC: 0.974523 - logloss: 0.190037
2022-01-25 13:07:30,287 P151130 INFO Save best model: monitor(max): 0.974523
2022-01-25 13:07:30,290 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:30,332 P151130 INFO Train loss: 0.114803
2022-01-25 13:07:30,332 P151130 INFO ************ Epoch=53 end ************
2022-01-25 13:07:35,213 P151130 INFO [Metrics] AUC: 0.974723 - logloss: 0.189784
2022-01-25 13:07:35,213 P151130 INFO Save best model: monitor(max): 0.974723
2022-01-25 13:07:35,216 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:35,260 P151130 INFO Train loss: 0.113125
2022-01-25 13:07:35,261 P151130 INFO ************ Epoch=54 end ************
2022-01-25 13:07:40,228 P151130 INFO [Metrics] AUC: 0.974929 - logloss: 0.189521
2022-01-25 13:07:40,228 P151130 INFO Save best model: monitor(max): 0.974929
2022-01-25 13:07:40,231 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:40,274 P151130 INFO Train loss: 0.111603
2022-01-25 13:07:40,275 P151130 INFO ************ Epoch=55 end ************
2022-01-25 13:07:45,357 P151130 INFO [Metrics] AUC: 0.975130 - logloss: 0.189064
2022-01-25 13:07:45,358 P151130 INFO Save best model: monitor(max): 0.975130
2022-01-25 13:07:45,361 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:45,399 P151130 INFO Train loss: 0.109957
2022-01-25 13:07:45,400 P151130 INFO ************ Epoch=56 end ************
2022-01-25 13:07:50,474 P151130 INFO [Metrics] AUC: 0.975320 - logloss: 0.189088
2022-01-25 13:07:50,474 P151130 INFO Save best model: monitor(max): 0.975320
2022-01-25 13:07:50,477 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:50,515 P151130 INFO Train loss: 0.108105
2022-01-25 13:07:50,516 P151130 INFO ************ Epoch=57 end ************
2022-01-25 13:07:55,645 P151130 INFO [Metrics] AUC: 0.975443 - logloss: 0.188968
2022-01-25 13:07:55,646 P151130 INFO Save best model: monitor(max): 0.975443
2022-01-25 13:07:55,649 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:07:55,688 P151130 INFO Train loss: 0.106790
2022-01-25 13:07:55,688 P151130 INFO ************ Epoch=58 end ************
2022-01-25 13:08:00,748 P151130 INFO [Metrics] AUC: 0.975669 - logloss: 0.188673
2022-01-25 13:08:00,748 P151130 INFO Save best model: monitor(max): 0.975669
2022-01-25 13:08:00,751 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:00,796 P151130 INFO Train loss: 0.104968
2022-01-25 13:08:00,796 P151130 INFO ************ Epoch=59 end ************
2022-01-25 13:08:06,088 P151130 INFO [Metrics] AUC: 0.975805 - logloss: 0.188587
2022-01-25 13:08:06,089 P151130 INFO Save best model: monitor(max): 0.975805
2022-01-25 13:08:06,092 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:06,139 P151130 INFO Train loss: 0.103423
2022-01-25 13:08:06,139 P151130 INFO ************ Epoch=60 end ************
2022-01-25 13:08:11,321 P151130 INFO [Metrics] AUC: 0.975964 - logloss: 0.188326
2022-01-25 13:08:11,321 P151130 INFO Save best model: monitor(max): 0.975964
2022-01-25 13:08:11,325 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:11,366 P151130 INFO Train loss: 0.101775
2022-01-25 13:08:11,366 P151130 INFO ************ Epoch=61 end ************
2022-01-25 13:08:16,490 P151130 INFO [Metrics] AUC: 0.976103 - logloss: 0.188300
2022-01-25 13:08:16,491 P151130 INFO Save best model: monitor(max): 0.976103
2022-01-25 13:08:16,494 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:16,541 P151130 INFO Train loss: 0.100580
2022-01-25 13:08:16,541 P151130 INFO ************ Epoch=62 end ************
2022-01-25 13:08:21,624 P151130 INFO [Metrics] AUC: 0.976294 - logloss: 0.188147
2022-01-25 13:08:21,625 P151130 INFO Save best model: monitor(max): 0.976294
2022-01-25 13:08:21,627 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:21,677 P151130 INFO Train loss: 0.099216
2022-01-25 13:08:21,677 P151130 INFO ************ Epoch=63 end ************
2022-01-25 13:08:26,899 P151130 INFO [Metrics] AUC: 0.976429 - logloss: 0.188130
2022-01-25 13:08:26,899 P151130 INFO Save best model: monitor(max): 0.976429
2022-01-25 13:08:26,903 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:26,945 P151130 INFO Train loss: 0.097642
2022-01-25 13:08:26,945 P151130 INFO ************ Epoch=64 end ************
2022-01-25 13:08:32,111 P151130 INFO [Metrics] AUC: 0.976563 - logloss: 0.188406
2022-01-25 13:08:32,112 P151130 INFO Save best model: monitor(max): 0.976563
2022-01-25 13:08:32,115 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:32,157 P151130 INFO Train loss: 0.096320
2022-01-25 13:08:32,158 P151130 INFO ************ Epoch=65 end ************
2022-01-25 13:08:37,433 P151130 INFO [Metrics] AUC: 0.976689 - logloss: 0.188268
2022-01-25 13:08:37,434 P151130 INFO Save best model: monitor(max): 0.976689
2022-01-25 13:08:37,437 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:37,481 P151130 INFO Train loss: 0.094901
2022-01-25 13:08:37,482 P151130 INFO ************ Epoch=66 end ************
2022-01-25 13:08:42,868 P151130 INFO [Metrics] AUC: 0.976796 - logloss: 0.188085
2022-01-25 13:08:42,868 P151130 INFO Save best model: monitor(max): 0.976796
2022-01-25 13:08:42,874 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:42,918 P151130 INFO Train loss: 0.093579
2022-01-25 13:08:42,918 P151130 INFO ************ Epoch=67 end ************
2022-01-25 13:08:48,355 P151130 INFO [Metrics] AUC: 0.976985 - logloss: 0.187967
2022-01-25 13:08:48,356 P151130 INFO Save best model: monitor(max): 0.976985
2022-01-25 13:08:48,359 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:48,398 P151130 INFO Train loss: 0.092258
2022-01-25 13:08:48,398 P151130 INFO ************ Epoch=68 end ************
2022-01-25 13:08:54,030 P151130 INFO [Metrics] AUC: 0.977079 - logloss: 0.188214
2022-01-25 13:08:54,031 P151130 INFO Save best model: monitor(max): 0.977079
2022-01-25 13:08:54,034 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:54,082 P151130 INFO Train loss: 0.091115
2022-01-25 13:08:54,082 P151130 INFO ************ Epoch=69 end ************
2022-01-25 13:08:59,555 P151130 INFO [Metrics] AUC: 0.977169 - logloss: 0.188373
2022-01-25 13:08:59,555 P151130 INFO Save best model: monitor(max): 0.977169
2022-01-25 13:08:59,558 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:08:59,599 P151130 INFO Train loss: 0.089781
2022-01-25 13:08:59,599 P151130 INFO ************ Epoch=70 end ************
2022-01-25 13:09:04,997 P151130 INFO [Metrics] AUC: 0.977319 - logloss: 0.188373
2022-01-25 13:09:04,997 P151130 INFO Save best model: monitor(max): 0.977319
2022-01-25 13:09:05,000 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:05,045 P151130 INFO Train loss: 0.088693
2022-01-25 13:09:05,045 P151130 INFO ************ Epoch=71 end ************
2022-01-25 13:09:10,404 P151130 INFO [Metrics] AUC: 0.977408 - logloss: 0.188566
2022-01-25 13:09:10,405 P151130 INFO Save best model: monitor(max): 0.977408
2022-01-25 13:09:10,408 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:10,453 P151130 INFO Train loss: 0.087247
2022-01-25 13:09:10,453 P151130 INFO ************ Epoch=72 end ************
2022-01-25 13:09:15,780 P151130 INFO [Metrics] AUC: 0.977503 - logloss: 0.188809
2022-01-25 13:09:15,780 P151130 INFO Save best model: monitor(max): 0.977503
2022-01-25 13:09:15,783 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:15,822 P151130 INFO Train loss: 0.086016
2022-01-25 13:09:15,822 P151130 INFO ************ Epoch=73 end ************
2022-01-25 13:09:21,072 P151130 INFO [Metrics] AUC: 0.977633 - logloss: 0.188784
2022-01-25 13:09:21,073 P151130 INFO Save best model: monitor(max): 0.977633
2022-01-25 13:09:21,076 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:21,123 P151130 INFO Train loss: 0.085163
2022-01-25 13:09:21,123 P151130 INFO ************ Epoch=74 end ************
2022-01-25 13:09:26,459 P151130 INFO [Metrics] AUC: 0.977726 - logloss: 0.189094
2022-01-25 13:09:26,459 P151130 INFO Save best model: monitor(max): 0.977726
2022-01-25 13:09:26,462 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:26,506 P151130 INFO Train loss: 0.083863
2022-01-25 13:09:26,506 P151130 INFO ************ Epoch=75 end ************
2022-01-25 13:09:31,861 P151130 INFO [Metrics] AUC: 0.977821 - logloss: 0.189253
2022-01-25 13:09:31,861 P151130 INFO Save best model: monitor(max): 0.977821
2022-01-25 13:09:31,864 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:31,914 P151130 INFO Train loss: 0.082829
2022-01-25 13:09:31,914 P151130 INFO ************ Epoch=76 end ************
2022-01-25 13:09:37,133 P151130 INFO [Metrics] AUC: 0.977874 - logloss: 0.189827
2022-01-25 13:09:37,134 P151130 INFO Save best model: monitor(max): 0.977874
2022-01-25 13:09:37,137 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:37,177 P151130 INFO Train loss: 0.081920
2022-01-25 13:09:37,177 P151130 INFO ************ Epoch=77 end ************
2022-01-25 13:09:42,436 P151130 INFO [Metrics] AUC: 0.978004 - logloss: 0.189648
2022-01-25 13:09:42,436 P151130 INFO Save best model: monitor(max): 0.978004
2022-01-25 13:09:42,439 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:42,484 P151130 INFO Train loss: 0.080738
2022-01-25 13:09:42,484 P151130 INFO ************ Epoch=78 end ************
2022-01-25 13:09:47,714 P151130 INFO [Metrics] AUC: 0.978042 - logloss: 0.190110
2022-01-25 13:09:47,715 P151130 INFO Save best model: monitor(max): 0.978042
2022-01-25 13:09:47,717 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:47,761 P151130 INFO Train loss: 0.079549
2022-01-25 13:09:47,761 P151130 INFO ************ Epoch=79 end ************
2022-01-25 13:09:52,947 P151130 INFO [Metrics] AUC: 0.978117 - logloss: 0.190528
2022-01-25 13:09:52,947 P151130 INFO Save best model: monitor(max): 0.978117
2022-01-25 13:09:52,950 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:52,989 P151130 INFO Train loss: 0.079000
2022-01-25 13:09:52,990 P151130 INFO ************ Epoch=80 end ************
2022-01-25 13:09:58,195 P151130 INFO [Metrics] AUC: 0.978238 - logloss: 0.190582
2022-01-25 13:09:58,196 P151130 INFO Save best model: monitor(max): 0.978238
2022-01-25 13:09:58,199 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:09:58,247 P151130 INFO Train loss: 0.077692
2022-01-25 13:09:58,247 P151130 INFO ************ Epoch=81 end ************
2022-01-25 13:10:03,444 P151130 INFO [Metrics] AUC: 0.978263 - logloss: 0.191232
2022-01-25 13:10:03,444 P151130 INFO Save best model: monitor(max): 0.978263
2022-01-25 13:10:03,447 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:03,488 P151130 INFO Train loss: 0.076643
2022-01-25 13:10:03,489 P151130 INFO ************ Epoch=82 end ************
2022-01-25 13:10:08,700 P151130 INFO [Metrics] AUC: 0.978348 - logloss: 0.191340
2022-01-25 13:10:08,701 P151130 INFO Save best model: monitor(max): 0.978348
2022-01-25 13:10:08,704 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:08,743 P151130 INFO Train loss: 0.075748
2022-01-25 13:10:08,743 P151130 INFO ************ Epoch=83 end ************
2022-01-25 13:10:13,918 P151130 INFO [Metrics] AUC: 0.978380 - logloss: 0.191859
2022-01-25 13:10:13,919 P151130 INFO Save best model: monitor(max): 0.978380
2022-01-25 13:10:13,922 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:13,965 P151130 INFO Train loss: 0.074976
2022-01-25 13:10:13,966 P151130 INFO ************ Epoch=84 end ************
2022-01-25 13:10:19,184 P151130 INFO [Metrics] AUC: 0.978442 - logloss: 0.192301
2022-01-25 13:10:19,185 P151130 INFO Save best model: monitor(max): 0.978442
2022-01-25 13:10:19,188 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:19,225 P151130 INFO Train loss: 0.074011
2022-01-25 13:10:19,225 P151130 INFO ************ Epoch=85 end ************
2022-01-25 13:10:24,508 P151130 INFO [Metrics] AUC: 0.978481 - logloss: 0.192496
2022-01-25 13:10:24,508 P151130 INFO Save best model: monitor(max): 0.978481
2022-01-25 13:10:24,511 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:24,558 P151130 INFO Train loss: 0.073279
2022-01-25 13:10:24,559 P151130 INFO ************ Epoch=86 end ************
2022-01-25 13:10:29,954 P151130 INFO [Metrics] AUC: 0.978569 - logloss: 0.192687
2022-01-25 13:10:29,955 P151130 INFO Save best model: monitor(max): 0.978569
2022-01-25 13:10:29,958 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:30,000 P151130 INFO Train loss: 0.072356
2022-01-25 13:10:30,000 P151130 INFO ************ Epoch=87 end ************
2022-01-25 13:10:35,429 P151130 INFO [Metrics] AUC: 0.978592 - logloss: 0.193449
2022-01-25 13:10:35,429 P151130 INFO Save best model: monitor(max): 0.978592
2022-01-25 13:10:35,432 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:35,474 P151130 INFO Train loss: 0.071599
2022-01-25 13:10:35,475 P151130 INFO ************ Epoch=88 end ************
2022-01-25 13:10:40,906 P151130 INFO [Metrics] AUC: 0.978621 - logloss: 0.194155
2022-01-25 13:10:40,906 P151130 INFO Save best model: monitor(max): 0.978621
2022-01-25 13:10:40,909 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:40,949 P151130 INFO Train loss: 0.070588
2022-01-25 13:10:40,949 P151130 INFO ************ Epoch=89 end ************
2022-01-25 13:10:46,325 P151130 INFO [Metrics] AUC: 0.978700 - logloss: 0.194167
2022-01-25 13:10:46,325 P151130 INFO Save best model: monitor(max): 0.978700
2022-01-25 13:10:46,328 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:46,367 P151130 INFO Train loss: 0.070221
2022-01-25 13:10:46,367 P151130 INFO ************ Epoch=90 end ************
2022-01-25 13:10:51,757 P151130 INFO [Metrics] AUC: 0.978737 - logloss: 0.194948
2022-01-25 13:10:51,758 P151130 INFO Save best model: monitor(max): 0.978737
2022-01-25 13:10:51,761 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:51,800 P151130 INFO Train loss: 0.069266
2022-01-25 13:10:51,800 P151130 INFO ************ Epoch=91 end ************
2022-01-25 13:10:57,155 P151130 INFO [Metrics] AUC: 0.978731 - logloss: 0.195443
2022-01-25 13:10:57,155 P151130 INFO Monitor(max) STOP: 0.978731 !
2022-01-25 13:10:57,155 P151130 INFO Reduce learning rate on plateau: 0.000100
2022-01-25 13:10:57,155 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:10:57,201 P151130 INFO Train loss: 0.068575
2022-01-25 13:10:57,201 P151130 INFO ************ Epoch=92 end ************
2022-01-25 13:11:02,505 P151130 INFO [Metrics] AUC: 0.978747 - logloss: 0.195509
2022-01-25 13:11:02,505 P151130 INFO Save best model: monitor(max): 0.978747
2022-01-25 13:11:02,508 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:02,561 P151130 INFO Train loss: 0.066667
2022-01-25 13:11:02,561 P151130 INFO ************ Epoch=93 end ************
2022-01-25 13:11:06,216 P151130 INFO [Metrics] AUC: 0.978762 - logloss: 0.195612
2022-01-25 13:11:06,217 P151130 INFO Save best model: monitor(max): 0.978762
2022-01-25 13:11:06,219 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:06,265 P151130 INFO Train loss: 0.066504
2022-01-25 13:11:06,265 P151130 INFO ************ Epoch=94 end ************
2022-01-25 13:11:09,873 P151130 INFO [Metrics] AUC: 0.978766 - logloss: 0.195639
2022-01-25 13:11:09,874 P151130 INFO Save best model: monitor(max): 0.978766
2022-01-25 13:11:09,877 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:09,920 P151130 INFO Train loss: 0.066406
2022-01-25 13:11:09,920 P151130 INFO ************ Epoch=95 end ************
2022-01-25 13:11:13,461 P151130 INFO [Metrics] AUC: 0.978774 - logloss: 0.195711
2022-01-25 13:11:13,462 P151130 INFO Save best model: monitor(max): 0.978774
2022-01-25 13:11:13,464 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:13,507 P151130 INFO Train loss: 0.066324
2022-01-25 13:11:13,507 P151130 INFO ************ Epoch=96 end ************
2022-01-25 13:11:17,110 P151130 INFO [Metrics] AUC: 0.978777 - logloss: 0.195771
2022-01-25 13:11:17,110 P151130 INFO Save best model: monitor(max): 0.978777
2022-01-25 13:11:17,114 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:17,158 P151130 INFO Train loss: 0.066350
2022-01-25 13:11:17,159 P151130 INFO ************ Epoch=97 end ************
2022-01-25 13:11:20,673 P151130 INFO [Metrics] AUC: 0.978780 - logloss: 0.195864
2022-01-25 13:11:20,673 P151130 INFO Save best model: monitor(max): 0.978780
2022-01-25 13:11:20,676 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:20,715 P151130 INFO Train loss: 0.066241
2022-01-25 13:11:20,715 P151130 INFO ************ Epoch=98 end ************
2022-01-25 13:11:23,043 P151130 INFO [Metrics] AUC: 0.978791 - logloss: 0.195894
2022-01-25 13:11:23,044 P151130 INFO Save best model: monitor(max): 0.978791
2022-01-25 13:11:23,047 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:23,091 P151130 INFO Train loss: 0.066019
2022-01-25 13:11:23,091 P151130 INFO ************ Epoch=99 end ************
2022-01-25 13:11:25,401 P151130 INFO [Metrics] AUC: 0.978790 - logloss: 0.195961
2022-01-25 13:11:25,402 P151130 INFO Monitor(max) STOP: 0.978790 !
2022-01-25 13:11:25,402 P151130 INFO Reduce learning rate on plateau: 0.000010
2022-01-25 13:11:25,402 P151130 INFO --- 50/50 batches finished ---
2022-01-25 13:11:25,442 P151130 INFO Train loss: 0.065954
2022-01-25 13:11:25,442 P151130 INFO ************ Epoch=100 end ************
2022-01-25 13:11:25,442 P151130 INFO Training finished.
2022-01-25 13:11:25,443 P151130 INFO Load best model: /home/ma-user/work/FuxiCTRv1.1/benchmarks/Frappe/FwFM_frappe_x1/frappe_x1_04e961e9/FwFM_frappe_x1_005_95cf3ccd.model
2022-01-25 13:11:25,449 P151130 INFO ****** Validation evaluation ******
2022-01-25 13:11:25,784 P151130 INFO [Metrics] AUC: 0.978791 - logloss: 0.195894
2022-01-25 13:11:25,830 P151130 INFO ******** Test evaluation ********
2022-01-25 13:11:25,830 P151130 INFO Loading data...
2022-01-25 13:11:25,831 P151130 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-25 13:11:25,833 P151130 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-25 13:11:25,833 P151130 INFO Loading test data done.
2022-01-25 13:11:26,085 P151130 INFO [Metrics] AUC: 0.977585 - logloss: 0.202971

```
