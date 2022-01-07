## FM_criteo_x1

A guide to benchmark FM on [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1).

Author: [zhujiem](https://github.com/zhujiem)

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
  CUDA: 10.0.130
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  ```

### Dataset

To reproduce the dataset splitting, please follow the details of [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1) to get data ready.

### Code

We use [FuxiCTR v1.1](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment.

1. Install FuxiCTR and all the dependencies. 
   ```bash
   pip install fuxictr==1.1.*
   ```
   
2. Put the downloaded dataset in `../data/Criteo/Criteo_x1`. 

3. The dataset_config and model_config files are available in the sub-folder `FM_criteo_x1_tuner_config_02`.

   Note that in this setting, we follow the AFN work to fix embedding_dim=10, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons. Other hyper-parameters are tuned via grid search.

4. Run the following script to start.

  ```bash
  nohup python run_expid.py --version pytorch --config Criteo/FM_criteo_x1/FM_criteo_x1_tuner_config_02 --expid FM_criteo_x1_001_8f8d954b --gpu 0 > run.log & 
  tail -f run.log
  ```

### Results
```python
[Metrics] AUC: 0.802157 - logloss: 0.449063
```

### Logs
```python
2021-12-28 10:43:40,934 P44588 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_criteo_x1_001_8f8d954b",
    "model_root": "./Criteo/FM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-12-28 10:43:40,935 P44588 INFO Set up feature encoder...
2021-12-28 10:43:40,935 P44588 INFO Reading file: ../data/Criteo/Criteo_x1/train.csv
2021-12-28 10:45:18,145 P44588 INFO Reading file: ../data/Criteo/Criteo_x1/valid.csv
2021-12-28 10:45:39,549 P44588 INFO Reading file: ../data/Criteo/Criteo_x1/test.csv
2021-12-28 10:45:51,733 P44588 INFO Preprocess feature columns...
2021-12-28 10:45:56,916 P44588 INFO Fit feature encoder...
2021-12-28 10:45:56,916 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I1', 'type': 'numeric'}
2021-12-28 10:45:56,916 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I2', 'type': 'numeric'}
2021-12-28 10:45:56,916 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I3', 'type': 'numeric'}
2021-12-28 10:45:56,916 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I4', 'type': 'numeric'}
2021-12-28 10:45:56,916 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I5', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I6', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I7', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I8', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I9', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I10', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I11', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I12', 'type': 'numeric'}
2021-12-28 10:45:56,917 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'I13', 'type': 'numeric'}
2021-12-28 10:45:56,918 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C1', 'type': 'categorical'}
2021-12-28 10:46:05,438 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C2', 'type': 'categorical'}
2021-12-28 10:46:13,612 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C3', 'type': 'categorical'}
2021-12-28 10:46:27,060 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C4', 'type': 'categorical'}
2021-12-28 10:46:38,585 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C5', 'type': 'categorical'}
2021-12-28 10:46:46,391 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C6', 'type': 'categorical'}
2021-12-28 10:46:54,063 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C7', 'type': 'categorical'}
2021-12-28 10:47:03,431 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C8', 'type': 'categorical'}
2021-12-28 10:47:11,367 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C9', 'type': 'categorical'}
2021-12-28 10:47:18,970 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C10', 'type': 'categorical'}
2021-12-28 10:47:28,847 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C11', 'type': 'categorical'}
2021-12-28 10:47:38,188 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C12', 'type': 'categorical'}
2021-12-28 10:47:52,684 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C13', 'type': 'categorical'}
2021-12-28 10:48:02,687 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C14', 'type': 'categorical'}
2021-12-28 10:48:10,467 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C15', 'type': 'categorical'}
2021-12-28 10:48:19,855 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C16', 'type': 'categorical'}
2021-12-28 10:48:33,003 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C17', 'type': 'categorical'}
2021-12-28 10:48:41,009 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C18', 'type': 'categorical'}
2021-12-28 10:48:49,893 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C19', 'type': 'categorical'}
2021-12-28 10:48:58,279 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C20', 'type': 'categorical'}
2021-12-28 10:49:06,049 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C21', 'type': 'categorical'}
2021-12-28 10:49:19,443 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C22', 'type': 'categorical'}
2021-12-28 10:49:27,096 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C23', 'type': 'categorical'}
2021-12-28 10:49:34,871 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C24', 'type': 'categorical'}
2021-12-28 10:49:44,639 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C25', 'type': 'categorical'}
2021-12-28 10:49:52,593 P44588 INFO Processing column: {'active': True, 'dtype': 'float', 'name': 'C26', 'type': 'categorical'}
2021-12-28 10:50:02,363 P44588 INFO Set feature index...
2021-12-28 10:50:02,363 P44588 INFO Pickle feature_encode: ../data/Criteo/criteo_x1_7b681156/feature_encoder.pkl
2021-12-28 10:50:10,544 P44588 INFO Save feature_map to json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2021-12-28 10:50:10,545 P44588 INFO Set feature encoder done.
2021-12-28 10:50:10,545 P44588 INFO Transform feature columns...
2021-12-28 10:59:52,092 P44588 INFO Saving data to h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2021-12-28 10:59:58,873 P44588 INFO Preprocess feature columns...
2021-12-28 11:00:00,311 P44588 INFO Transform feature columns...
2021-12-28 11:02:26,541 P44588 INFO Saving data to h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2021-12-28 11:02:28,115 P44588 INFO Preprocess feature columns...
2021-12-28 11:02:29,071 P44588 INFO Transform feature columns...
2021-12-28 11:03:49,335 P44588 INFO Saving data to h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2021-12-28 11:03:50,220 P44588 INFO Transform csv data to h5 done.
2021-12-28 11:03:50,220 P44588 INFO Loading data...
2021-12-28 11:03:50,224 P44588 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2021-12-28 11:03:55,573 P44588 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2021-12-28 11:03:56,890 P44588 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2021-12-28 11:03:56,890 P44588 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2021-12-28 11:03:56,890 P44588 INFO Loading train data done.
2021-12-28 11:03:57,958 P44588 INFO Total number of parameters: 22949477.
2021-12-28 11:04:01,122 P44588 INFO Start training: 8058 batches/epoch
2021-12-28 11:04:01,122 P44588 INFO ************ Epoch=1 start ************
2021-12-28 11:15:08,128 P44588 INFO [Metrics] AUC: 0.793791 - logloss: 0.456591
2021-12-28 11:15:08,129 P44588 INFO Save best model: monitor(max): 0.793791
2021-12-28 11:15:08,221 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 11:15:08,589 P44588 INFO Train loss: 0.471009
2021-12-28 11:15:08,589 P44588 INFO ************ Epoch=1 end ************
2021-12-28 11:26:23,477 P44588 INFO [Metrics] AUC: 0.795107 - logloss: 0.455468
2021-12-28 11:26:23,478 P44588 INFO Save best model: monitor(max): 0.795107
2021-12-28 11:26:23,601 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 11:26:23,963 P44588 INFO Train loss: 0.465477
2021-12-28 11:26:23,963 P44588 INFO ************ Epoch=2 end ************
2021-12-28 11:46:46,427 P44588 INFO [Metrics] AUC: 0.795598 - logloss: 0.454960
2021-12-28 11:46:46,427 P44588 INFO Save best model: monitor(max): 0.795598
2021-12-28 11:46:46,563 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 11:46:47,009 P44588 INFO Train loss: 0.464934
2021-12-28 11:46:47,009 P44588 INFO ************ Epoch=3 end ************
2021-12-28 12:47:50,219 P44588 INFO [Metrics] AUC: 0.795763 - logloss: 0.454858
2021-12-28 12:47:50,220 P44588 INFO Save best model: monitor(max): 0.795763
2021-12-28 12:47:50,359 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 12:47:50,729 P44588 INFO Train loss: 0.464727
2021-12-28 12:47:50,729 P44588 INFO ************ Epoch=4 end ************
2021-12-28 13:37:39,219 P44588 INFO [Metrics] AUC: 0.795982 - logloss: 0.454646
2021-12-28 13:37:39,220 P44588 INFO Save best model: monitor(max): 0.795982
2021-12-28 13:37:39,376 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 13:37:39,750 P44588 INFO Train loss: 0.464639
2021-12-28 13:37:39,750 P44588 INFO ************ Epoch=5 end ************
2021-12-28 14:37:09,609 P44588 INFO [Metrics] AUC: 0.795896 - logloss: 0.454714
2021-12-28 14:37:09,609 P44588 INFO Monitor(max) STOP: 0.795896 !
2021-12-28 14:37:09,610 P44588 INFO Reduce learning rate on plateau: 0.000100
2021-12-28 14:37:09,610 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 14:37:09,961 P44588 INFO Train loss: 0.464562
2021-12-28 14:37:09,962 P44588 INFO ************ Epoch=6 end ************
2021-12-28 15:27:13,343 P44588 INFO [Metrics] AUC: 0.799876 - logloss: 0.451159
2021-12-28 15:27:13,344 P44588 INFO Save best model: monitor(max): 0.799876
2021-12-28 15:27:13,485 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 15:27:13,849 P44588 INFO Train loss: 0.454956
2021-12-28 15:27:13,849 P44588 INFO ************ Epoch=7 end ************
2021-12-28 16:45:54,382 P44588 INFO [Metrics] AUC: 0.800415 - logloss: 0.450699
2021-12-28 16:45:54,383 P44588 INFO Save best model: monitor(max): 0.800415
2021-12-28 16:45:54,528 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 16:45:54,848 P44588 INFO Train loss: 0.452030
2021-12-28 16:45:54,848 P44588 INFO ************ Epoch=8 end ************
2021-12-28 17:59:52,755 P44588 INFO [Metrics] AUC: 0.800658 - logloss: 0.450475
2021-12-28 17:59:52,756 P44588 INFO Save best model: monitor(max): 0.800658
2021-12-28 17:59:52,897 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 17:59:53,249 P44588 INFO Train loss: 0.451344
2021-12-28 17:59:53,249 P44588 INFO ************ Epoch=9 end ************
2021-12-28 18:54:00,960 P44588 INFO [Metrics] AUC: 0.800839 - logloss: 0.450337
2021-12-28 18:54:00,961 P44588 INFO Save best model: monitor(max): 0.800839
2021-12-28 18:54:01,107 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 18:54:01,467 P44588 INFO Train loss: 0.450972
2021-12-28 18:54:01,467 P44588 INFO ************ Epoch=10 end ************
2021-12-28 20:12:22,340 P44588 INFO [Metrics] AUC: 0.800953 - logloss: 0.450240
2021-12-28 20:12:22,341 P44588 INFO Save best model: monitor(max): 0.800953
2021-12-28 20:12:22,497 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 20:12:22,850 P44588 INFO Train loss: 0.450701
2021-12-28 20:12:22,851 P44588 INFO ************ Epoch=11 end ************
2021-12-28 21:31:22,453 P44588 INFO [Metrics] AUC: 0.801014 - logloss: 0.450158
2021-12-28 21:31:22,454 P44588 INFO Save best model: monitor(max): 0.801014
2021-12-28 21:31:22,606 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 21:31:22,967 P44588 INFO Train loss: 0.450469
2021-12-28 21:31:22,967 P44588 INFO ************ Epoch=12 end ************
2021-12-28 22:50:22,438 P44588 INFO [Metrics] AUC: 0.801136 - logloss: 0.450061
2021-12-28 22:50:22,439 P44588 INFO Save best model: monitor(max): 0.801136
2021-12-28 22:50:22,589 P44588 INFO --- 8058/8058 batches finished ---
2021-12-28 22:50:22,959 P44588 INFO Train loss: 0.450266
2021-12-28 22:50:22,959 P44588 INFO ************ Epoch=13 end ************
2021-12-29 00:07:39,997 P44588 INFO [Metrics] AUC: 0.801240 - logloss: 0.449983
2021-12-29 00:07:39,998 P44588 INFO Save best model: monitor(max): 0.801240
2021-12-29 00:07:40,156 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 00:07:40,535 P44588 INFO Train loss: 0.450081
2021-12-29 00:07:40,536 P44588 INFO ************ Epoch=14 end ************
2021-12-29 01:25:41,603 P44588 INFO [Metrics] AUC: 0.801272 - logloss: 0.449947
2021-12-29 01:25:41,604 P44588 INFO Save best model: monitor(max): 0.801272
2021-12-29 01:25:41,739 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 01:25:42,120 P44588 INFO Train loss: 0.449894
2021-12-29 01:25:42,120 P44588 INFO ************ Epoch=15 end ************
2021-12-29 02:42:51,862 P44588 INFO [Metrics] AUC: 0.801339 - logloss: 0.449896
2021-12-29 02:42:51,864 P44588 INFO Save best model: monitor(max): 0.801339
2021-12-29 02:42:52,019 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 02:42:52,428 P44588 INFO Train loss: 0.449727
2021-12-29 02:42:52,428 P44588 INFO ************ Epoch=16 end ************
2021-12-29 04:04:18,429 P44588 INFO [Metrics] AUC: 0.801422 - logloss: 0.449828
2021-12-29 04:04:18,430 P44588 INFO Save best model: monitor(max): 0.801422
2021-12-29 04:04:18,575 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 04:04:18,941 P44588 INFO Train loss: 0.449564
2021-12-29 04:04:18,942 P44588 INFO ************ Epoch=17 end ************
2021-12-29 05:22:14,245 P44588 INFO [Metrics] AUC: 0.801474 - logloss: 0.449796
2021-12-29 05:22:14,246 P44588 INFO Save best model: monitor(max): 0.801474
2021-12-29 05:22:14,391 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 05:22:14,766 P44588 INFO Train loss: 0.449415
2021-12-29 05:22:14,766 P44588 INFO ************ Epoch=18 end ************
2021-12-29 06:42:16,934 P44588 INFO [Metrics] AUC: 0.801491 - logloss: 0.449771
2021-12-29 06:42:16,935 P44588 INFO Save best model: monitor(max): 0.801491
2021-12-29 06:42:17,087 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 06:42:17,428 P44588 INFO Train loss: 0.449265
2021-12-29 06:42:17,428 P44588 INFO ************ Epoch=19 end ************
2021-12-29 08:03:36,267 P44588 INFO [Metrics] AUC: 0.801552 - logloss: 0.449739
2021-12-29 08:03:36,267 P44588 INFO Save best model: monitor(max): 0.801552
2021-12-29 08:03:36,415 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 08:03:36,791 P44588 INFO Train loss: 0.449127
2021-12-29 08:03:36,792 P44588 INFO ************ Epoch=20 end ************
2021-12-29 08:43:40,375 P44588 INFO [Metrics] AUC: 0.801597 - logloss: 0.449683
2021-12-29 08:43:40,376 P44588 INFO Save best model: monitor(max): 0.801597
2021-12-29 08:43:40,507 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 08:43:40,847 P44588 INFO Train loss: 0.448997
2021-12-29 08:43:40,847 P44588 INFO ************ Epoch=21 end ************
2021-12-29 08:54:44,194 P44588 INFO [Metrics] AUC: 0.801595 - logloss: 0.449691
2021-12-29 08:54:44,194 P44588 INFO Monitor(max) STOP: 0.801595 !
2021-12-29 08:54:44,195 P44588 INFO Reduce learning rate on plateau: 0.000010
2021-12-29 08:54:44,195 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 08:54:44,541 P44588 INFO Train loss: 0.448870
2021-12-29 08:54:44,541 P44588 INFO ************ Epoch=22 end ************
2021-12-29 09:05:59,230 P44588 INFO [Metrics] AUC: 0.801839 - logloss: 0.449461
2021-12-29 09:05:59,231 P44588 INFO Save best model: monitor(max): 0.801839
2021-12-29 09:05:59,384 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 09:05:59,799 P44588 INFO Train loss: 0.446134
2021-12-29 09:05:59,799 P44588 INFO ************ Epoch=23 end ************
2021-12-29 09:16:58,998 P44588 INFO [Metrics] AUC: 0.801877 - logloss: 0.449429
2021-12-29 09:16:58,999 P44588 INFO Save best model: monitor(max): 0.801877
2021-12-29 09:16:59,161 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 09:16:59,531 P44588 INFO Train loss: 0.446013
2021-12-29 09:16:59,531 P44588 INFO ************ Epoch=24 end ************
2021-12-29 09:28:01,430 P44588 INFO [Metrics] AUC: 0.801892 - logloss: 0.449415
2021-12-29 09:28:01,430 P44588 INFO Save best model: monitor(max): 0.801892
2021-12-29 09:28:01,570 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 09:28:01,933 P44588 INFO Train loss: 0.445960
2021-12-29 09:28:01,934 P44588 INFO ************ Epoch=25 end ************
2021-12-29 09:39:04,385 P44588 INFO [Metrics] AUC: 0.801906 - logloss: 0.449403
2021-12-29 09:39:04,386 P44588 INFO Save best model: monitor(max): 0.801906
2021-12-29 09:39:04,521 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 09:39:04,900 P44588 INFO Train loss: 0.445922
2021-12-29 09:39:04,900 P44588 INFO ************ Epoch=26 end ************
2021-12-29 09:50:09,825 P44588 INFO [Metrics] AUC: 0.801910 - logloss: 0.449401
2021-12-29 09:50:09,826 P44588 INFO Save best model: monitor(max): 0.801910
2021-12-29 09:50:09,980 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 09:50:10,333 P44588 INFO Train loss: 0.445891
2021-12-29 09:50:10,334 P44588 INFO ************ Epoch=27 end ************
2021-12-29 10:01:13,097 P44588 INFO [Metrics] AUC: 0.801912 - logloss: 0.449398
2021-12-29 10:01:13,098 P44588 INFO Save best model: monitor(max): 0.801912
2021-12-29 10:01:13,231 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 10:01:13,617 P44588 INFO Train loss: 0.445861
2021-12-29 10:01:13,617 P44588 INFO ************ Epoch=28 end ************
2021-12-29 10:12:14,989 P44588 INFO [Metrics] AUC: 0.801917 - logloss: 0.449395
2021-12-29 10:12:14,990 P44588 INFO Save best model: monitor(max): 0.801917
2021-12-29 10:12:15,136 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 10:12:15,503 P44588 INFO Train loss: 0.445835
2021-12-29 10:12:15,503 P44588 INFO ************ Epoch=29 end ************
2021-12-29 10:23:14,898 P44588 INFO [Metrics] AUC: 0.801924 - logloss: 0.449392
2021-12-29 10:23:14,898 P44588 INFO Save best model: monitor(max): 0.801924
2021-12-29 10:23:15,030 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 10:23:15,407 P44588 INFO Train loss: 0.445810
2021-12-29 10:23:15,407 P44588 INFO ************ Epoch=30 end ************
2021-12-29 10:34:16,684 P44588 INFO [Metrics] AUC: 0.801928 - logloss: 0.449388
2021-12-29 10:34:16,685 P44588 INFO Save best model: monitor(max): 0.801928
2021-12-29 10:34:16,818 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 10:34:17,188 P44588 INFO Train loss: 0.445783
2021-12-29 10:34:17,188 P44588 INFO ************ Epoch=31 end ************
2021-12-29 10:45:22,993 P44588 INFO [Metrics] AUC: 0.801932 - logloss: 0.449383
2021-12-29 10:45:22,994 P44588 INFO Save best model: monitor(max): 0.801932
2021-12-29 10:45:23,128 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 10:45:23,504 P44588 INFO Train loss: 0.445757
2021-12-29 10:45:23,504 P44588 INFO ************ Epoch=32 end ************
2021-12-29 10:56:23,132 P44588 INFO [Metrics] AUC: 0.801932 - logloss: 0.449384
2021-12-29 10:56:23,133 P44588 INFO Monitor(max) STOP: 0.801932 !
2021-12-29 10:56:23,133 P44588 INFO Reduce learning rate on plateau: 0.000001
2021-12-29 10:56:23,133 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 10:56:23,509 P44588 INFO Train loss: 0.445729
2021-12-29 10:56:23,509 P44588 INFO ************ Epoch=33 end ************
2021-12-29 11:07:23,064 P44588 INFO [Metrics] AUC: 0.801936 - logloss: 0.449381
2021-12-29 11:07:23,065 P44588 INFO Save best model: monitor(max): 0.801936
2021-12-29 11:07:23,198 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 11:07:23,588 P44588 INFO Train loss: 0.445332
2021-12-29 11:07:23,589 P44588 INFO ************ Epoch=34 end ************
2021-12-29 11:18:23,084 P44588 INFO [Metrics] AUC: 0.801937 - logloss: 0.449380
2021-12-29 11:18:23,085 P44588 INFO Save best model: monitor(max): 0.801937
2021-12-29 11:18:23,217 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 11:18:23,586 P44588 INFO Train loss: 0.445329
2021-12-29 11:18:23,586 P44588 INFO ************ Epoch=35 end ************
2021-12-29 11:29:21,783 P44588 INFO [Metrics] AUC: 0.801938 - logloss: 0.449380
2021-12-29 11:29:21,784 P44588 INFO Monitor(max) STOP: 0.801938 !
2021-12-29 11:29:21,784 P44588 INFO Reduce learning rate on plateau: 0.000001
2021-12-29 11:29:21,784 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 11:29:22,151 P44588 INFO Train loss: 0.445327
2021-12-29 11:29:22,151 P44588 INFO ************ Epoch=36 end ************
2021-12-29 11:40:54,482 P44588 INFO [Metrics] AUC: 0.801938 - logloss: 0.449380
2021-12-29 11:40:54,483 P44588 INFO Monitor(max) STOP: 0.801938 !
2021-12-29 11:40:54,483 P44588 INFO Reduce learning rate on plateau: 0.000001
2021-12-29 11:40:54,483 P44588 INFO Early stopping at epoch=37
2021-12-29 11:40:54,483 P44588 INFO --- 8058/8058 batches finished ---
2021-12-29 11:40:54,849 P44588 INFO Train loss: 0.445326
2021-12-29 11:40:54,849 P44588 INFO Training finished.
2021-12-29 11:40:54,849 P44588 INFO Load best model: /home/xx/FuxiCTR_v1.1/benchmarks/Criteo/FM_criteo_x1/criteo_x1_7b681156/FM_criteo_x1_001_8f8d954b.model
2021-12-29 11:40:54,937 P44588 INFO ****** Validation evaluation ******
2021-12-29 11:41:24,280 P44588 INFO [Metrics] AUC: 0.801937 - logloss: 0.449380
2021-12-29 11:41:24,320 P44588 INFO ******** Test evaluation ********
2021-12-29 11:41:24,320 P44588 INFO Loading data...
2021-12-29 11:41:24,320 P44588 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2021-12-29 11:41:25,404 P44588 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2021-12-29 11:41:25,404 P44588 INFO Loading test data done.
2021-12-29 11:41:42,369 P44588 INFO [Metrics] AUC: 0.802157 - logloss: 0.449063

```
