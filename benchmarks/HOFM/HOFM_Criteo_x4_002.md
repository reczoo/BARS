## HOFM_Criteo_x4_002

A notebook to benchmark HOFM on Criteo_x4_002 dataset.

Author: [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  RAM: 500G+
  ```
+ Software

  ```python
  python: 3.6.5
  pandas: 1.0.0
  numpy: 1.18.1
  ```

### Dataset
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=2.

We fix embedding_dim=40 in this setting.
### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [HOFM_criteo_x4_tuner_config_05.yaml](./002/HOFM_criteo_x4_tuner_config_05.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/HOFM_criteo_x4_tuner_config_05.yaml --tag 001 --gpu 0
  ```
### Results
```python
[Metrics] logloss: 0.440415 - AUC: 0.811455
```


### Logs
```python
2020-02-25 12:03:08,272 P590 INFO {
    "batch_size": "3000",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "[40, 5]",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HOFM",
    "model_id": "HOFM_criteo_x4_001_f22c1010",
    "model_root": "./Criteo/HOFM_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "order": "3",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
    "reuse_embedding": "False",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-02-25 12:03:08,280 P590 INFO Set up feature encoder...
2020-02-25 12:03:08,280 P590 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-25 12:03:08,280 P590 INFO Loading data...
2020-02-25 12:03:08,290 P590 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-25 12:03:13,000 P590 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-25 12:03:14,899 P590 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-25 12:03:15,125 P590 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-25 12:03:15,125 P590 INFO Loading train data done.
2020-02-25 12:03:29,826 P590 INFO **** Start training: 12225 batches/epoch ****
2020-02-25 13:45:03,226 P590 INFO [Metrics] logloss: 0.452741 - AUC: 0.797658
2020-02-25 13:45:03,310 P590 INFO Save best model: monitor(max): 0.344917
2020-02-25 13:45:04,674 P590 INFO --- 12225/12225 batches finished ---
2020-02-25 13:45:04,730 P590 INFO Train loss: 0.482865
2020-02-25 13:45:04,730 P590 INFO ************ Epoch=1 end ************
2020-02-25 15:26:52,314 P590 INFO [Metrics] logloss: 0.452367 - AUC: 0.798152
2020-02-25 15:26:52,398 P590 INFO Save best model: monitor(max): 0.345785
2020-02-25 15:26:54,534 P590 INFO --- 12225/12225 batches finished ---
2020-02-25 15:26:54,606 P590 INFO Train loss: 0.481838
2020-02-25 15:26:54,607 P590 INFO ************ Epoch=2 end ************
2020-02-25 17:08:26,380 P590 INFO [Metrics] logloss: 0.451993 - AUC: 0.798611
2020-02-25 17:08:26,470 P590 INFO Save best model: monitor(max): 0.346618
2020-02-25 17:08:28,883 P590 INFO --- 12225/12225 batches finished ---
2020-02-25 17:08:28,957 P590 INFO Train loss: 0.481575
2020-02-25 17:08:28,957 P590 INFO ************ Epoch=3 end ************
2020-02-25 18:50:21,243 P590 INFO [Metrics] logloss: 0.452063 - AUC: 0.798448
2020-02-25 18:50:21,371 P590 INFO Monitor(max) STOP: 0.346385 !
2020-02-25 18:50:21,371 P590 INFO Reduce learning rate on plateau: 0.000100
2020-02-25 18:50:21,371 P590 INFO --- 12225/12225 batches finished ---
2020-02-25 18:50:21,436 P590 INFO Train loss: 0.481441
2020-02-25 18:50:21,436 P590 INFO ************ Epoch=4 end ************
2020-02-25 20:32:12,415 P590 INFO [Metrics] logloss: 0.444316 - AUC: 0.807045
2020-02-25 20:32:12,511 P590 INFO Save best model: monitor(max): 0.362729
2020-02-25 20:32:14,857 P590 INFO --- 12225/12225 batches finished ---
2020-02-25 20:32:14,924 P590 INFO Train loss: 0.454585
2020-02-25 20:32:14,924 P590 INFO ************ Epoch=5 end ************
2020-02-25 22:14:09,762 P590 INFO [Metrics] logloss: 0.443294 - AUC: 0.808196
2020-02-25 22:14:09,838 P590 INFO Save best model: monitor(max): 0.364902
2020-02-25 22:14:12,270 P590 INFO --- 12225/12225 batches finished ---
2020-02-25 22:14:12,339 P590 INFO Train loss: 0.448891
2020-02-25 22:14:12,339 P590 INFO ************ Epoch=6 end ************
2020-02-25 23:56:14,466 P590 INFO [Metrics] logloss: 0.442748 - AUC: 0.808801
2020-02-25 23:56:14,547 P590 INFO Save best model: monitor(max): 0.366052
2020-02-25 23:56:16,640 P590 INFO --- 12225/12225 batches finished ---
2020-02-25 23:56:16,716 P590 INFO Train loss: 0.447550
2020-02-25 23:56:16,716 P590 INFO ************ Epoch=7 end ************
2020-02-26 01:40:05,889 P590 INFO [Metrics] logloss: 0.442507 - AUC: 0.809088
2020-02-26 01:40:05,975 P590 INFO Save best model: monitor(max): 0.366581
2020-02-26 01:40:08,057 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 01:40:08,126 P590 INFO Train loss: 0.446761
2020-02-26 01:40:08,126 P590 INFO ************ Epoch=8 end ************
2020-02-26 03:23:05,957 P590 INFO [Metrics] logloss: 0.442221 - AUC: 0.809419
2020-02-26 03:23:06,059 P590 INFO Save best model: monitor(max): 0.367197
2020-02-26 03:23:08,382 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 03:23:08,442 P590 INFO Train loss: 0.446199
2020-02-26 03:23:08,442 P590 INFO ************ Epoch=9 end ************
2020-02-26 05:05:25,359 P590 INFO [Metrics] logloss: 0.442074 - AUC: 0.809582
2020-02-26 05:05:25,442 P590 INFO Save best model: monitor(max): 0.367508
2020-02-26 05:05:28,039 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 05:05:28,133 P590 INFO Train loss: 0.445765
2020-02-26 05:05:28,133 P590 INFO ************ Epoch=10 end ************
2020-02-26 06:47:48,549 P590 INFO [Metrics] logloss: 0.441956 - AUC: 0.809719
2020-02-26 06:47:48,629 P590 INFO Save best model: monitor(max): 0.367764
2020-02-26 06:47:51,034 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 06:47:51,098 P590 INFO Train loss: 0.445435
2020-02-26 06:47:51,098 P590 INFO ************ Epoch=11 end ************
2020-02-26 08:29:45,645 P590 INFO [Metrics] logloss: 0.441873 - AUC: 0.809806
2020-02-26 08:29:45,722 P590 INFO Save best model: monitor(max): 0.367933
2020-02-26 08:29:48,007 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 08:29:48,110 P590 INFO Train loss: 0.445150
2020-02-26 08:29:48,110 P590 INFO ************ Epoch=12 end ************
2020-02-26 10:11:19,699 P590 INFO [Metrics] logloss: 0.441783 - AUC: 0.809884
2020-02-26 10:11:19,805 P590 INFO Save best model: monitor(max): 0.368100
2020-02-26 10:11:21,939 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 10:11:22,006 P590 INFO Train loss: 0.444931
2020-02-26 10:11:22,006 P590 INFO ************ Epoch=13 end ************
2020-02-26 11:52:58,495 P590 INFO [Metrics] logloss: 0.441773 - AUC: 0.809927
2020-02-26 11:52:58,576 P590 INFO Save best model: monitor(max): 0.368153
2020-02-26 11:53:00,730 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 11:53:00,803 P590 INFO Train loss: 0.444736
2020-02-26 11:53:00,803 P590 INFO ************ Epoch=14 end ************
2020-02-26 13:34:29,840 P590 INFO [Metrics] logloss: 0.441717 - AUC: 0.809983
2020-02-26 13:34:29,969 P590 INFO Save best model: monitor(max): 0.368266
2020-02-26 13:34:32,075 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 13:34:32,157 P590 INFO Train loss: 0.444588
2020-02-26 13:34:32,157 P590 INFO ************ Epoch=15 end ************
2020-02-26 15:16:00,784 P590 INFO [Metrics] logloss: 0.441716 - AUC: 0.809997
2020-02-26 15:16:00,873 P590 INFO Save best model: monitor(max): 0.368280
2020-02-26 15:16:03,098 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 15:16:03,179 P590 INFO Train loss: 0.444451
2020-02-26 15:16:03,179 P590 INFO ************ Epoch=16 end ************
2020-02-26 16:57:37,487 P590 INFO [Metrics] logloss: 0.441678 - AUC: 0.810023
2020-02-26 16:57:37,592 P590 INFO Save best model: monitor(max): 0.368345
2020-02-26 16:57:39,674 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 16:57:39,787 P590 INFO Train loss: 0.444340
2020-02-26 16:57:39,788 P590 INFO ************ Epoch=17 end ************
2020-02-26 18:39:10,375 P590 INFO [Metrics] logloss: 0.441627 - AUC: 0.810087
2020-02-26 18:39:10,455 P590 INFO Save best model: monitor(max): 0.368460
2020-02-26 18:39:12,582 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 18:39:12,655 P590 INFO Train loss: 0.444234
2020-02-26 18:39:12,655 P590 INFO ************ Epoch=18 end ************
2020-02-26 20:21:07,949 P590 INFO [Metrics] logloss: 0.441826 - AUC: 0.809940
2020-02-26 20:21:08,068 P590 INFO Monitor(max) STOP: 0.368114 !
2020-02-26 20:21:08,068 P590 INFO Reduce learning rate on plateau: 0.000010
2020-02-26 20:21:08,068 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 20:21:08,141 P590 INFO Train loss: 0.444146
2020-02-26 20:21:08,141 P590 INFO ************ Epoch=19 end ************
2020-02-26 22:02:28,258 P590 INFO [Metrics] logloss: 0.440848 - AUC: 0.810913
2020-02-26 22:02:28,392 P590 INFO Save best model: monitor(max): 0.370065
2020-02-26 22:02:30,483 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 22:02:30,570 P590 INFO Train loss: 0.437630
2020-02-26 22:02:30,570 P590 INFO ************ Epoch=20 end ************
2020-02-26 23:44:00,608 P590 INFO [Metrics] logloss: 0.440753 - AUC: 0.811035
2020-02-26 23:44:00,690 P590 INFO Save best model: monitor(max): 0.370282
2020-02-26 23:44:02,929 P590 INFO --- 12225/12225 batches finished ---
2020-02-26 23:44:03,020 P590 INFO Train loss: 0.436947
2020-02-26 23:44:03,020 P590 INFO ************ Epoch=21 end ************
2020-02-27 01:25:34,359 P590 INFO [Metrics] logloss: 0.440714 - AUC: 0.811063
2020-02-27 01:25:34,438 P590 INFO Save best model: monitor(max): 0.370349
2020-02-27 01:25:36,688 P590 INFO --- 12225/12225 batches finished ---
2020-02-27 01:25:36,753 P590 INFO Train loss: 0.436734
2020-02-27 01:25:36,753 P590 INFO ************ Epoch=22 end ************
2020-02-27 03:08:07,282 P590 INFO [Metrics] logloss: 0.440696 - AUC: 0.811085
2020-02-27 03:08:07,377 P590 INFO Save best model: monitor(max): 0.370389
2020-02-27 03:08:09,582 P590 INFO --- 12225/12225 batches finished ---
2020-02-27 03:08:09,661 P590 INFO Train loss: 0.436594
2020-02-27 03:08:09,661 P590 INFO ************ Epoch=23 end ************
2020-02-27 04:49:52,450 P590 INFO [Metrics] logloss: 0.440695 - AUC: 0.811095
2020-02-27 04:49:52,532 P590 INFO Save best model: monitor(max): 0.370400
2020-02-27 04:49:54,769 P590 INFO --- 12225/12225 batches finished ---
2020-02-27 04:49:54,853 P590 INFO Train loss: 0.436488
2020-02-27 04:49:54,853 P590 INFO ************ Epoch=24 end ************
2020-02-27 06:31:51,431 P590 INFO [Metrics] logloss: 0.440696 - AUC: 0.811087
2020-02-27 06:31:51,513 P590 INFO Monitor(max) STOP: 0.370391 !
2020-02-27 06:31:51,513 P590 INFO Reduce learning rate on plateau: 0.000001
2020-02-27 06:31:51,513 P590 INFO --- 12225/12225 batches finished ---
2020-02-27 06:31:51,577 P590 INFO Train loss: 0.436397
2020-02-27 06:31:51,577 P590 INFO ************ Epoch=25 end ************
2020-02-27 08:13:44,541 P590 INFO [Metrics] logloss: 0.440684 - AUC: 0.811097
2020-02-27 08:13:44,625 P590 INFO Save best model: monitor(max): 0.370413
2020-02-27 08:13:46,818 P590 INFO --- 12225/12225 batches finished ---
2020-02-27 08:13:46,892 P590 INFO Train loss: 0.435255
2020-02-27 08:13:46,892 P590 INFO ************ Epoch=26 end ************
2020-02-27 09:55:32,778 P590 INFO [Metrics] logloss: 0.440687 - AUC: 0.811100
2020-02-27 09:55:32,878 P590 INFO Monitor(max) STOP: 0.370413 !
2020-02-27 09:55:32,878 P590 INFO Reduce learning rate on plateau: 0.000001
2020-02-27 09:55:32,878 P590 INFO --- 12225/12225 batches finished ---
2020-02-27 09:55:32,943 P590 INFO Train loss: 0.435281
2020-02-27 09:55:32,943 P590 INFO ************ Epoch=27 end ************
2020-02-27 11:36:50,355 P590 INFO [Metrics] logloss: 0.440687 - AUC: 0.811098
2020-02-27 11:36:50,456 P590 INFO Monitor(max) STOP: 0.370410 !
2020-02-27 11:36:50,456 P590 INFO Reduce learning rate on plateau: 0.000001
2020-02-27 11:36:50,456 P590 INFO Early stopping at epoch=28
2020-02-27 11:36:50,456 P590 INFO --- 12225/12225 batches finished ---
2020-02-27 11:36:50,514 P590 INFO Train loss: 0.435295
2020-02-27 11:36:50,514 P590 INFO Training finished.
2020-02-27 11:36:50,514 P590 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/HOFM_criteo/criteo_x4_001_be98441d/HOFM_criteo_x4_001_f22c1010_criteo_x4_001_be98441d_model.ckpt
2020-02-27 11:36:52,151 P590 INFO ****** Train/validation evaluation ******
2020-02-27 12:00:59,862 P590 INFO [Metrics] logloss: 0.427715 - AUC: 0.825014
2020-02-27 12:02:13,083 P590 INFO [Metrics] logloss: 0.440684 - AUC: 0.811097
2020-02-27 12:02:13,270 P590 INFO ******** Test evaluation ********
2020-02-27 12:02:13,270 P590 INFO Loading data...
2020-02-27 12:02:13,270 P590 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-27 12:02:14,369 P590 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-27 12:02:14,370 P590 INFO Loading test data done.
2020-02-27 12:03:26,781 P590 INFO [Metrics] logloss: 0.440415 - AUC: 0.811455


```