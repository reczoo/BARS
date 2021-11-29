## HOFM_Criteo_x4_001 

A notebook to benchmark HOFM on Criteo_x4_001 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2
(x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default ``<OOV>`` token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better. 

To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.


### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [HOFM_criteo_x4_tuner_config_01-001.yaml](./HOFM_criteo_x4_tuner_config_01-001.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/HOFM_criteo_x4_tuner_config_01-001.yaml --tag 001 --gpu 0
  ```



### Results
```python
[Metrics] logloss: 0.441079 - AUC: 0.810747
```


### Logs
```python
2020-07-08 21:01:51,736 P45734 INFO Set up feature encoder...
2020-07-08 21:12:35,797 P47106 INFO {
    "batch_size": "3000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "[16, 16]",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HOFM",
    "model_id": "HOFM_criteo_x4_5c863b0f_001_cef08737",
    "model_root": "./Criteo/HOFM_criteo/min10/",
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
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-08 21:12:35,797 P47106 INFO Set up feature encoder...
2020-07-08 21:12:35,797 P47106 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-08 21:12:35,798 P47106 INFO Loading data...
2020-07-08 21:12:35,806 P47106 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-08 21:12:41,932 P47106 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-08 21:12:44,420 P47106 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-08 21:12:44,561 P47106 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-08 21:12:44,561 P47106 INFO Loading train data done.
2020-07-08 21:12:48,072 P47106 INFO **** Start training: 12225 batches/epoch ****
2020-07-08 22:18:08,502 P47106 INFO [Metrics] logloss: 0.451751 - AUC: 0.798820
2020-07-08 22:18:08,506 P47106 INFO Save best model: monitor(max): 0.347069
2020-07-08 22:18:08,606 P47106 INFO --- 12225/12225 batches finished ---
2020-07-08 22:18:08,659 P47106 INFO Train loss: 0.469915
2020-07-08 22:18:08,659 P47106 INFO ************ Epoch=1 end ************
2020-07-08 23:23:28,888 P47106 INFO [Metrics] logloss: 0.451395 - AUC: 0.799322
2020-07-08 23:23:28,889 P47106 INFO Save best model: monitor(max): 0.347927
2020-07-08 23:23:29,078 P47106 INFO --- 12225/12225 batches finished ---
2020-07-08 23:23:29,137 P47106 INFO Train loss: 0.467524
2020-07-08 23:23:29,137 P47106 INFO ************ Epoch=2 end ************
2020-07-09 00:28:40,523 P47106 INFO [Metrics] logloss: 0.450973 - AUC: 0.799693
2020-07-09 00:28:40,524 P47106 INFO Save best model: monitor(max): 0.348720
2020-07-09 00:28:40,694 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 00:28:40,749 P47106 INFO Train loss: 0.467421
2020-07-09 00:28:40,750 P47106 INFO ************ Epoch=3 end ************
2020-07-09 01:33:56,179 P47106 INFO [Metrics] logloss: 0.450687 - AUC: 0.799982
2020-07-09 01:33:56,180 P47106 INFO Save best model: monitor(max): 0.349295
2020-07-09 01:33:56,377 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 01:33:56,432 P47106 INFO Train loss: 0.467370
2020-07-09 01:33:56,432 P47106 INFO ************ Epoch=4 end ************
2020-07-09 02:39:05,873 P47106 INFO [Metrics] logloss: 0.450781 - AUC: 0.799841
2020-07-09 02:39:05,874 P47106 INFO Monitor(max) STOP: 0.349060 !
2020-07-09 02:39:05,874 P47106 INFO Reduce learning rate on plateau: 0.000100
2020-07-09 02:39:05,874 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 02:39:05,930 P47106 INFO Train loss: 0.467352
2020-07-09 02:39:05,930 P47106 INFO ************ Epoch=5 end ************
2020-07-09 03:44:34,876 P47106 INFO [Metrics] logloss: 0.444739 - AUC: 0.806555
2020-07-09 03:44:34,877 P47106 INFO Save best model: monitor(max): 0.361816
2020-07-09 03:44:35,075 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 03:44:35,130 P47106 INFO Train loss: 0.452823
2020-07-09 03:44:35,130 P47106 INFO ************ Epoch=6 end ************
2020-07-09 04:49:49,685 P47106 INFO [Metrics] logloss: 0.443860 - AUC: 0.807570
2020-07-09 04:49:49,685 P47106 INFO Save best model: monitor(max): 0.363710
2020-07-09 04:49:49,888 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 04:49:49,943 P47106 INFO Train loss: 0.448632
2020-07-09 04:49:49,943 P47106 INFO ************ Epoch=7 end ************
2020-07-09 05:55:06,763 P47106 INFO [Metrics] logloss: 0.443333 - AUC: 0.808128
2020-07-09 05:55:06,763 P47106 INFO Save best model: monitor(max): 0.364795
2020-07-09 05:55:06,947 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 05:55:07,003 P47106 INFO Train loss: 0.447429
2020-07-09 05:55:07,003 P47106 INFO ************ Epoch=8 end ************
2020-07-09 07:00:24,155 P47106 INFO [Metrics] logloss: 0.443033 - AUC: 0.808484
2020-07-09 07:00:24,155 P47106 INFO Save best model: monitor(max): 0.365451
2020-07-09 07:00:24,350 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 07:00:24,406 P47106 INFO Train loss: 0.446717
2020-07-09 07:00:24,406 P47106 INFO ************ Epoch=9 end ************
2020-07-09 08:05:42,684 P47106 INFO [Metrics] logloss: 0.442993 - AUC: 0.808565
2020-07-09 08:05:42,685 P47106 INFO Save best model: monitor(max): 0.365572
2020-07-09 08:05:42,887 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 08:05:42,944 P47106 INFO Train loss: 0.446231
2020-07-09 08:05:42,944 P47106 INFO ************ Epoch=10 end ************
2020-07-09 09:10:48,686 P47106 INFO [Metrics] logloss: 0.442801 - AUC: 0.808742
2020-07-09 09:10:48,687 P47106 INFO Save best model: monitor(max): 0.365940
2020-07-09 09:10:48,867 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 09:10:48,923 P47106 INFO Train loss: 0.445883
2020-07-09 09:10:48,923 P47106 INFO ************ Epoch=11 end ************
2020-07-09 10:16:03,301 P47106 INFO [Metrics] logloss: 0.442759 - AUC: 0.808799
2020-07-09 10:16:03,302 P47106 INFO Save best model: monitor(max): 0.366040
2020-07-09 10:16:03,513 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 10:16:03,570 P47106 INFO Train loss: 0.445619
2020-07-09 10:16:03,570 P47106 INFO ************ Epoch=12 end ************
2020-07-09 11:21:51,700 P47106 INFO [Metrics] logloss: 0.442542 - AUC: 0.809043
2020-07-09 11:21:51,701 P47106 INFO Save best model: monitor(max): 0.366502
2020-07-09 11:21:51,908 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 11:21:51,959 P47106 INFO Train loss: 0.445407
2020-07-09 11:21:51,959 P47106 INFO ************ Epoch=13 end ************
2020-07-09 12:27:37,476 P47106 INFO [Metrics] logloss: 0.442610 - AUC: 0.808984
2020-07-09 12:27:37,477 P47106 INFO Monitor(max) STOP: 0.366374 !
2020-07-09 12:27:37,477 P47106 INFO Reduce learning rate on plateau: 0.000010
2020-07-09 12:27:37,477 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 12:27:37,530 P47106 INFO Train loss: 0.445227
2020-07-09 12:27:37,530 P47106 INFO ************ Epoch=14 end ************
2020-07-09 13:33:14,104 P47106 INFO [Metrics] logloss: 0.441779 - AUC: 0.809891
2020-07-09 13:33:14,104 P47106 INFO Save best model: monitor(max): 0.368112
2020-07-09 13:33:14,307 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 13:33:14,361 P47106 INFO Train loss: 0.441124
2020-07-09 13:33:14,361 P47106 INFO ************ Epoch=15 end ************
2020-07-09 14:39:00,755 P47106 INFO [Metrics] logloss: 0.441627 - AUC: 0.810060
2020-07-09 14:39:00,756 P47106 INFO Save best model: monitor(max): 0.368432
2020-07-09 14:39:00,959 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 14:39:01,013 P47106 INFO Train loss: 0.440448
2020-07-09 14:39:01,013 P47106 INFO ************ Epoch=16 end ************
2020-07-09 15:44:57,361 P47106 INFO [Metrics] logloss: 0.441564 - AUC: 0.810130
2020-07-09 15:44:57,362 P47106 INFO Save best model: monitor(max): 0.368566
2020-07-09 15:44:57,563 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 15:44:57,618 P47106 INFO Train loss: 0.440146
2020-07-09 15:44:57,618 P47106 INFO ************ Epoch=17 end ************
2020-07-09 16:50:13,651 P47106 INFO [Metrics] logloss: 0.441530 - AUC: 0.810175
2020-07-09 16:50:13,652 P47106 INFO Save best model: monitor(max): 0.368645
2020-07-09 16:50:13,857 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 16:50:13,912 P47106 INFO Train loss: 0.439935
2020-07-09 16:50:13,912 P47106 INFO ************ Epoch=18 end ************
2020-07-09 17:55:26,247 P47106 INFO [Metrics] logloss: 0.441510 - AUC: 0.810199
2020-07-09 17:55:26,248 P47106 INFO Save best model: monitor(max): 0.368689
2020-07-09 17:55:26,442 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 17:55:26,500 P47106 INFO Train loss: 0.439768
2020-07-09 17:55:26,500 P47106 INFO ************ Epoch=19 end ************
2020-07-09 19:00:30,460 P47106 INFO [Metrics] logloss: 0.441498 - AUC: 0.810216
2020-07-09 19:00:30,461 P47106 INFO Save best model: monitor(max): 0.368718
2020-07-09 19:00:30,657 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 19:00:30,713 P47106 INFO Train loss: 0.439626
2020-07-09 19:00:30,713 P47106 INFO ************ Epoch=20 end ************
2020-07-09 20:05:37,002 P47106 INFO [Metrics] logloss: 0.441483 - AUC: 0.810234
2020-07-09 20:05:37,003 P47106 INFO Save best model: monitor(max): 0.368751
2020-07-09 20:05:37,206 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 20:05:37,265 P47106 INFO Train loss: 0.439503
2020-07-09 20:05:37,265 P47106 INFO ************ Epoch=21 end ************
2020-07-09 21:10:42,215 P47106 INFO [Metrics] logloss: 0.441472 - AUC: 0.810245
2020-07-09 21:10:42,216 P47106 INFO Save best model: monitor(max): 0.368773
2020-07-09 21:10:42,401 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 21:10:42,458 P47106 INFO Train loss: 0.439382
2020-07-09 21:10:42,458 P47106 INFO ************ Epoch=22 end ************
2020-07-09 22:15:55,662 P47106 INFO [Metrics] logloss: 0.441472 - AUC: 0.810255
2020-07-09 22:15:55,663 P47106 INFO Save best model: monitor(max): 0.368782
2020-07-09 22:15:55,874 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 22:15:55,930 P47106 INFO Train loss: 0.439278
2020-07-09 22:15:55,930 P47106 INFO ************ Epoch=23 end ************
2020-07-09 23:20:57,637 P47106 INFO [Metrics] logloss: 0.441460 - AUC: 0.810266
2020-07-09 23:20:57,638 P47106 INFO Save best model: monitor(max): 0.368806
2020-07-09 23:20:57,829 P47106 INFO --- 12225/12225 batches finished ---
2020-07-09 23:20:57,885 P47106 INFO Train loss: 0.439177
2020-07-09 23:20:57,885 P47106 INFO ************ Epoch=24 end ************
2020-07-10 00:26:09,337 P47106 INFO [Metrics] logloss: 0.441444 - AUC: 0.810284
2020-07-10 00:26:09,338 P47106 INFO Save best model: monitor(max): 0.368840
2020-07-10 00:26:09,543 P47106 INFO --- 12225/12225 batches finished ---
2020-07-10 00:26:09,599 P47106 INFO Train loss: 0.439082
2020-07-10 00:26:09,599 P47106 INFO ************ Epoch=25 end ************
2020-07-10 01:31:19,372 P47106 INFO [Metrics] logloss: 0.441464 - AUC: 0.810262
2020-07-10 01:31:19,373 P47106 INFO Monitor(max) STOP: 0.368797 !
2020-07-10 01:31:19,373 P47106 INFO Reduce learning rate on plateau: 0.000001
2020-07-10 01:31:19,373 P47106 INFO --- 12225/12225 batches finished ---
2020-07-10 01:31:19,429 P47106 INFO Train loss: 0.438993
2020-07-10 01:31:19,430 P47106 INFO ************ Epoch=26 end ************
2020-07-10 02:36:33,786 P47106 INFO [Metrics] logloss: 0.441425 - AUC: 0.810311
2020-07-10 02:36:33,786 P47106 INFO Save best model: monitor(max): 0.368886
2020-07-10 02:36:33,991 P47106 INFO --- 12225/12225 batches finished ---
2020-07-10 02:36:34,048 P47106 INFO Train loss: 0.438130
2020-07-10 02:36:34,048 P47106 INFO ************ Epoch=27 end ************
2020-07-10 03:41:49,375 P47106 INFO [Metrics] logloss: 0.441417 - AUC: 0.810320
2020-07-10 03:41:49,376 P47106 INFO Save best model: monitor(max): 0.368903
2020-07-10 03:41:49,562 P47106 INFO --- 12225/12225 batches finished ---
2020-07-10 03:41:49,618 P47106 INFO Train loss: 0.438100
2020-07-10 03:41:49,618 P47106 INFO ************ Epoch=28 end ************
2020-07-10 04:47:04,240 P47106 INFO [Metrics] logloss: 0.441416 - AUC: 0.810327
2020-07-10 04:47:04,241 P47106 INFO Save best model: monitor(max): 0.368911
2020-07-10 04:47:04,454 P47106 INFO --- 12225/12225 batches finished ---
2020-07-10 04:47:04,511 P47106 INFO Train loss: 0.438083
2020-07-10 04:47:04,511 P47106 INFO ************ Epoch=29 end ************
2020-07-10 05:52:14,430 P47106 INFO [Metrics] logloss: 0.441415 - AUC: 0.810324
2020-07-10 05:52:14,431 P47106 INFO Monitor(max) STOP: 0.368909 !
2020-07-10 05:52:14,431 P47106 INFO Reduce learning rate on plateau: 0.000001
2020-07-10 05:52:14,431 P47106 INFO --- 12225/12225 batches finished ---
2020-07-10 05:52:14,487 P47106 INFO Train loss: 0.438072
2020-07-10 05:52:14,487 P47106 INFO ************ Epoch=30 end ************
2020-07-10 06:57:28,298 P47106 INFO [Metrics] logloss: 0.441415 - AUC: 0.810327
2020-07-10 06:57:28,299 P47106 INFO Monitor(max) STOP: 0.368912 !
2020-07-10 06:57:28,299 P47106 INFO Reduce learning rate on plateau: 0.000001
2020-07-10 06:57:28,299 P47106 INFO Early stopping at epoch=31
2020-07-10 06:57:28,299 P47106 INFO --- 12225/12225 batches finished ---
2020-07-10 06:57:28,355 P47106 INFO Train loss: 0.438057
2020-07-10 06:57:28,355 P47106 INFO Training finished.
2020-07-10 06:57:28,355 P47106 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Criteo/HOFM_criteo/min10/criteo_x4_5c863b0f/HOFM_criteo_x4_5c863b0f_001_cef08737_model.ckpt
2020-07-10 06:57:28,503 P47106 INFO ****** Train/validation evaluation ******
2020-07-10 07:00:17,265 P47106 INFO [Metrics] logloss: 0.441416 - AUC: 0.810327
2020-07-10 07:00:17,350 P47106 INFO ******** Test evaluation ********
2020-07-10 07:00:17,350 P47106 INFO Loading data...
2020-07-10 07:00:17,350 P47106 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-10 07:00:18,116 P47106 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-10 07:00:18,116 P47106 INFO Loading test data done.
2020-07-10 07:03:06,848 P47106 INFO [Metrics] logloss: 0.441079 - AUC: 0.810747
```
