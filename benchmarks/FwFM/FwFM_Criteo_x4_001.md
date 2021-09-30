## FwFM_Criteo_x4_001

A notebook to benchmark FwFM on Criteo_x4_001 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default <OOV> token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless id field nor specially preprocess the timestamp field.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.


### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [FwFM_criteo_x4_tuner_config_02.yaml](./FwFM_criteo_x4_tuner_config_02.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/FwFM_criteo_x4_tuner_config_02.yaml --tag 002 --gpu 0
  ```


### Results
```python
[Metrics] logloss: 0.440797 - AUC: 0.811214
```


### Logs
```python
2021-09-08 11:36:29,491 P510 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_9ea3bdfc",
    "debug": "False",
    "embedding_dim": "16",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "linear_type": "LW",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FwFM",
    "model_id": "FwFM_criteo_x4_9ea3bdfc_002_883457ca",
    "model_root": "./Criteo/FwFM_criteo_x4_001/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_9ea3bdfc/test.h5",
    "train_data": "../data/Criteo/criteo_x4_9ea3bdfc/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_9ea3bdfc/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2021-09-08 11:36:29,492 P510 INFO Set up feature encoder...
2021-09-08 11:36:29,492 P510 INFO Load feature_map from json: ../data/Criteo/criteo_x4_9ea3bdfc/feature_map.json
2021-09-08 11:36:30,033 P510 INFO Total number of parameters: 15482778.
2021-09-08 11:36:30,034 P510 INFO Loading data...
2021-09-08 11:36:30,036 P510 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/train.h5
2021-09-08 11:36:34,918 P510 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/valid.h5
2021-09-08 11:36:36,521 P510 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2021-09-08 11:36:36,651 P510 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-08 11:36:36,651 P510 INFO Loading train data done.
2021-09-08 11:36:39,714 P510 INFO Start training: 3668 batches/epoch
2021-09-08 11:36:39,714 P510 INFO ************ Epoch=1 start ************
2021-09-08 11:45:19,012 P510 INFO [Metrics] logloss: 0.450626 - AUC: 0.800116
2021-09-08 11:45:19,014 P510 INFO Save best model: monitor(max): 0.349490
2021-09-08 11:45:19,234 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 11:45:19,270 P510 INFO Train loss: 0.463842
2021-09-08 11:45:19,270 P510 INFO ************ Epoch=1 end ************
2021-09-08 11:53:57,042 P510 INFO [Metrics] logloss: 0.446472 - AUC: 0.804799
2021-09-08 11:53:57,043 P510 INFO Save best model: monitor(max): 0.358327
2021-09-08 11:53:57,117 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 11:53:57,158 P510 INFO Train loss: 0.455840
2021-09-08 11:53:57,159 P510 INFO ************ Epoch=2 end ************
2021-09-08 12:02:40,512 P510 INFO [Metrics] logloss: 0.445139 - AUC: 0.806271
2021-09-08 12:02:40,513 P510 INFO Save best model: monitor(max): 0.361132
2021-09-08 12:02:40,587 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 12:02:40,623 P510 INFO Train loss: 0.453898
2021-09-08 12:02:40,623 P510 INFO ************ Epoch=3 end ************
2021-09-08 12:11:24,283 P510 INFO [Metrics] logloss: 0.444461 - AUC: 0.807067
2021-09-08 12:11:24,284 P510 INFO Save best model: monitor(max): 0.362607
2021-09-08 12:11:24,351 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 12:11:24,388 P510 INFO Train loss: 0.453056
2021-09-08 12:11:24,389 P510 INFO ************ Epoch=4 end ************
2021-09-08 12:20:08,121 P510 INFO [Metrics] logloss: 0.444274 - AUC: 0.807399
2021-09-08 12:20:08,123 P510 INFO Save best model: monitor(max): 0.363125
2021-09-08 12:20:08,202 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 12:20:08,241 P510 INFO Train loss: 0.452614
2021-09-08 12:20:08,242 P510 INFO ************ Epoch=5 end ************
2021-09-08 12:28:46,145 P510 INFO [Metrics] logloss: 0.444012 - AUC: 0.807562
2021-09-08 12:28:46,146 P510 INFO Save best model: monitor(max): 0.363549
2021-09-08 12:28:46,220 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 12:28:46,255 P510 INFO Train loss: 0.452340
2021-09-08 12:28:46,255 P510 INFO ************ Epoch=6 end ************
2021-09-08 12:37:26,797 P510 INFO [Metrics] logloss: 0.443883 - AUC: 0.807693
2021-09-08 12:37:26,798 P510 INFO Save best model: monitor(max): 0.363810
2021-09-08 12:37:26,876 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 12:37:26,919 P510 INFO Train loss: 0.452158
2021-09-08 12:37:26,919 P510 INFO ************ Epoch=7 end ************
2021-09-08 12:46:06,140 P510 INFO [Metrics] logloss: 0.443772 - AUC: 0.807870
2021-09-08 12:46:06,142 P510 INFO Save best model: monitor(max): 0.364098
2021-09-08 12:46:06,219 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 12:46:06,259 P510 INFO Train loss: 0.452008
2021-09-08 12:46:06,259 P510 INFO ************ Epoch=8 end ************
2021-09-08 12:54:43,418 P510 INFO [Metrics] logloss: 0.443756 - AUC: 0.807828
2021-09-08 12:54:43,419 P510 INFO Monitor(max) STOP: 0.364072 !
2021-09-08 12:54:43,419 P510 INFO Reduce learning rate on plateau: 0.000100
2021-09-08 12:54:43,419 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 12:54:43,454 P510 INFO Train loss: 0.451890
2021-09-08 12:54:43,455 P510 INFO ************ Epoch=9 end ************
2021-09-08 13:03:26,112 P510 INFO [Metrics] logloss: 0.441507 - AUC: 0.810300
2021-09-08 13:03:26,113 P510 INFO Save best model: monitor(max): 0.368793
2021-09-08 13:03:26,184 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 13:03:26,223 P510 INFO Train loss: 0.443317
2021-09-08 13:03:26,223 P510 INFO ************ Epoch=10 end ************
2021-09-08 13:12:07,885 P510 INFO [Metrics] logloss: 0.441180 - AUC: 0.810683
2021-09-08 13:12:07,887 P510 INFO Save best model: monitor(max): 0.369503
2021-09-08 13:12:07,975 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 13:12:08,019 P510 INFO Train loss: 0.440620
2021-09-08 13:12:08,019 P510 INFO ************ Epoch=11 end ************
2021-09-08 13:20:50,198 P510 INFO [Metrics] logloss: 0.441090 - AUC: 0.810798
2021-09-08 13:20:50,200 P510 INFO Save best model: monitor(max): 0.369708
2021-09-08 13:20:50,268 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 13:20:50,308 P510 INFO Train loss: 0.439427
2021-09-08 13:20:50,309 P510 INFO ************ Epoch=12 end ************
2021-09-08 13:29:31,298 P510 INFO [Metrics] logloss: 0.441097 - AUC: 0.810831
2021-09-08 13:29:31,299 P510 INFO Save best model: monitor(max): 0.369734
2021-09-08 13:29:31,384 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 13:29:31,431 P510 INFO Train loss: 0.438564
2021-09-08 13:29:31,431 P510 INFO ************ Epoch=13 end ************
2021-09-08 13:38:10,177 P510 INFO [Metrics] logloss: 0.441125 - AUC: 0.810815
2021-09-08 13:38:10,178 P510 INFO Monitor(max) STOP: 0.369691 !
2021-09-08 13:38:10,178 P510 INFO Reduce learning rate on plateau: 0.000010
2021-09-08 13:38:10,178 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 13:38:10,213 P510 INFO Train loss: 0.437825
2021-09-08 13:38:10,213 P510 INFO ************ Epoch=14 end ************
2021-09-08 13:46:47,618 P510 INFO [Metrics] logloss: 0.441087 - AUC: 0.810875
2021-09-08 13:46:47,619 P510 INFO Save best model: monitor(max): 0.369788
2021-09-08 13:46:47,700 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 13:46:47,740 P510 INFO Train loss: 0.435120
2021-09-08 13:46:47,740 P510 INFO ************ Epoch=15 end ************
2021-09-08 13:55:27,054 P510 INFO [Metrics] logloss: 0.441106 - AUC: 0.810862
2021-09-08 13:55:27,056 P510 INFO Monitor(max) STOP: 0.369756 !
2021-09-08 13:55:27,056 P510 INFO Reduce learning rate on plateau: 0.000001
2021-09-08 13:55:27,056 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 13:55:27,098 P510 INFO Train loss: 0.434958
2021-09-08 13:55:27,099 P510 INFO ************ Epoch=16 end ************
2021-09-08 14:04:09,088 P510 INFO [Metrics] logloss: 0.441108 - AUC: 0.810861
2021-09-08 14:04:09,090 P510 INFO Monitor(max) STOP: 0.369752 !
2021-09-08 14:04:09,090 P510 INFO Reduce learning rate on plateau: 0.000001
2021-09-08 14:04:09,090 P510 INFO Early stopping at epoch=17
2021-09-08 14:04:09,090 P510 INFO --- 3668/3668 batches finished ---
2021-09-08 14:04:09,127 P510 INFO Train loss: 0.434601
2021-09-08 14:04:09,127 P510 INFO Training finished.
2021-09-08 14:04:09,128 P510 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/FwFM_criteo_x4_001/criteo_x4_9ea3bdfc/FwFM_criteo_x4_9ea3bdfc_002_883457ca_model.ckpt
2021-09-08 14:04:09,213 P510 INFO ****** Train/validation evaluation ******
2021-09-08 14:04:34,600 P510 INFO [Metrics] logloss: 0.441087 - AUC: 0.810875
2021-09-08 14:04:34,684 P510 INFO ******** Test evaluation ********
2021-09-08 14:04:34,684 P510 INFO Loading data...
2021-09-08 14:04:34,684 P510 INFO Loading data from h5: ../data/Criteo/criteo_x4_9ea3bdfc/test.h5
2021-09-08 14:04:35,531 P510 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2021-09-08 14:04:35,531 P510 INFO Loading test data done.
2021-09-08 14:04:58,451 P510 INFO [Metrics] logloss: 0.440797 - AUC: 0.811214

```
