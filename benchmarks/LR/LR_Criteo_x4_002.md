## LR_Criteo_x4_002

A notebook to benchmark LR on Criteo_x4_002 dataset.

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

3. Download the hyper-parameter configuration file: [LR_criteo_x4_tuner_config_01.yaml](./002/LR_criteo_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/LR_criteo_x4_tuner_config_01.yaml --tag 004 --gpu 0
  ```
### Results
```python
[Metrics] logloss: 0.456581 - AUC: 0.793551
```


### Logs
```python
2020-02-23 19:52:05,796 P29361 INFO {
    "batch_size": "10000",
    "dataset_id": "criteo_x4_001_be98441d",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "LR",
    "model_id": "LR_criteo_x4_003_72dcd6fe",
    "model_root": "./Criteo/AFN_criteo/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "l2(1.e-7)",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "1",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-02-23 19:52:05,797 P29361 INFO Set up feature encoder...
2020-02-23 19:52:05,797 P29361 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-02-23 19:52:05,797 P29361 INFO Loading data...
2020-02-23 19:52:05,800 P29361 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-02-23 19:52:10,846 P29361 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-02-23 19:52:12,831 P29361 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-02-23 19:52:12,965 P29361 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-23 19:52:12,965 P29361 INFO Loading train data done.
2020-02-23 19:52:16,192 P29361 INFO **** Start training: 3668 batches/epoch ****
2020-02-23 19:59:26,246 P29361 INFO [Metrics] logloss: 0.458307 - AUC: 0.791528
2020-02-23 19:59:26,325 P29361 INFO Save best model: monitor(max): 0.333221
2020-02-23 19:59:26,349 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 19:59:26,432 P29361 INFO Train loss: 0.466786
2020-02-23 19:59:26,432 P29361 INFO ************ Epoch=1 end ************
2020-02-23 20:06:34,605 P29361 INFO [Metrics] logloss: 0.457311 - AUC: 0.792653
2020-02-23 20:06:34,686 P29361 INFO Save best model: monitor(max): 0.335342
2020-02-23 20:06:34,715 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:06:34,793 P29361 INFO Train loss: 0.454869
2020-02-23 20:06:34,793 P29361 INFO ************ Epoch=2 end ************
2020-02-23 20:13:42,980 P29361 INFO [Metrics] logloss: 0.457204 - AUC: 0.792756
2020-02-23 20:13:43,052 P29361 INFO Save best model: monitor(max): 0.335552
2020-02-23 20:13:43,080 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:13:43,156 P29361 INFO Train loss: 0.452924
2020-02-23 20:13:43,156 P29361 INFO ************ Epoch=3 end ************
2020-02-23 20:20:49,663 P29361 INFO [Metrics] logloss: 0.457256 - AUC: 0.792729
2020-02-23 20:20:49,733 P29361 INFO Monitor(max) STOP: 0.335473 !
2020-02-23 20:20:49,733 P29361 INFO Reduce learning rate on plateau: 0.000100
2020-02-23 20:20:49,733 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:20:49,841 P29361 INFO Train loss: 0.452126
2020-02-23 20:20:49,841 P29361 INFO ************ Epoch=4 end ************
2020-02-23 20:27:56,196 P29361 INFO [Metrics] logloss: 0.456965 - AUC: 0.793028
2020-02-23 20:27:56,266 P29361 INFO Save best model: monitor(max): 0.336063
2020-02-23 20:27:56,293 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:27:56,393 P29361 INFO Train loss: 0.448910
2020-02-23 20:27:56,393 P29361 INFO ************ Epoch=5 end ************
2020-02-23 20:35:09,455 P29361 INFO [Metrics] logloss: 0.456922 - AUC: 0.793088
2020-02-23 20:35:09,525 P29361 INFO Save best model: monitor(max): 0.336166
2020-02-23 20:35:09,553 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:35:09,656 P29361 INFO Train loss: 0.448790
2020-02-23 20:35:09,656 P29361 INFO ************ Epoch=6 end ************
2020-02-23 20:42:15,968 P29361 INFO [Metrics] logloss: 0.456906 - AUC: 0.793097
2020-02-23 20:42:16,043 P29361 INFO Save best model: monitor(max): 0.336191
2020-02-23 20:42:16,071 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:42:16,174 P29361 INFO Train loss: 0.448731
2020-02-23 20:42:16,174 P29361 INFO ************ Epoch=7 end ************
2020-02-23 20:49:21,740 P29361 INFO [Metrics] logloss: 0.456896 - AUC: 0.793124
2020-02-23 20:49:21,820 P29361 INFO Save best model: monitor(max): 0.336228
2020-02-23 20:49:21,847 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:49:21,922 P29361 INFO Train loss: 0.448703
2020-02-23 20:49:21,922 P29361 INFO ************ Epoch=8 end ************
2020-02-23 20:56:29,552 P29361 INFO [Metrics] logloss: 0.456896 - AUC: 0.793124
2020-02-23 20:56:29,643 P29361 INFO Monitor(max) STOP: 0.336228 !
2020-02-23 20:56:29,643 P29361 INFO Reduce learning rate on plateau: 0.000010
2020-02-23 20:56:29,643 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 20:56:29,717 P29361 INFO Train loss: 0.448669
2020-02-23 20:56:29,717 P29361 INFO ************ Epoch=9 end ************
2020-02-23 21:03:39,131 P29361 INFO [Metrics] logloss: 0.456891 - AUC: 0.793126
2020-02-23 21:03:39,207 P29361 INFO Save best model: monitor(max): 0.336235
2020-02-23 21:03:39,233 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 21:03:39,308 P29361 INFO Train loss: 0.448293
2020-02-23 21:03:39,308 P29361 INFO ************ Epoch=10 end ************
2020-02-23 21:10:49,200 P29361 INFO [Metrics] logloss: 0.456892 - AUC: 0.793127
2020-02-23 21:10:49,308 P29361 INFO Monitor(max) STOP: 0.336235 !
2020-02-23 21:10:49,309 P29361 INFO Reduce learning rate on plateau: 0.000001
2020-02-23 21:10:49,309 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 21:10:49,381 P29361 INFO Train loss: 0.448290
2020-02-23 21:10:49,381 P29361 INFO ************ Epoch=11 end ************
2020-02-23 21:17:55,927 P29361 INFO [Metrics] logloss: 0.456889 - AUC: 0.793127
2020-02-23 21:17:56,034 P29361 INFO Save best model: monitor(max): 0.336238
2020-02-23 21:17:56,071 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 21:17:56,161 P29361 INFO Train loss: 0.448251
2020-02-23 21:17:56,161 P29361 INFO ************ Epoch=12 end ************
2020-02-23 21:25:06,455 P29361 INFO [Metrics] logloss: 0.456889 - AUC: 0.793127
2020-02-23 21:25:06,561 P29361 INFO Monitor(max) STOP: 0.336238 !
2020-02-23 21:25:06,561 P29361 INFO Reduce learning rate on plateau: 0.000001
2020-02-23 21:25:06,562 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 21:25:06,644 P29361 INFO Train loss: 0.448244
2020-02-23 21:25:06,644 P29361 INFO ************ Epoch=13 end ************
2020-02-23 21:32:13,282 P29361 INFO [Metrics] logloss: 0.456889 - AUC: 0.793127
2020-02-23 21:32:13,359 P29361 INFO Monitor(max) STOP: 0.336238 !
2020-02-23 21:32:13,360 P29361 INFO Reduce learning rate on plateau: 0.000001
2020-02-23 21:32:13,360 P29361 INFO Early stopping at epoch=14
2020-02-23 21:32:13,360 P29361 INFO --- 3668/3668 batches finished ---
2020-02-23 21:32:13,433 P29361 INFO Train loss: 0.448249
2020-02-23 21:32:13,434 P29361 INFO Training finished.
2020-02-23 21:32:13,434 P29361 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Criteo/AFN_criteo/criteo_x4_001_be98441d/LR_criteo_x4_003_72dcd6fe_criteo_x4_001_be98441d_model.ckpt
2020-02-23 21:32:13,468 P29361 INFO ****** Train/validation evaluation ******
2020-02-23 21:38:10,377 P29361 INFO [Metrics] logloss: 0.445684 - AUC: 0.806137
2020-02-23 21:38:55,084 P29361 INFO [Metrics] logloss: 0.456889 - AUC: 0.793127
2020-02-23 21:38:55,253 P29361 INFO ******** Test evaluation ********
2020-02-23 21:38:55,254 P29361 INFO Loading data...
2020-02-23 21:38:55,254 P29361 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-02-23 21:38:56,368 P29361 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-02-23 21:38:56,369 P29361 INFO Loading test data done.
2020-02-23 21:39:39,183 P29361 INFO [Metrics] logloss: 0.456581 - AUC: 0.793551

```