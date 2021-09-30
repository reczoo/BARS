## CIN_KKBox_x4_001

A notebook to benchmark CIN on KKBox_x4_001 dataset.

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
In this setting, For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10.

To make a fair comparison, we fix embedding_dim=128, which performs well.


### Code

1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/KKBox/KKBox_x4/split_kkbox_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [CIN_kkbox_x4_tuner_config_02-022.yaml](./CIN_kkbox_x4_tuner_config_02-022.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/CIN_kkbox_x4_tuner_config_02-022.yaml --tag 022 --gpu 0
  ```



### Results
```python
[Metrics] logloss: 0.490878 - AUC: 0.842620
```


### Logs
```python
2020-04-29 05:17:18,308 P22608 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "cin_layer_units": "[78, 78]",
    "dataset_id": "kkbox_x4_001_c5c9c6e3",
    "dnn_hidden_units": "[]",
    "embedding_dim": "128",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_kkbox_x4_022_d115d58a",
    "model_root": "./KKBox/CIN_kkbox/",
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
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/KKBox/",
    "test_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5",
    "train_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5",
    "valid_data": "../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-04-29 05:17:18,309 P22608 INFO Set up feature encoder...
2020-04-29 05:17:18,309 P22608 INFO Load feature_map from json: ../data/KKBox/kkbox_x4_001_c5c9c6e3/feature_map.json
2020-04-29 05:17:18,309 P22608 INFO Loading data...
2020-04-29 05:17:18,311 P22608 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/train.h5
2020-04-29 05:17:18,598 P22608 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/valid.h5
2020-04-29 05:17:18,790 P22608 INFO Train samples: total/5901932, pos/2971724, neg/2930208, ratio/50.35%
2020-04-29 05:17:18,810 P22608 INFO Validation samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-29 05:17:18,810 P22608 INFO Loading train data done.
2020-04-29 05:17:22,691 P22608 INFO **** Start training: 1181 batches/epoch ****
2020-04-29 05:26:56,265 P22608 INFO [Metrics] logloss: 0.570631 - AUC: 0.781371
2020-04-29 05:26:56,281 P22608 INFO Save best model: monitor(max): 0.210740
2020-04-29 05:26:56,324 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:26:56,362 P22608 INFO Train loss: 0.592051
2020-04-29 05:26:56,363 P22608 INFO ************ Epoch=1 end ************
2020-04-29 05:36:27,847 P22608 INFO [Metrics] logloss: 0.528563 - AUC: 0.812455
2020-04-29 05:36:27,862 P22608 INFO Save best model: monitor(max): 0.283893
2020-04-29 05:36:27,920 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:36:27,989 P22608 INFO Train loss: 0.558708
2020-04-29 05:36:27,990 P22608 INFO ************ Epoch=2 end ************
2020-04-29 05:45:59,970 P22608 INFO [Metrics] logloss: 0.500528 - AUC: 0.834321
2020-04-29 05:45:59,984 P22608 INFO Save best model: monitor(max): 0.333793
2020-04-29 05:46:00,051 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:46:00,122 P22608 INFO Train loss: 0.531992
2020-04-29 05:46:00,123 P22608 INFO ************ Epoch=3 end ************
2020-04-29 05:55:33,363 P22608 INFO [Metrics] logloss: 0.490362 - AUC: 0.842968
2020-04-29 05:55:33,381 P22608 INFO Save best model: monitor(max): 0.352606
2020-04-29 05:55:33,443 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 05:55:33,493 P22608 INFO Train loss: 0.507425
2020-04-29 05:55:33,493 P22608 INFO ************ Epoch=4 end ************
2020-04-29 06:05:06,034 P22608 INFO [Metrics] logloss: 0.500369 - AUC: 0.839405
2020-04-29 06:05:06,060 P22608 INFO Monitor(max) STOP: 0.339036 !
2020-04-29 06:05:06,060 P22608 INFO Reduce learning rate on plateau: 0.000100
2020-04-29 06:05:06,060 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 06:05:06,105 P22608 INFO Train loss: 0.479181
2020-04-29 06:05:06,106 P22608 INFO ************ Epoch=5 end ************
2020-04-29 06:14:39,634 P22608 INFO [Metrics] logloss: 0.638060 - AUC: 0.821444
2020-04-29 06:14:39,649 P22608 INFO Monitor(max) STOP: 0.183383 !
2020-04-29 06:14:39,649 P22608 INFO Reduce learning rate on plateau: 0.000010
2020-04-29 06:14:39,649 P22608 INFO Early stopping at epoch=6
2020-04-29 06:14:39,649 P22608 INFO --- 1181/1181 batches finished ---
2020-04-29 06:14:39,696 P22608 INFO Train loss: 0.326641
2020-04-29 06:14:39,696 P22608 INFO Training finished.
2020-04-29 06:14:39,697 P22608 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/KKBox/CIN_kkbox/kkbox_x4_001_c5c9c6e3/xDeepFM_kkbox_x4_022_d115d58a_kkbox_x4_001_c5c9c6e3_model.ckpt
2020-04-29 06:14:39,773 P22608 INFO ****** Train/validation evaluation ******
2020-04-29 06:15:58,568 P22608 INFO [Metrics] logloss: 0.393448 - AUC: 0.909427
2020-04-29 06:16:08,778 P22608 INFO [Metrics] logloss: 0.490362 - AUC: 0.842968
2020-04-29 06:16:08,888 P22608 INFO ******** Test evaluation ********
2020-04-29 06:16:08,888 P22608 INFO Loading data...
2020-04-29 06:16:08,888 P22608 INFO Loading data from h5: ../data/KKBox/kkbox_x4_001_c5c9c6e3/test.h5
2020-04-29 06:16:08,949 P22608 INFO Test samples: total/737743, pos/371466, neg/366277, ratio/50.35%
2020-04-29 06:16:08,949 P22608 INFO Loading test data done.
2020-04-29 06:16:18,845 P22608 INFO [Metrics] logloss: 0.490878 - AUC: 0.842620


```
