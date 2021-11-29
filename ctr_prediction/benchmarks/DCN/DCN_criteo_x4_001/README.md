## DCN_Criteo_x4_001

A notebook to benchmark DCN on Criteo_x4_001 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.
### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [DCN_criteo_x4_tuner_config_02.yaml](./DCN_criteo_x4_tuner_config_02.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/DCN_criteo_x4_tuner_config_02.yaml --tag 012 --gpu 0
  ```


### Results
```python
[Metrics][Metrics] logloss: 0.437612 - AUC: 0.814437
```


### Logs
```python
2020-06-21 12:02:48,255 P3142 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "4",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_criteo_x4_5c863b0f_012_56382345",
    "model_root": "./Criteo/DCN_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-21 12:02:48,258 P3142 INFO Set up feature encoder...
2020-06-21 12:02:48,258 P3142 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-21 12:02:48,258 P3142 INFO Loading data...
2020-06-21 12:02:48,273 P3142 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-21 12:02:54,658 P3142 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-21 12:02:56,526 P3142 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-21 12:02:56,650 P3142 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-21 12:02:56,650 P3142 INFO Loading train data done.
2020-06-21 12:03:02,356 P3142 INFO Start training: 3668 batches/epoch
2020-06-21 12:03:02,357 P3142 INFO ************ Epoch=1 start ************
2020-06-21 12:09:48,477 P3142 INFO [Metrics] logloss: 0.445750 - AUC: 0.805451
2020-06-21 12:09:48,482 P3142 INFO Save best model: monitor(max): 0.359701
2020-06-21 12:09:48,984 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:09:49,034 P3142 INFO Train loss: 0.459598
2020-06-21 12:09:49,034 P3142 INFO ************ Epoch=1 end ************
2020-06-21 12:16:38,140 P3142 INFO [Metrics] logloss: 0.443639 - AUC: 0.808019
2020-06-21 12:16:38,142 P3142 INFO Save best model: monitor(max): 0.364380
2020-06-21 12:16:38,235 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:16:38,298 P3142 INFO Train loss: 0.453779
2020-06-21 12:16:38,298 P3142 INFO ************ Epoch=2 end ************
2020-06-21 12:23:20,235 P3142 INFO [Metrics] logloss: 0.442435 - AUC: 0.809123
2020-06-21 12:23:20,236 P3142 INFO Save best model: monitor(max): 0.366688
2020-06-21 12:23:20,336 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:23:20,390 P3142 INFO Train loss: 0.452163
2020-06-21 12:23:20,390 P3142 INFO ************ Epoch=3 end ************
2020-06-21 12:30:01,697 P3142 INFO [Metrics] logloss: 0.441999 - AUC: 0.809616
2020-06-21 12:30:01,698 P3142 INFO Save best model: monitor(max): 0.367617
2020-06-21 12:30:01,799 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:30:01,853 P3142 INFO Train loss: 0.451251
2020-06-21 12:30:01,853 P3142 INFO ************ Epoch=4 end ************
2020-06-21 12:36:44,780 P3142 INFO [Metrics] logloss: 0.441421 - AUC: 0.810201
2020-06-21 12:36:44,781 P3142 INFO Save best model: monitor(max): 0.368781
2020-06-21 12:36:44,865 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:36:44,922 P3142 INFO Train loss: 0.450697
2020-06-21 12:36:44,922 P3142 INFO ************ Epoch=5 end ************
2020-06-21 12:43:27,657 P3142 INFO [Metrics] logloss: 0.441843 - AUC: 0.810091
2020-06-21 12:43:27,659 P3142 INFO Monitor(max) STOP: 0.368248 !
2020-06-21 12:43:27,659 P3142 INFO Reduce learning rate on plateau: 0.000100
2020-06-21 12:43:27,659 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:43:27,712 P3142 INFO Train loss: 0.450262
2020-06-21 12:43:27,713 P3142 INFO ************ Epoch=6 end ************
2020-06-21 12:50:07,595 P3142 INFO [Metrics] logloss: 0.438286 - AUC: 0.813516
2020-06-21 12:50:07,596 P3142 INFO Save best model: monitor(max): 0.375229
2020-06-21 12:50:07,717 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:50:07,770 P3142 INFO Train loss: 0.440124
2020-06-21 12:50:07,770 P3142 INFO ************ Epoch=7 end ************
2020-06-21 12:56:47,769 P3142 INFO [Metrics] logloss: 0.437958 - AUC: 0.813908
2020-06-21 12:56:47,770 P3142 INFO Save best model: monitor(max): 0.375950
2020-06-21 12:56:47,855 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 12:56:47,907 P3142 INFO Train loss: 0.435962
2020-06-21 12:56:47,907 P3142 INFO ************ Epoch=8 end ************
2020-06-21 13:03:29,378 P3142 INFO [Metrics] logloss: 0.437998 - AUC: 0.813987
2020-06-21 13:03:29,379 P3142 INFO Save best model: monitor(max): 0.375989
2020-06-21 13:03:29,489 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 13:03:29,541 P3142 INFO Train loss: 0.433825
2020-06-21 13:03:29,541 P3142 INFO ************ Epoch=9 end ************
2020-06-21 13:10:09,954 P3142 INFO [Metrics] logloss: 0.437999 - AUC: 0.813987
2020-06-21 13:10:09,955 P3142 INFO Monitor(max) STOP: 0.375988 !
2020-06-21 13:10:09,955 P3142 INFO Reduce learning rate on plateau: 0.000010
2020-06-21 13:10:09,955 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 13:10:10,007 P3142 INFO Train loss: 0.432134
2020-06-21 13:10:10,007 P3142 INFO ************ Epoch=10 end ************
2020-06-21 13:16:51,858 P3142 INFO [Metrics] logloss: 0.438618 - AUC: 0.813478
2020-06-21 13:16:51,859 P3142 INFO Monitor(max) STOP: 0.374859 !
2020-06-21 13:16:51,859 P3142 INFO Reduce learning rate on plateau: 0.000001
2020-06-21 13:16:51,859 P3142 INFO Early stopping at epoch=11
2020-06-21 13:16:51,859 P3142 INFO --- 3668/3668 batches finished ---
2020-06-21 13:16:51,911 P3142 INFO Train loss: 0.428005
2020-06-21 13:16:51,911 P3142 INFO Training finished.
2020-06-21 13:16:51,911 P3142 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/DCN_criteo/min10/criteo_x4_5c863b0f/DCN_criteo_x4_5c863b0f_012_56382345_model.ckpt
2020-06-21 13:16:52,037 P3142 INFO ****** Train/validation evaluation ******
2020-06-21 13:20:46,272 P3142 INFO [Metrics] logloss: 0.422779 - AUC: 0.830139
2020-06-21 13:21:12,196 P3142 INFO [Metrics] logloss: 0.437998 - AUC: 0.813987
2020-06-21 13:21:12,272 P3142 INFO ******** Test evaluation ********
2020-06-21 13:21:12,272 P3142 INFO Loading data...
2020-06-21 13:21:12,272 P3142 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-21 13:21:13,225 P3142 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-21 13:21:13,225 P3142 INFO Loading test data done.
2020-06-21 13:21:38,598 P3142 INFO [Metrics] logloss: 0.437612 - AUC: 0.814437


```