## DeepFM_Criteo_x4_002

A notebook to benchmark DeepFM on Criteo_x4_002 dataset.

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

3. Download the hyper-parameter configuration file: [DeepFM_criteo_x4_tuner_config_03.yaml](./002/DeepFM_criteo_x4_tuner_config_03.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/DeepFM_criteo_x4_tuner_config_03.yaml --tag 033 --gpu 0
  ```
### Results
```python
[Metrics] logloss: 0.437793 - AUC: 0.814069
```


### Logs
```python
2020-03-09 15:11:42,678 P20516 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_41e78b20",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-5)",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[500, 500, 500, 500]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepFM",
    "model_id": "DeepFM_criteo_x4_001_5cc29d4e",
    "model_root": "./Criteo/MAEN_criteo/",
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
    "test_data": "../data/Criteo/criteo_x4_001_41e78b20/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_41e78b20/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_41e78b20/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-03-09 15:11:42,679 P20516 INFO Set up feature encoder...
2020-03-09 15:11:42,679 P20516 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_41e78b20/feature_map.json
2020-03-09 15:11:42,679 P20516 INFO Loading data...
2020-03-09 15:11:42,682 P20516 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_41e78b20/train.h5
2020-03-09 15:11:47,892 P20516 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_41e78b20/valid.h5
2020-03-09 15:11:49,649 P20516 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-03-09 15:11:49,781 P20516 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-09 15:11:49,781 P20516 INFO Loading train data done.
2020-03-09 15:11:58,098 P20516 INFO **** Start training: 3668 batches/epoch ****
2020-03-09 15:29:24,220 P20516 INFO [Metrics] logloss: 0.447009 - AUC: 0.804310
2020-03-09 15:29:24,225 P20516 INFO Save best model: monitor(max): 0.357301
2020-03-09 15:29:25,070 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 15:29:25,116 P20516 INFO Train loss: 0.469432
2020-03-09 15:29:25,116 P20516 INFO ************ Epoch=1 end ************
2020-03-09 15:46:45,967 P20516 INFO [Metrics] logloss: 0.444596 - AUC: 0.806757
2020-03-09 15:46:45,968 P20516 INFO Save best model: monitor(max): 0.362162
2020-03-09 15:46:47,633 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 15:46:47,684 P20516 INFO Train loss: 0.466698
2020-03-09 15:46:47,684 P20516 INFO ************ Epoch=2 end ************
2020-03-09 16:04:07,226 P20516 INFO [Metrics] logloss: 0.443954 - AUC: 0.807551
2020-03-09 16:04:07,227 P20516 INFO Save best model: monitor(max): 0.363596
2020-03-09 16:04:08,917 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 16:04:08,968 P20516 INFO Train loss: 0.465737
2020-03-09 16:04:08,968 P20516 INFO ************ Epoch=3 end ************
2020-03-09 16:21:26,997 P20516 INFO [Metrics] logloss: 0.443479 - AUC: 0.807993
2020-03-09 16:21:26,998 P20516 INFO Save best model: monitor(max): 0.364514
2020-03-09 16:21:28,652 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 16:21:28,708 P20516 INFO Train loss: 0.465235
2020-03-09 16:21:28,708 P20516 INFO ************ Epoch=4 end ************
2020-03-09 16:38:48,218 P20516 INFO [Metrics] logloss: 0.443123 - AUC: 0.808426
2020-03-09 16:38:48,220 P20516 INFO Save best model: monitor(max): 0.365303
2020-03-09 16:38:49,191 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 16:38:49,262 P20516 INFO Train loss: 0.464952
2020-03-09 16:38:49,262 P20516 INFO ************ Epoch=5 end ************
2020-03-09 16:56:11,071 P20516 INFO [Metrics] logloss: 0.443433 - AUC: 0.808265
2020-03-09 16:56:11,073 P20516 INFO Monitor(max) STOP: 0.364832 !
2020-03-09 16:56:11,073 P20516 INFO Reduce learning rate on plateau: 0.000100
2020-03-09 16:56:11,073 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 16:56:11,125 P20516 INFO Train loss: 0.464794
2020-03-09 16:56:11,125 P20516 INFO ************ Epoch=6 end ************
2020-03-09 17:13:33,746 P20516 INFO [Metrics] logloss: 0.438750 - AUC: 0.812991
2020-03-09 17:13:33,748 P20516 INFO Save best model: monitor(max): 0.374241
2020-03-09 17:13:35,523 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 17:13:35,571 P20516 INFO Train loss: 0.444358
2020-03-09 17:13:35,571 P20516 INFO ************ Epoch=7 end ************
2020-03-09 17:30:56,211 P20516 INFO [Metrics] logloss: 0.438331 - AUC: 0.813491
2020-03-09 17:30:56,213 P20516 INFO Save best model: monitor(max): 0.375160
2020-03-09 17:30:57,903 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 17:30:57,952 P20516 INFO Train loss: 0.438518
2020-03-09 17:30:57,952 P20516 INFO ************ Epoch=8 end ************
2020-03-09 17:48:18,343 P20516 INFO [Metrics] logloss: 0.438184 - AUC: 0.813612
2020-03-09 17:48:18,344 P20516 INFO Save best model: monitor(max): 0.375428
2020-03-09 17:48:20,023 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 17:48:20,076 P20516 INFO Train loss: 0.436459
2020-03-09 17:48:20,076 P20516 INFO ************ Epoch=9 end ************
2020-03-09 18:05:38,271 P20516 INFO [Metrics] logloss: 0.438377 - AUC: 0.813431
2020-03-09 18:05:38,273 P20516 INFO Monitor(max) STOP: 0.375053 !
2020-03-09 18:05:38,273 P20516 INFO Reduce learning rate on plateau: 0.000010
2020-03-09 18:05:38,273 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 18:05:38,342 P20516 INFO Train loss: 0.434892
2020-03-09 18:05:38,342 P20516 INFO ************ Epoch=10 end ************
2020-03-09 18:22:56,281 P20516 INFO [Metrics] logloss: 0.439260 - AUC: 0.812784
2020-03-09 18:22:56,283 P20516 INFO Monitor(max) STOP: 0.373524 !
2020-03-09 18:22:56,283 P20516 INFO Reduce learning rate on plateau: 0.000001
2020-03-09 18:22:56,283 P20516 INFO Early stopping at epoch=11
2020-03-09 18:22:56,283 P20516 INFO --- 3668/3668 batches finished ---
2020-03-09 18:22:56,338 P20516 INFO Train loss: 0.428222
2020-03-09 18:22:56,338 P20516 INFO Training finished.
2020-03-09 18:22:56,338 P20516 INFO Load best model: /home/xxx/zhujieming/OpenCTR1030/benchmarks/Criteo/MAEN_criteo/criteo_x4_001_41e78b20/DeepFM_criteo_x4_001_5cc29d4e_model.ckpt
2020-03-09 18:22:57,315 P20516 INFO ****** Train/validation evaluation ******
2020-03-09 18:27:53,036 P20516 INFO [Metrics] logloss: 0.421343 - AUC: 0.831921
2020-03-09 18:28:28,626 P20516 INFO [Metrics] logloss: 0.438184 - AUC: 0.813612
2020-03-09 18:28:28,709 P20516 INFO ******** Test evaluation ********
2020-03-09 18:28:28,710 P20516 INFO Loading data...
2020-03-09 18:28:28,710 P20516 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_41e78b20/test.h5
2020-03-09 18:28:29,495 P20516 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-09 18:28:29,495 P20516 INFO Loading test data done.
2020-03-09 18:29:06,051 P20516 INFO [Metrics] logloss: 0.437793 - AUC: 0.814069


```