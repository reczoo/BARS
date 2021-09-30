## FGCNN_Criteo_x4_001

A notebook to benchmark FGCNN on Criteo_x4_001 dataset.

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

3. Download the hyper-parameter configuration file: [FGCNN_criteo_x4_tuner_config_01.yaml](./FGCNN_criteo_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/FGCNN_criteo_x4_tuner_config_01.yaml --tag 001 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.439800 - AUC: 0.812061
```


### Logs
```python
2020-06-29 12:02:11,762 P585 INFO {
    "batch_size": "5000",
    "channels": "[38, 40, 42, 44]",
    "conv_activation": "Tanh",
    "conv_batch_norm": "True",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_activations": "ReLU",
    "dnn_batch_norm": "False",
    "dnn_hidden_units": "[1000, 1000, 1000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "kernel_heights": "9",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FGCNN",
    "model_id": "FGCNN_criteo_x4_5c863b0f_001_aa4d4a89",
    "model_root": "./Criteo/FGCNN_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "pooling_sizes": "2",
    "recombined_channels": "3",
    "save_best_only": "True",
    "seed": "2019",
    "share_embedding": "False",
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
2020-06-29 12:02:11,764 P585 INFO Set up feature encoder...
2020-06-29 12:02:11,764 P585 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-29 12:02:11,764 P585 INFO Loading data...
2020-06-29 12:02:11,770 P585 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-29 12:02:18,861 P585 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-29 12:02:21,631 P585 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-29 12:02:21,845 P585 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-29 12:02:21,845 P585 INFO Loading train data done.
2020-06-29 12:04:01,673 P585 INFO Start training: 7335 batches/epoch
2020-06-29 12:04:01,673 P585 INFO ************ Epoch=1 start ************
2020-06-29 12:47:57,069 P585 INFO [Metrics] logloss: 0.448108 - AUC: 0.803004
2020-06-29 12:47:57,071 P585 INFO Save best model: monitor(max): 0.354895
2020-06-29 12:47:57,736 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 12:47:57,827 P585 INFO Train loss: 0.467665
2020-06-29 12:47:57,828 P585 INFO ************ Epoch=1 end ************
2020-06-29 13:28:03,687 P585 INFO [Metrics] logloss: 0.445829 - AUC: 0.805674
2020-06-29 13:28:03,689 P585 INFO Save best model: monitor(max): 0.359845
2020-06-29 13:28:04,155 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 13:28:04,215 P585 INFO Train loss: 0.459341
2020-06-29 13:28:04,215 P585 INFO ************ Epoch=2 end ************
2020-06-29 14:08:13,687 P585 INFO [Metrics] logloss: 0.446285 - AUC: 0.806552
2020-06-29 14:08:13,690 P585 INFO Save best model: monitor(max): 0.360267
2020-06-29 14:08:14,089 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 14:08:14,148 P585 INFO Train loss: 0.458019
2020-06-29 14:08:14,148 P585 INFO ************ Epoch=3 end ************
2020-06-29 14:48:15,703 P585 INFO [Metrics] logloss: 0.444209 - AUC: 0.807111
2020-06-29 14:48:15,704 P585 INFO Save best model: monitor(max): 0.362902
2020-06-29 14:48:16,133 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 14:48:16,201 P585 INFO Train loss: 0.457285
2020-06-29 14:48:16,202 P585 INFO ************ Epoch=4 end ************
2020-06-29 15:29:16,387 P585 INFO [Metrics] logloss: 0.443927 - AUC: 0.807468
2020-06-29 15:29:16,389 P585 INFO Save best model: monitor(max): 0.363542
2020-06-29 15:29:16,847 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 15:29:16,922 P585 INFO Train loss: 0.456642
2020-06-29 15:29:16,922 P585 INFO ************ Epoch=5 end ************
2020-06-29 16:09:23,828 P585 INFO [Metrics] logloss: 0.443501 - AUC: 0.807929
2020-06-29 16:09:23,830 P585 INFO Save best model: monitor(max): 0.364429
2020-06-29 16:09:24,282 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 16:09:24,351 P585 INFO Train loss: 0.456059
2020-06-29 16:09:24,351 P585 INFO ************ Epoch=6 end ************
2020-06-29 16:49:58,499 P585 INFO [Metrics] logloss: 0.443385 - AUC: 0.808165
2020-06-29 16:49:58,502 P585 INFO Save best model: monitor(max): 0.364780
2020-06-29 16:49:58,963 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 16:49:59,028 P585 INFO Train loss: 0.455580
2020-06-29 16:49:59,028 P585 INFO ************ Epoch=7 end ************
2020-06-29 17:30:17,719 P585 INFO [Metrics] logloss: 0.443072 - AUC: 0.808349
2020-06-29 17:30:17,722 P585 INFO Save best model: monitor(max): 0.365278
2020-06-29 17:30:18,126 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 17:30:18,188 P585 INFO Train loss: 0.455062
2020-06-29 17:30:18,188 P585 INFO ************ Epoch=8 end ************
2020-06-29 18:10:46,954 P585 INFO [Metrics] logloss: 0.443226 - AUC: 0.808389
2020-06-29 18:10:46,956 P585 INFO Monitor(max) STOP: 0.365163 !
2020-06-29 18:10:46,956 P585 INFO Reduce learning rate on plateau: 0.000100
2020-06-29 18:10:46,956 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 18:10:47,013 P585 INFO Train loss: 0.454655
2020-06-29 18:10:47,014 P585 INFO ************ Epoch=9 end ************
2020-06-29 18:51:28,063 P585 INFO [Metrics] logloss: 0.440299 - AUC: 0.811523
2020-06-29 18:51:28,064 P585 INFO Save best model: monitor(max): 0.371223
2020-06-29 18:51:28,497 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 18:51:28,579 P585 INFO Train loss: 0.441741
2020-06-29 18:51:28,579 P585 INFO ************ Epoch=10 end ************
2020-06-29 19:32:26,740 P585 INFO [Metrics] logloss: 0.440686 - AUC: 0.811480
2020-06-29 19:32:26,741 P585 INFO Monitor(max) STOP: 0.370794 !
2020-06-29 19:32:26,741 P585 INFO Reduce learning rate on plateau: 0.000010
2020-06-29 19:32:26,741 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 19:32:26,796 P585 INFO Train loss: 0.435093
2020-06-29 19:32:26,797 P585 INFO ************ Epoch=11 end ************
2020-06-29 20:13:32,121 P585 INFO [Metrics] logloss: 0.441912 - AUC: 0.810810
2020-06-29 20:13:32,123 P585 INFO Monitor(max) STOP: 0.368898 !
2020-06-29 20:13:32,124 P585 INFO Reduce learning rate on plateau: 0.000001
2020-06-29 20:13:32,128 P585 INFO Early stopping at epoch=12
2020-06-29 20:13:32,128 P585 INFO --- 7335/7335 batches finished ---
2020-06-29 20:13:32,193 P585 INFO Train loss: 0.428736
2020-06-29 20:13:32,194 P585 INFO Training finished.
2020-06-29 20:13:32,194 P585 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/FGCNN_criteo/min10/criteo_x4_5c863b0f/FGCNN_criteo_x4_5c863b0f_001_aa4d4a89_model.ckpt
2020-06-29 20:13:32,550 P585 INFO ****** Train/validation evaluation ******
2020-06-29 20:23:19,424 P585 INFO [Metrics] logloss: 0.426271 - AUC: 0.826100
2020-06-29 20:24:27,139 P585 INFO [Metrics] logloss: 0.440299 - AUC: 0.811523
2020-06-29 20:24:27,246 P585 INFO ******** Test evaluation ********
2020-06-29 20:24:27,246 P585 INFO Loading data...
2020-06-29 20:24:27,246 P585 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-29 20:24:28,203 P585 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-29 20:24:28,203 P585 INFO Loading test data done.
2020-06-29 20:25:39,519 P585 INFO [Metrics] logloss: 0.439800 - AUC: 0.812061


```
