## CCPM_Avazu_x4_002

A notebook to benchmark CCPM on Avazu_x4_002 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default <OOV> token. Note that we found that min_category_count=1 performs the best, which is surprising.

We fix embedding_dim=40 following the existing FGCNN work.
### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Avazu/Avazu_x4/split_avazu_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [CCPM_avazu_x4_tuner_config_01.yaml](./002/CCPM_avazu_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/002/CCPM_avazu_x4_tuner_config_01.yaml --tag 025 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.372100 - AUC: 0.793210
```


### Logs
```python
2020-05-12 08:32:12,386 P6185 INFO {
    "activation": "Tanh",
    "batch_size": "10000",
    "channels": "[16, 32, 64]",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_74410863",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "kernel_heights": "[7, 5, 3]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "CCPM",
    "model_id": "CCPM_avazu_x4_025_a3bc05b9",
    "model_root": "./Avazu/CCPM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_001_74410863/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_74410863/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_001_74410863/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-05-12 08:32:12,389 P6185 INFO Set up feature encoder...
2020-05-12 08:32:12,389 P6185 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_74410863/feature_map.json
2020-05-12 08:32:12,389 P6185 INFO Loading data...
2020-05-12 08:32:12,434 P6185 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_74410863/train.h5
2020-05-12 08:32:16,290 P6185 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_74410863/valid.h5
2020-05-12 08:32:17,632 P6185 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-12 08:32:17,742 P6185 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-12 08:32:17,743 P6185 INFO Loading train data done.
2020-05-12 08:32:31,507 P6185 INFO **** Start training: 3235 batches/epoch ****
2020-05-12 09:31:16,216 P6185 INFO [Metrics] logloss: 0.372102 - AUC: 0.793156
2020-05-12 09:31:16,224 P6185 INFO Save best model: monitor(max): 0.421054
2020-05-12 09:31:17,847 P6185 INFO --- 3235/3235 batches finished ---
2020-05-12 09:31:17,900 P6185 INFO Train loss: 0.384212
2020-05-12 09:31:17,900 P6185 INFO ************ Epoch=1 end ************
2020-05-12 10:29:22,954 P6185 INFO [Metrics] logloss: 0.387160 - AUC: 0.781956
2020-05-12 10:29:22,960 P6185 INFO Monitor(max) STOP: 0.394796 !
2020-05-12 10:29:22,960 P6185 INFO Reduce learning rate on plateau: 0.000100
2020-05-12 10:29:22,960 P6185 INFO --- 3235/3235 batches finished ---
2020-05-12 10:29:23,018 P6185 INFO Train loss: 0.311386
2020-05-12 10:29:23,018 P6185 INFO ************ Epoch=2 end ************
2020-05-12 11:27:31,848 P6185 INFO [Metrics] logloss: 0.439963 - AUC: 0.763997
2020-05-12 11:27:31,860 P6185 INFO Monitor(max) STOP: 0.324034 !
2020-05-12 11:27:31,860 P6185 INFO Reduce learning rate on plateau: 0.000010
2020-05-12 11:27:31,860 P6185 INFO Early stopping at epoch=3
2020-05-12 11:27:31,860 P6185 INFO --- 3235/3235 batches finished ---
2020-05-12 11:27:31,914 P6185 INFO Train loss: 0.243214
2020-05-12 11:27:31,915 P6185 INFO Training finished.
2020-05-12 11:27:31,915 P6185 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/CCPM_avazu/avazu_x4_001_74410863/CCPM_avazu_x4_025_a3bc05b9_model.ckpt
2020-05-12 11:27:33,958 P6185 INFO ****** Train/validation evaluation ******
2020-05-12 11:59:06,294 P6185 INFO [Metrics] logloss: 0.326900 - AUC: 0.859540
2020-05-12 12:03:02,097 P6185 INFO [Metrics] logloss: 0.372102 - AUC: 0.793156
2020-05-12 12:03:02,216 P6185 INFO ******** Test evaluation ********
2020-05-12 12:03:02,216 P6185 INFO Loading data...
2020-05-12 12:03:02,216 P6185 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_74410863/test.h5
2020-05-12 12:03:02,931 P6185 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-12 12:03:02,932 P6185 INFO Loading test data done.
2020-05-12 12:06:58,854 P6185 INFO [Metrics] logloss: 0.372100 - AUC: 0.793210

```
