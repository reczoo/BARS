## DCN_Avazu_x4_001

A notebook to benchmark DCN on Avazu_x4_001 dataset.

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

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Avazu/Avazu_x4/split_avazu_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [DCN_avazu_x4_tuner_config_02.yaml](./DCN_avazu_x4_tuner_config_02.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/DCN_avazu_x4_tuner_config_02.yaml --tag 018 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.371936 - AUC: 0.793061
```


### Logs
```python
2020-06-13 17:44:00,674 P4075 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[2000, 2000, 2000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_avazu_x4_3bbbc4c9_018_b2ab697a",
    "model_root": "./Avazu/DCN_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-13 17:44:00,677 P4075 INFO Set up feature encoder...
2020-06-13 17:44:00,677 P4075 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-13 17:44:00,677 P4075 INFO Loading data...
2020-06-13 17:44:00,681 P4075 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-13 17:44:04,530 P4075 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-13 17:44:06,164 P4075 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-13 17:44:06,269 P4075 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-13 17:44:06,270 P4075 INFO Loading train data done.
2020-06-13 17:44:11,963 P4075 INFO Start training: 3235 batches/epoch
2020-06-13 17:44:11,964 P4075 INFO ************ Epoch=1 start ************
2020-06-13 17:50:38,489 P4075 INFO [Metrics] logloss: 0.372058 - AUC: 0.792786
2020-06-13 17:50:38,489 P4075 INFO Save best model: monitor(max): 0.420728
2020-06-13 17:50:38,791 P4075 INFO --- 3235/3235 batches finished ---
2020-06-13 17:50:38,850 P4075 INFO Train loss: 0.380479
2020-06-13 17:50:38,850 P4075 INFO ************ Epoch=1 end ************
2020-06-13 17:57:03,463 P4075 INFO [Metrics] logloss: 0.379733 - AUC: 0.788047
2020-06-13 17:57:03,468 P4075 INFO Monitor(max) STOP: 0.408314 !
2020-06-13 17:57:03,468 P4075 INFO Reduce learning rate on plateau: 0.000100
2020-06-13 17:57:03,468 P4075 INFO --- 3235/3235 batches finished ---
2020-06-13 17:57:03,522 P4075 INFO Train loss: 0.334254
2020-06-13 17:57:03,522 P4075 INFO ************ Epoch=2 end ************
2020-06-13 18:03:31,323 P4075 INFO [Metrics] logloss: 0.423422 - AUC: 0.776726
2020-06-13 18:03:31,329 P4075 INFO Monitor(max) STOP: 0.353304 !
2020-06-13 18:03:31,329 P4075 INFO Reduce learning rate on plateau: 0.000010
2020-06-13 18:03:31,329 P4075 INFO Early stopping at epoch=3
2020-06-13 18:03:31,329 P4075 INFO --- 3235/3235 batches finished ---
2020-06-13 18:03:31,387 P4075 INFO Train loss: 0.294529
2020-06-13 18:03:31,387 P4075 INFO Training finished.
2020-06-13 18:03:31,387 P4075 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/DCN_avazu/min2/avazu_x4_3bbbc4c9/DCN_avazu_x4_3bbbc4c9_018_b2ab697a_model.ckpt
2020-06-13 18:03:31,891 P4075 INFO ****** Train/validation evaluation ******
2020-06-13 18:06:58,280 P4075 INFO [Metrics] logloss: 0.339480 - AUC: 0.843293
2020-06-13 18:07:21,422 P4075 INFO [Metrics] logloss: 0.372058 - AUC: 0.792786
2020-06-13 18:07:21,501 P4075 INFO ******** Test evaluation ********
2020-06-13 18:07:21,501 P4075 INFO Loading data...
2020-06-13 18:07:21,501 P4075 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-13 18:07:22,092 P4075 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-13 18:07:22,092 P4075 INFO Loading test data done.
2020-06-13 18:07:44,456 P4075 INFO [Metrics] logloss: 0.371936 - AUC: 0.793061

```
