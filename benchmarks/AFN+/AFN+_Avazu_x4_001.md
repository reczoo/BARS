## AFN+_Avazu_x4_001

A notebook to benchmark AFN+ on Avazu_x4_001 dataset.

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

3. Download the hyper-parameter configuration file: [AFN+_avazu_x4_tuner_config_07.yaml](./AFN+_avazu_x4_tuner_config_07.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/AFN+_avazu_x4_tuner_config_07.yaml --tag 003 --gpu 0
  ```




### Results
```python
[Metrics] logloss: 0.372589 - AUC: 0.792929
```


### Logs
```python
2020-07-15 19:27:16,774 P9162 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0",
    "afn_hidden_units": "[1000, 1000]",
    "batch_norm": "True",
    "batch_size": "2000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[1000, 1000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1200",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_avazu_x4_3bbbc4c9_003_59b37b70",
    "model_root": "./Avazu/AFN_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-15 19:27:16,775 P9162 INFO Set up feature encoder...
2020-07-15 19:27:16,775 P9162 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-15 19:27:19,933 P9162 INFO Total number of parameters: 141660453.
2020-07-15 19:27:19,933 P9162 INFO Loading data...
2020-07-15 19:27:19,935 P9162 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-15 19:27:22,730 P9162 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-15 19:27:23,953 P9162 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-15 19:27:24,064 P9162 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-15 19:27:24,064 P9162 INFO Loading train data done.
2020-07-15 19:27:28,742 P9162 INFO Start training: 16172 batches/epoch
2020-07-15 19:27:28,742 P9162 INFO ************ Epoch=1 start ************
2020-07-15 20:03:04,342 P9162 INFO [Metrics] logloss: 0.373648 - AUC: 0.790299
2020-07-15 20:03:04,343 P9162 INFO Save best model: monitor(max): 0.416651
2020-07-15 20:03:04,887 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 20:03:04,924 P9162 INFO Train loss: 0.380791
2020-07-15 20:03:04,924 P9162 INFO ************ Epoch=1 end ************
2020-07-15 20:39:10,562 P9162 INFO [Metrics] logloss: 0.372653 - AUC: 0.792778
2020-07-15 20:39:10,566 P9162 INFO Save best model: monitor(max): 0.420125
2020-07-15 20:39:11,711 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 20:39:11,749 P9162 INFO Train loss: 0.350438
2020-07-15 20:39:11,749 P9162 INFO ************ Epoch=2 end ************
2020-07-15 21:15:01,438 P9162 INFO [Metrics] logloss: 0.375738 - AUC: 0.791107
2020-07-15 21:15:01,440 P9162 INFO Monitor(max) STOP: 0.415369 !
2020-07-15 21:15:01,440 P9162 INFO Reduce learning rate on plateau: 0.000100
2020-07-15 21:15:01,441 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 21:15:01,480 P9162 INFO Train loss: 0.336072
2020-07-15 21:15:01,480 P9162 INFO ************ Epoch=3 end ************
2020-07-15 21:50:47,278 P9162 INFO [Metrics] logloss: 0.412006 - AUC: 0.775337
2020-07-15 21:50:47,292 P9162 INFO Monitor(max) STOP: 0.363331 !
2020-07-15 21:50:47,292 P9162 INFO Reduce learning rate on plateau: 0.000010
2020-07-15 21:50:47,292 P9162 INFO Early stopping at epoch=4
2020-07-15 21:50:47,292 P9162 INFO --- 16172/16172 batches finished ---
2020-07-15 21:50:47,333 P9162 INFO Train loss: 0.299461
2020-07-15 21:50:47,334 P9162 INFO Training finished.
2020-07-15 21:50:47,334 P9162 INFO Load best model: /home/xxx/zhujieming/OpenCTR1030/benchmarks/Avazu/AFN_avazu/min2/avazu_x4_3bbbc4c9/AFN_avazu_x4_3bbbc4c9_003_59b37b70_model.ckpt
2020-07-15 21:50:48,542 P9162 INFO ****** Train/validation evaluation ******
2020-07-15 22:00:20,270 P9162 INFO [Metrics] logloss: 0.327129 - AUC: 0.855984
2020-07-15 22:01:26,125 P9162 INFO [Metrics] logloss: 0.372653 - AUC: 0.792778
2020-07-15 22:01:26,172 P9162 INFO ******** Test evaluation ********
2020-07-15 22:01:26,172 P9162 INFO Loading data...
2020-07-15 22:01:26,172 P9162 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-15 22:01:27,500 P9162 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-15 22:01:27,500 P9162 INFO Loading test data done.
2020-07-15 22:02:34,136 P9162 INFO [Metrics] logloss: 0.372589 - AUC: 0.792929


```
