## AFN_Avazu_x4_001

A notebook to benchmark AFN on Avazu_x4_001 dataset.

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

3. Download the hyper-parameter configuration file: [AFN_avazu_x4_tuner_config_04.yaml](./AFN_avazu_x4_tuner_config_04.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/AFN_avazu_x4_tuner_config_04.yaml --tag 003 --gpu 0
  ```



### Results
```python
[Metrics] logloss: 0.374026 - AUC: 0.790656
```


### Logs
```python
2020-07-15 10:01:26,771 P30984 INFO {
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
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-09",
    "ensemble_dnn": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1200",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_avazu_x4_3bbbc4c9_003_f329d7af",
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
2020-07-15 10:01:26,772 P30984 INFO Set up feature encoder...
2020-07-15 10:01:26,772 P30984 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-07-15 10:01:28,550 P30984 INFO Total number of parameters: 80253849.
2020-07-15 10:01:28,550 P30984 INFO Loading data...
2020-07-15 10:01:28,552 P30984 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-07-15 10:01:32,014 P30984 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-07-15 10:01:33,806 P30984 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-07-15 10:01:33,979 P30984 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-07-15 10:01:33,979 P30984 INFO Loading train data done.
2020-07-15 10:01:36,801 P30984 INFO Start training: 16172 batches/epoch
2020-07-15 10:01:36,801 P30984 INFO ************ Epoch=1 start ************
2020-07-15 10:26:17,214 P30984 INFO [Metrics] logloss: 0.378240 - AUC: 0.782770
2020-07-15 10:26:17,215 P30984 INFO Save best model: monitor(max): 0.404530
2020-07-15 10:26:17,519 P30984 INFO --- 16172/16172 batches finished ---
2020-07-15 10:26:17,566 P30984 INFO Train loss: 0.387026
2020-07-15 10:26:17,566 P30984 INFO ************ Epoch=1 end ************
2020-07-15 10:50:59,495 P30984 INFO [Metrics] logloss: 0.374248 - AUC: 0.790309
2020-07-15 10:50:59,501 P30984 INFO Save best model: monitor(max): 0.416061
2020-07-15 10:51:00,148 P30984 INFO --- 16172/16172 batches finished ---
2020-07-15 10:51:00,198 P30984 INFO Train loss: 0.360963
2020-07-15 10:51:00,198 P30984 INFO ************ Epoch=2 end ************
2020-07-15 11:15:40,712 P30984 INFO [Metrics] logloss: 0.378126 - AUC: 0.786924
2020-07-15 11:15:40,715 P30984 INFO Monitor(max) STOP: 0.408798 !
2020-07-15 11:15:40,715 P30984 INFO Reduce learning rate on plateau: 0.000100
2020-07-15 11:15:40,715 P30984 INFO --- 16172/16172 batches finished ---
2020-07-15 11:15:40,763 P30984 INFO Train loss: 0.338213
2020-07-15 11:15:40,763 P30984 INFO ************ Epoch=3 end ************
2020-07-15 11:40:28,438 P30984 INFO [Metrics] logloss: 0.407860 - AUC: 0.775994
2020-07-15 11:40:28,442 P30984 INFO Monitor(max) STOP: 0.368134 !
2020-07-15 11:40:28,442 P30984 INFO Reduce learning rate on plateau: 0.000010
2020-07-15 11:40:28,442 P30984 INFO Early stopping at epoch=4
2020-07-15 11:40:28,442 P30984 INFO --- 16172/16172 batches finished ---
2020-07-15 11:40:28,489 P30984 INFO Train loss: 0.308559
2020-07-15 11:40:28,489 P30984 INFO Training finished.
2020-07-15 11:40:28,489 P30984 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Avazu/AFN_avazu/min2/avazu_x4_3bbbc4c9/AFN_avazu_x4_3bbbc4c9_003_f329d7af_model.ckpt
2020-07-15 11:40:28,969 P30984 INFO ****** Train/validation evaluation ******
2020-07-15 11:48:31,887 P30984 INFO [Metrics] logloss: 0.334237 - AUC: 0.847131
2020-07-15 11:49:28,049 P30984 INFO [Metrics] logloss: 0.374248 - AUC: 0.790309
2020-07-15 11:49:28,090 P30984 INFO ******** Test evaluation ********
2020-07-15 11:49:28,090 P30984 INFO Loading data...
2020-07-15 11:49:28,090 P30984 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-07-15 11:49:29,545 P30984 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-07-15 11:49:29,545 P30984 INFO Loading test data done.
2020-07-15 11:50:31,137 P30984 INFO [Metrics] logloss: 0.374026 - AUC: 0.790656


```