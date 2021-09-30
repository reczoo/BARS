## CIN_Avazu_x4_002

A notebook to benchmark CIN on Avazu_x4_002 dataset.

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

3. Download the hyper-parameter configuration file: [AFM_avazu_x4_tuner_config_02.yaml](./AFM_avazu_x4_tuner_config_02.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/AFM_avazu_x4_tuner_config_02.yaml --tag 009 --gpu 0
  ```
  
### Results
```python
[Metrics] logloss: 0.372583 - AUC: 0.792293
```


### Logs
```python
2020-05-13 23:37:28,498 P22378 INFO {
    "batch_norm": "False",
    "batch_size": "2000",
    "cin_layer_units": "[250, 250, 250]",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x4_001_dea040c7",
    "model_root": "./Avazu/xDeepFM_avazu/",
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
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-05-13 23:37:28,499 P22378 INFO Set up feature encoder...
2020-05-13 23:37:28,499 P22378 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-05-13 23:37:28,499 P22378 INFO Loading data...
2020-05-13 23:37:28,502 P22378 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-05-13 23:37:30,796 P22378 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-05-13 23:37:32,052 P22378 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-13 23:37:32,173 P22378 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-13 23:37:32,173 P22378 INFO Loading train data done.
2020-05-13 23:37:44,934 P22378 INFO **** Start training: 16172 batches/epoch ****
2020-05-14 01:34:22,691 P22378 INFO [Metrics] logloss: 0.372709 - AUC: 0.792041
2020-05-14 01:34:22,841 P22378 INFO Save best model: monitor(max): 0.419332
2020-05-14 01:34:25,380 P22378 INFO --- 16172/16172 batches finished ---
2020-05-14 01:34:25,422 P22378 INFO Train loss: 0.381918
2020-05-14 01:34:25,422 P22378 INFO ************ Epoch=1 end ************
2020-05-14 03:31:04,779 P22378 INFO [Metrics] logloss: 0.392677 - AUC: 0.782110
2020-05-14 03:31:04,871 P22378 INFO Monitor(max) STOP: 0.389433 !
2020-05-14 03:31:04,871 P22378 INFO Reduce learning rate on plateau: 0.000100
2020-05-14 03:31:04,871 P22378 INFO --- 16172/16172 batches finished ---
2020-05-14 03:31:04,968 P22378 INFO Train loss: 0.318665
2020-05-14 03:31:04,968 P22378 INFO ************ Epoch=2 end ************
2020-05-14 05:27:44,832 P22378 INFO [Metrics] logloss: 0.434505 - AUC: 0.765835
2020-05-14 05:27:44,953 P22378 INFO Monitor(max) STOP: 0.331330 !
2020-05-14 05:27:44,954 P22378 INFO Reduce learning rate on plateau: 0.000010
2020-05-14 05:27:44,954 P22378 INFO Early stopping at epoch=3
2020-05-14 05:27:44,954 P22378 INFO --- 16172/16172 batches finished ---
2020-05-14 05:27:45,032 P22378 INFO Train loss: 0.257800
2020-05-14 05:27:45,032 P22378 INFO Training finished.
2020-05-14 05:27:45,032 P22378 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu/avazu_x4_001_d45ad60e/xDeepFM_avazu_x4_001_dea040c7_avazu_x4_001_d45ad60e_model.ckpt
2020-05-14 05:27:46,754 P22378 INFO ****** Train/validation evaluation ******
2020-05-14 05:54:49,750 P22378 INFO [Metrics] logloss: 0.332161 - AUC: 0.852738
2020-05-14 05:58:12,864 P22378 INFO [Metrics] logloss: 0.372709 - AUC: 0.792041
2020-05-14 05:58:13,091 P22378 INFO ******** Test evaluation ********
2020-05-14 05:58:13,091 P22378 INFO Loading data...
2020-05-14 05:58:13,091 P22378 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-05-14 05:58:13,615 P22378 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-14 05:58:13,615 P22378 INFO Loading test data done.
2020-05-14 06:01:36,594 P22378 INFO [Metrics] logloss: 0.372583 - AUC: 0.792293


```
