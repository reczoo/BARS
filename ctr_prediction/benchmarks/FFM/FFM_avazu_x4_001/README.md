## FFM_Avazu_x4_001

A notebook to benchmark FFM on Avazu_x4_001 dataset.

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

3. Download the hyper-parameter configuration file: [FFM_avazu_x4_tuner_config_01.yaml](./FFM_avazu_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/FFM_avazu_x4_tuner_config_01.yaml --tag 003 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.371864 - AUC: 0.793279
```

### Logs
The following log is to be updated.
```python
2020-05-13 23:47:15,648 P555 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FFM",
    "model_id": "FFM_avazu_x4_3bbbc4c9_001_9dcde48f",
    "model_root": "./Avazu/FFM_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "0",
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
2020-05-13 23:47:15,663 P555 INFO Set up feature encoder...
2020-05-13 23:47:15,663 P555 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-05-13 23:47:15,664 P555 INFO Loading data...
2020-05-13 23:47:15,671 P555 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-05-13 23:47:18,481 P555 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-05-13 23:47:19,842 P555 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-13 23:47:19,942 P555 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-13 23:47:19,942 P555 INFO Loading train data done.
2020-05-13 23:47:43,035 P555 INFO **** Start training: 3235 batches/epoch ****
2020-05-14 01:02:08,064 P555 INFO [Metrics] logloss: 0.371391 - AUC: 0.794224
2020-05-14 01:02:08,065 P555 INFO Save best model: monitor(max): 0.422833
2020-05-14 01:02:12,109 P555 INFO --- 3235/3235 batches finished ---
2020-05-14 01:02:12,180 P555 INFO Train loss: 0.381109
2020-05-14 01:02:12,181 P555 INFO ************ Epoch=1 end ************
2020-05-14 02:16:47,682 P555 INFO [Metrics] logloss: 0.378682 - AUC: 0.790775
2020-05-14 02:16:47,688 P555 INFO Monitor(max) STOP: 0.412093 !
2020-05-14 02:16:47,688 P555 INFO Reduce learning rate on plateau: 0.000100
2020-05-14 02:16:47,688 P555 INFO --- 3235/3235 batches finished ---
2020-05-14 02:16:47,767 P555 INFO Train loss: 0.327801
2020-05-14 02:16:47,768 P555 INFO ************ Epoch=2 end ************
2020-05-14 03:31:21,825 P555 INFO [Metrics] logloss: 0.402067 - AUC: 0.781527
2020-05-14 03:31:21,826 P555 INFO Monitor(max) STOP: 0.379460 !
2020-05-14 03:31:21,826 P555 INFO Reduce learning rate on plateau: 0.000010
2020-05-14 03:31:21,826 P555 INFO Early stopping at epoch=3
2020-05-14 03:31:21,826 P555 INFO --- 3235/3235 batches finished ---
2020-05-14 03:31:21,887 P555 INFO Train loss: 0.277228
2020-05-14 03:31:21,888 P555 INFO Training finished.
2020-05-14 03:31:21,888 P555 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/FFM_avazu/min2/avazu_x4_3bbbc4c9/FFM_avazu_x4_3bbbc4c9_001_9dcde48f_model.ckpt
2020-05-14 03:31:28,151 P555 INFO ****** Train/validation evaluation ******
2020-05-14 03:35:36,057 P555 INFO [Metrics] logloss: 0.331382 - AUC: 0.854428
2020-05-14 03:36:05,955 P555 INFO [Metrics] logloss: 0.371391 - AUC: 0.794224
2020-05-14 03:36:06,090 P555 INFO ******** Test evaluation ********
2020-05-14 03:36:06,091 P555 INFO Loading data...
2020-05-14 03:36:06,091 P555 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-05-14 03:36:06,847 P555 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-14 03:36:06,848 P555 INFO Loading test data done.
2020-05-14 03:36:34,581 P555 INFO [Metrics] logloss: 0.371450 - AUC: 0.794166


```
