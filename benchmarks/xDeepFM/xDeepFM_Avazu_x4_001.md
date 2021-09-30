## xDeepFM_Avazu_x4_001 

A notebook to benchmark xDeepFM on Avazu_x4_001 dataset.

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
In this setting, we preprocess the data split by removing the ``id`` field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default ``<OOV>`` token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless ``id`` field nor specially preprocess the timestamp field.

To make a fair comparison, we fix **embedding_dim=16** as with AutoInt.


### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Avazu/Avazu_x4/split_avazu_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [xDeepFM_avazu_x4_tuner_config_03.yaml](./xDeepFM_avazu_x4_tuner_config_03.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/xDeepFM_avazu_x4_tuner_config_03.yaml --tag 001 --gpu 0
  ```


### Results
```python
[Metrics] logloss: 0.371780 - AUC: 0.793283
```


### Logs
```python
2020-06-14 21:28:50,584 P660 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "cin_layer_units": "[276]",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_hidden_units": "[500, 500, 500]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x4_3bbbc4c9_001_5e656a3d",
    "model_root": "./Avazu/xDeepFM_avazu/min2/",
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
2020-06-14 21:28:50,587 P660 INFO Set up feature encoder...
2020-06-14 21:28:50,587 P660 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-14 21:28:50,588 P660 INFO Loading data...
2020-06-14 21:28:50,593 P660 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-14 21:28:53,183 P660 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-14 21:28:54,479 P660 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-14 21:28:54,618 P660 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-14 21:28:54,618 P660 INFO Loading train data done.
2020-06-14 21:29:03,158 P660 INFO Start training: 3235 batches/epoch
2020-06-14 21:29:03,158 P660 INFO ************ Epoch=1 start ************
2020-06-14 21:38:14,282 P660 INFO [Metrics] logloss: 0.371895 - AUC: 0.793132
2020-06-14 21:38:14,283 P660 INFO Save best model: monitor(max): 0.421237
2020-06-14 21:38:14,626 P660 INFO --- 3235/3235 batches finished ---
2020-06-14 21:38:14,669 P660 INFO Train loss: 0.380561
2020-06-14 21:38:14,670 P660 INFO ************ Epoch=1 end ************
2020-06-14 21:47:26,949 P660 INFO [Metrics] logloss: 0.380666 - AUC: 0.788586
2020-06-14 21:47:26,952 P660 INFO Monitor(max) STOP: 0.407919 !
2020-06-14 21:47:26,952 P660 INFO Reduce learning rate on plateau: 0.000100
2020-06-14 21:47:26,952 P660 INFO --- 3235/3235 batches finished ---
2020-06-14 21:47:26,992 P660 INFO Train loss: 0.331775
2020-06-14 21:47:26,993 P660 INFO ************ Epoch=2 end ************
2020-06-14 21:56:37,720 P660 INFO [Metrics] logloss: 0.424659 - AUC: 0.775277
2020-06-14 21:56:37,724 P660 INFO Monitor(max) STOP: 0.350618 !
2020-06-14 21:56:37,724 P660 INFO Reduce learning rate on plateau: 0.000010
2020-06-14 21:56:37,727 P660 INFO Early stopping at epoch=3
2020-06-14 21:56:37,728 P660 INFO --- 3235/3235 batches finished ---
2020-06-14 21:56:37,776 P660 INFO Train loss: 0.285276
2020-06-14 21:56:37,777 P660 INFO Training finished.
2020-06-14 21:56:37,777 P660 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu/min2/avazu_x4_3bbbc4c9/xDeepFM_avazu_x4_3bbbc4c9_001_5e656a3d_model.ckpt
2020-06-14 21:56:38,255 P660 INFO ****** Train/validation evaluation ******
2020-06-14 21:59:49,037 P660 INFO [Metrics] logloss: 0.338127 - AUC: 0.845617
2020-06-14 22:00:11,516 P660 INFO [Metrics] logloss: 0.371895 - AUC: 0.793132
2020-06-14 22:00:11,592 P660 INFO ******** Test evaluation ********
2020-06-14 22:00:11,592 P660 INFO Loading data...
2020-06-14 22:00:11,592 P660 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-14 22:00:12,305 P660 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-14 22:00:12,305 P660 INFO Loading test data done.
2020-06-14 22:00:36,785 P660 INFO [Metrics] logloss: 0.371780 - AUC: 0.793283
```
