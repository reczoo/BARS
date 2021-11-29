## FiBiNET_Avazu_x4_001

A notebook to benchmark FiBiNET on Avazu_x4_001 dataset.

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

3. Download the hyper-parameter configuration file: [FiBiNET_avazu_x4_tuner_config_02.yaml](./FiBiNET_avazu_x4_tuner_config_02.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/FiBiNET_avazu_x4_tuner_config_02.yaml --tag 002 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.370517 - AUC: 0.795261
```


### Logs
```python
2020-06-16 06:52:34,532 P2811 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "bilinear_type": "field_interaction",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[2000, 2000, 2000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiBiNET",
    "model_id": "FiBiNET_avazu_x4_3bbbc4c9_002_a11202f5",
    "model_root": "./Avazu/FiBiNET_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
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
2020-06-16 06:52:34,534 P2811 INFO Set up feature encoder...
2020-06-16 06:52:34,534 P2811 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-16 06:52:34,534 P2811 INFO Loading data...
2020-06-16 06:52:34,540 P2811 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-16 06:52:38,098 P2811 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-16 06:52:40,166 P2811 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-16 06:52:40,326 P2811 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-16 06:52:40,326 P2811 INFO Loading train data done.
2020-06-16 06:52:54,037 P2811 INFO Start training: 3235 batches/epoch
2020-06-16 06:52:54,037 P2811 INFO ************ Epoch=1 start ************
2020-06-16 07:13:41,176 P2811 INFO [Metrics] logloss: 0.370513 - AUC: 0.795259
2020-06-16 07:13:41,178 P2811 INFO Save best model: monitor(max): 0.424746
2020-06-16 07:13:41,859 P2811 INFO --- 3235/3235 batches finished ---
2020-06-16 07:13:41,908 P2811 INFO Train loss: 0.379595
2020-06-16 07:13:41,909 P2811 INFO ************ Epoch=1 end ************
2020-06-16 07:34:36,283 P2811 INFO [Metrics] logloss: 0.380033 - AUC: 0.790240
2020-06-16 07:34:36,289 P2811 INFO Monitor(max) STOP: 0.410206 !
2020-06-16 07:34:36,289 P2811 INFO Reduce learning rate on plateau: 0.000100
2020-06-16 07:34:36,289 P2811 INFO --- 3235/3235 batches finished ---
2020-06-16 07:34:36,355 P2811 INFO Train loss: 0.328817
2020-06-16 07:34:36,356 P2811 INFO ************ Epoch=2 end ************
2020-06-16 07:55:09,526 P2811 INFO [Metrics] logloss: 0.429876 - AUC: 0.775422
2020-06-16 07:55:09,530 P2811 INFO Monitor(max) STOP: 0.345546 !
2020-06-16 07:55:09,530 P2811 INFO Reduce learning rate on plateau: 0.000010
2020-06-16 07:55:09,530 P2811 INFO Early stopping at epoch=3
2020-06-16 07:55:09,530 P2811 INFO --- 3235/3235 batches finished ---
2020-06-16 07:55:09,590 P2811 INFO Train loss: 0.281427
2020-06-16 07:55:09,591 P2811 INFO Training finished.
2020-06-16 07:55:09,591 P2811 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/FiBiNET_avazu/min2/avazu_x4_3bbbc4c9/FiBiNET_avazu_x4_3bbbc4c9_002_a11202f5_model.ckpt
2020-06-16 07:55:10,206 P2811 INFO ****** Train/validation evaluation ******
2020-06-16 08:00:19,164 P2811 INFO [Metrics] logloss: 0.333176 - AUC: 0.851813
2020-06-16 08:00:55,994 P2811 INFO [Metrics] logloss: 0.370513 - AUC: 0.795259
2020-06-16 08:00:56,084 P2811 INFO ******** Test evaluation ********
2020-06-16 08:00:56,084 P2811 INFO Loading data...
2020-06-16 08:00:56,084 P2811 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-16 08:00:56,566 P2811 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-16 08:00:56,566 P2811 INFO Loading test data done.
2020-06-16 08:01:33,698 P2811 INFO [Metrics] logloss: 0.370517 - AUC: 0.795261


```
