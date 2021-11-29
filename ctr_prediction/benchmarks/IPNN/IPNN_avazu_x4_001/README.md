## IPNN_Avazu_x4_001 

A notebook to benchmark IPNN on Avazu_x4_001 dataset.

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

3. Download the hyper-parameter configuration file: [IPNN_avazu_x4_tuner_config_01.yaml](./IPNN_avazu_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/IPNN_avazu_x4_tuner_config_01.yaml --tag 009 --gpu 0
  ```


### Results
```python
[Metrics] logloss: 0.371151 - AUC: 0.794385
```


### Logs
```python
2020-06-13 00:42:03,203 P39926 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-09",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "PNN",
    "model_id": "PNN_avazu_x4_3bbbc4c9_009_af8e3c6e",
    "model_root": "./Avazu/PNN_avazu/min2/",
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
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-13 00:42:03,204 P39926 INFO Set up feature encoder...
2020-06-13 00:42:03,204 P39926 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-13 00:42:04,620 P39926 INFO Total number of parameters: 62679601.
2020-06-13 00:42:04,620 P39926 INFO Loading data...
2020-06-13 00:42:04,622 P39926 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-13 00:42:07,589 P39926 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-13 00:42:09,018 P39926 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-13 00:42:09,191 P39926 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-13 00:42:09,191 P39926 INFO Loading train data done.
2020-06-13 00:42:11,933 P39926 INFO Start training: 3235 batches/epoch
2020-06-13 00:42:11,933 P39926 INFO ************ Epoch=1 start ************
2020-06-13 00:50:29,385 P39926 INFO [Metrics] logloss: 0.371209 - AUC: 0.794282
2020-06-13 00:50:29,386 P39926 INFO Save best model: monitor(max): 0.423073
2020-06-13 00:50:29,637 P39926 INFO --- 3235/3235 batches finished ---
2020-06-13 00:50:29,671 P39926 INFO Train loss: 0.379881
2020-06-13 00:50:29,671 P39926 INFO ************ Epoch=1 end ************
2020-06-13 00:58:46,087 P39926 INFO [Metrics] logloss: 0.380731 - AUC: 0.788821
2020-06-13 00:58:46,089 P39926 INFO Monitor(max) STOP: 0.408090 !
2020-06-13 00:58:46,089 P39926 INFO Reduce learning rate on plateau: 0.000100
2020-06-13 00:58:46,090 P39926 INFO --- 3235/3235 batches finished ---
2020-06-13 00:58:46,122 P39926 INFO Train loss: 0.330273
2020-06-13 00:58:46,122 P39926 INFO ************ Epoch=2 end ************
2020-06-13 01:07:01,652 P39926 INFO [Metrics] logloss: 0.429553 - AUC: 0.775582
2020-06-13 01:07:01,655 P39926 INFO Monitor(max) STOP: 0.346029 !
2020-06-13 01:07:01,655 P39926 INFO Reduce learning rate on plateau: 0.000010
2020-06-13 01:07:01,655 P39926 INFO Early stopping at epoch=3
2020-06-13 01:07:01,655 P39926 INFO --- 3235/3235 batches finished ---
2020-06-13 01:07:01,688 P39926 INFO Train loss: 0.285750
2020-06-13 01:07:01,688 P39926 INFO Training finished.
2020-06-13 01:07:01,688 P39926 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Avazu/PNN_avazu/min2/avazu_x4_3bbbc4c9/PNN_avazu_x4_3bbbc4c9_009_af8e3c6e_model.ckpt
2020-06-13 01:07:01,998 P39926 INFO ****** Train/validation evaluation ******
2020-06-13 01:07:25,518 P39926 INFO [Metrics] logloss: 0.371209 - AUC: 0.794282
2020-06-13 01:07:25,571 P39926 INFO ******** Test evaluation ********
2020-06-13 01:07:25,571 P39926 INFO Loading data...
2020-06-13 01:07:25,571 P39926 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-13 01:07:26,124 P39926 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-13 01:07:26,125 P39926 INFO Loading test data done.
2020-06-13 01:07:49,267 P39926 INFO [Metrics] logloss: 0.371151 - AUC: 0.794385
```
