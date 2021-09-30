## AutoInt_Avazu_x4_001

A notebook to benchmark AutoInt on Avazu_x4_001 dataset.

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

3. Download the hyper-parameter configuration file: [AutoInt_avazu_x4_tuner_config_04.yaml](./AutoInt_avazu_x4_tuner_config_04.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/AutoInt_avazu_x4_tuner_config_04.yaml --tag 013 --gpu 0
  ```

### Results
```python
[Metrics] logloss: 0.374519 - AUC: 0.789103
```


### Logs
```python
2020-06-14 10:31:14,719 P14792 INFO {
    "attention_dim": "128",
    "attention_layers": "7",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x4_3bbbc4c9_072_112d976f",
    "model_root": "./Avazu/AutoInt_avazu/min2/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
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
    "use_residual": "False",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-14 10:31:14,721 P14792 INFO Set up feature encoder...
2020-06-14 10:31:14,721 P14792 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-14 10:31:14,721 P14792 INFO Loading data...
2020-06-14 10:31:14,744 P14792 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-14 10:31:18,677 P14792 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-14 10:31:20,442 P14792 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-14 10:31:20,621 P14792 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-14 10:31:20,621 P14792 INFO Loading train data done.
2020-06-14 10:31:28,827 P14792 INFO Start training: 3235 batches/epoch
2020-06-14 10:31:28,827 P14792 INFO ************ Epoch=1 start ************
2020-06-14 10:43:35,262 P14792 INFO [Metrics] logloss: 0.377925 - AUC: 0.783227
2020-06-14 10:43:35,263 P14792 INFO Save best model: monitor(max): 0.405302
2020-06-14 10:43:36,155 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 10:43:36,213 P14792 INFO Train loss: 0.393201
2020-06-14 10:43:36,213 P14792 INFO ************ Epoch=1 end ************
2020-06-14 10:55:35,535 P14792 INFO [Metrics] logloss: 0.374649 - AUC: 0.788863
2020-06-14 10:55:35,540 P14792 INFO Save best model: monitor(max): 0.414213
2020-06-14 10:55:36,005 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 10:55:36,066 P14792 INFO Train loss: 0.379079
2020-06-14 10:55:36,067 P14792 INFO ************ Epoch=2 end ************
2020-06-14 11:07:31,028 P14792 INFO [Metrics] logloss: 0.374997 - AUC: 0.789109
2020-06-14 11:07:31,031 P14792 INFO Monitor(max) STOP: 0.414112 !
2020-06-14 11:07:31,031 P14792 INFO Reduce learning rate on plateau: 0.000100
2020-06-14 11:07:31,031 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 11:07:31,092 P14792 INFO Train loss: 0.367851
2020-06-14 11:07:31,093 P14792 INFO ************ Epoch=3 end ************
2020-06-14 11:19:25,609 P14792 INFO [Metrics] logloss: 0.399006 - AUC: 0.779783
2020-06-14 11:19:25,613 P14792 INFO Monitor(max) STOP: 0.380777 !
2020-06-14 11:19:25,613 P14792 INFO Reduce learning rate on plateau: 0.000010
2020-06-14 11:19:25,614 P14792 INFO Early stopping at epoch=4
2020-06-14 11:19:25,614 P14792 INFO --- 3235/3235 batches finished ---
2020-06-14 11:19:25,674 P14792 INFO Train loss: 0.324028
2020-06-14 11:19:25,674 P14792 INFO Training finished.
2020-06-14 11:19:25,674 P14792 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/AutoInt_avazu/min2/avazu_x4_3bbbc4c9/AutoInt_avazu_x4_3bbbc4c9_072_112d976f_model.ckpt
2020-06-14 11:19:26,097 P14792 INFO ****** Train/validation evaluation ******
2020-06-14 11:22:58,032 P14792 INFO [Metrics] logloss: 0.339152 - AUC: 0.842612
2020-06-14 11:23:22,259 P14792 INFO [Metrics] logloss: 0.374649 - AUC: 0.788863
2020-06-14 11:23:22,342 P14792 INFO ******** Test evaluation ********
2020-06-14 11:23:22,342 P14792 INFO Loading data...
2020-06-14 11:23:22,342 P14792 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-14 11:23:23,028 P14792 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-14 11:23:23,029 P14792 INFO Loading test data done.
2020-06-14 11:23:48,327 P14792 INFO [Metrics] logloss: 0.374519 - AUC: 0.789103



```
