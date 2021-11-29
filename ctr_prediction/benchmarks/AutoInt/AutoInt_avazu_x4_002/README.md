## AutoInt_Avazu_x4_002

A notebook to benchmark AutoInt on Avazu_x4_002 dataset.

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
 [Metrics] logloss: 0.372581 - AUC: 0.792228
```


### Logs
```python
2020-05-11 18:36:47,739 P26065 INFO {
    "attention_dim": "160",
    "attention_layers": "6",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x4_013_3a66ab94",
    "model_root": "./Avazu/AutoInt_avazu/",
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
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-05-11 18:36:47,740 P26065 INFO Set up feature encoder...
2020-05-11 18:36:47,740 P26065 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-05-11 18:36:47,741 P26065 INFO Loading data...
2020-05-11 18:36:47,742 P26065 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-05-11 18:36:50,514 P26065 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-05-11 18:36:51,806 P26065 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-11 18:36:51,922 P26065 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-11 18:36:51,922 P26065 INFO Loading train data done.
2020-05-11 18:37:05,045 P26065 INFO **** Start training: 3235 batches/epoch ****
2020-05-11 18:59:28,106 P26065 INFO [Metrics] logloss: 0.372695 - AUC: 0.791962
2020-05-11 18:59:28,109 P26065 INFO Save best model: monitor(max): 0.419268
2020-05-11 18:59:29,324 P26065 INFO --- 3235/3235 batches finished ---
2020-05-11 18:59:29,357 P26065 INFO Train loss: 0.382909
2020-05-11 18:59:29,357 P26065 INFO ************ Epoch=1 end ************
2020-05-11 19:21:51,275 P26065 INFO [Metrics] logloss: 0.467224 - AUC: 0.756973
2020-05-11 19:21:51,277 P26065 INFO Monitor(max) STOP: 0.289749 !
2020-05-11 19:21:51,278 P26065 INFO Reduce learning rate on plateau: 0.000100
2020-05-11 19:21:51,278 P26065 INFO --- 3235/3235 batches finished ---
2020-05-11 19:21:51,311 P26065 INFO Train loss: 0.281661
2020-05-11 19:21:51,311 P26065 INFO ************ Epoch=2 end ************
2020-05-11 19:44:14,621 P26065 INFO [Metrics] logloss: 0.516160 - AUC: 0.749198
2020-05-11 19:44:14,624 P26065 INFO Monitor(max) STOP: 0.233038 !
2020-05-11 19:44:14,624 P26065 INFO Reduce learning rate on plateau: 0.000010
2020-05-11 19:44:14,624 P26065 INFO Early stopping at epoch=3
2020-05-11 19:44:14,624 P26065 INFO --- 3235/3235 batches finished ---
2020-05-11 19:44:14,657 P26065 INFO Train loss: 0.254104
2020-05-11 19:44:14,657 P26065 INFO Training finished.
2020-05-11 19:44:14,657 P26065 INFO Load best model: /home/zhujieming/xxx/OpenCTR1030/benchmarks/Avazu/AutoInt_avazu/avazu_x4_001_d45ad60e/AutoInt_avazu_x4_013_3a66ab94_model.ckpt
2020-05-11 19:44:16,275 P26065 INFO ****** Train/validation evaluation ******
2020-05-11 19:44:56,883 P26065 INFO [Metrics] logloss: 0.372695 - AUC: 0.791962
2020-05-11 19:44:56,984 P26065 INFO ******** Test evaluation ********
2020-05-11 19:44:56,984 P26065 INFO Loading data...
2020-05-11 19:44:56,984 P26065 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-05-11 19:44:57,417 P26065 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-11 19:44:57,418 P26065 INFO Loading test data done.
2020-05-11 19:45:38,162 P26065 INFO [Metrics] logloss: 0.372581 - AUC: 0.792228

```
