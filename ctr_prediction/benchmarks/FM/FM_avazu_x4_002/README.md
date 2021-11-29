## FM_Avazu_x4_002

A notebook to benchmark FM on Avazu_x4_002 dataset.

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

3. Download the hyper-parameter configuration file: [FM_avazu_x4_tuner_config_04.yaml](./FM_avazu_x4_tuner_config_04.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/FM_avazu_x4_tuner_config_04.yaml --tag 003 --gpu 0
  ```
### Results
```python
[Metrics] logloss: 0.373561 - AUC: 0.790887
```


### Logs
```python
2020-02-23 14:54:49,630 P28545 INFO {
    "batch_size": "5000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FM",
    "model_id": "FM_avazu_x4_003_509e8ea6",
    "model_root": "./Avazu/FM_avazu/",
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
    "use_hdf5": "True",
    "verbose": "1",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-02-23 14:54:49,631 P28545 INFO Set up feature encoder...
2020-02-23 14:54:49,631 P28545 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-23 14:54:49,632 P28545 INFO Loading data...
2020-02-23 14:54:49,634 P28545 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-23 14:54:52,575 P28545 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-23 14:54:54,131 P28545 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-23 14:54:54,302 P28545 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-23 14:54:54,302 P28545 INFO Loading train data done.
2020-02-23 14:55:05,283 P28545 INFO **** Start training: 6469 batches/epoch ****
2020-02-23 15:12:54,365 P28545 INFO [Metrics] logloss: 0.373535 - AUC: 0.790776
2020-02-23 15:12:54,456 P28545 INFO Save best model: monitor(max): 0.417241
2020-02-23 15:12:55,856 P28545 INFO --- 6469/6469 batches finished ---
2020-02-23 15:12:55,910 P28545 INFO Train loss: 0.383358
2020-02-23 15:12:55,910 P28545 INFO ************ Epoch=1 end ************
2020-02-23 15:30:45,511 P28545 INFO [Metrics] logloss: 0.383542 - AUC: 0.786443
2020-02-23 15:30:45,602 P28545 INFO Monitor(max) STOP: 0.402901 !
2020-02-23 15:30:45,602 P28545 INFO Reduce learning rate on plateau: 0.000100
2020-02-23 15:30:45,602 P28545 INFO --- 6469/6469 batches finished ---
2020-02-23 15:30:45,671 P28545 INFO Train loss: 0.325166
2020-02-23 15:30:45,671 P28545 INFO ************ Epoch=2 end ************
2020-02-23 15:48:34,738 P28545 INFO [Metrics] logloss: 0.402431 - AUC: 0.778739
2020-02-23 15:48:34,845 P28545 INFO Monitor(max) STOP: 0.376308 !
2020-02-23 15:48:34,845 P28545 INFO Reduce learning rate on plateau: 0.000010
2020-02-23 15:48:34,845 P28545 INFO Early stopping at epoch=3
2020-02-23 15:48:34,845 P28545 INFO --- 6469/6469 batches finished ---
2020-02-23 15:48:34,953 P28545 INFO Train loss: 0.270158
2020-02-23 15:48:34,953 P28545 INFO Training finished.
2020-02-23 15:48:34,953 P28545 INFO Load best model: /home/xxx/xxx/OpenCTR1030/benchmarks/Avazu/FM_avazu/avazu_x4_001_d45ad60e/FM_avazu_x4_003_509e8ea6_avazu_x4_001_d45ad60e_model.ckpt
2020-02-23 15:48:36,859 P28545 INFO ****** Train/validation evaluation ******
2020-02-23 15:53:43,890 P28545 INFO [Metrics] logloss: 0.331818 - AUC: 0.853327
2020-02-23 15:54:22,893 P28545 INFO [Metrics] logloss: 0.373535 - AUC: 0.790776
2020-02-23 15:54:23,067 P28545 INFO ******** Test evaluation ********
2020-02-23 15:54:23,067 P28545 INFO Loading data...
2020-02-23 15:54:23,067 P28545 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-23 15:54:23,944 P28545 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-23 15:54:23,944 P28545 INFO Loading test data done.
2020-02-23 15:55:00,616 P28545 INFO [Metrics] logloss: 0.373561 - AUC: 0.790887

```
