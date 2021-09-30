## AFN_Criteo_x4_001

A notebook to benchmark AFN+ on Criteo_x4_001 dataset.

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
This dataset split follows the setting in the AutoInt work. Specifically, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting.

### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [AFN+_criteo_x4_tuner_config_12.yaml](./AFN+_criteo_x4_tuner_config_12.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/AFN+_criteo_x4_tuner_config_12.yaml --tag 004 --gpu 0
  ```





### Results
```python
[Metrics] logloss: 0.438404 - AUC: 0.813804
```


### Logs
```python
2020-07-25 17:32:28,416 P50682 INFO {
    "afn_activations": "relu",
    "afn_dropout": "0.2",
    "afn_hidden_units": "[1000, 1000, 1000, 1000]",
    "batch_norm": "True",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_dropout": "0",
    "dnn_hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "ensemble_dnn": "True",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "logarithmic_neurons": "1200",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AFN",
    "model_id": "AFN_criteo_x4_5c863b0f_004_81f459ad",
    "model_root": "./Criteo/AFN_criteo/min10/",
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
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-07-25 17:32:28,416 P50682 INFO Set up feature encoder...
2020-07-25 17:32:28,417 P50682 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-07-25 17:32:36,765 P50682 INFO Total number of parameters: 56044939.
2020-07-25 17:32:36,765 P50682 INFO Loading data...
2020-07-25 17:32:36,767 P50682 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-07-25 17:32:42,138 P50682 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-07-25 17:32:43,864 P50682 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-07-25 17:32:43,997 P50682 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-25 17:32:43,997 P50682 INFO Loading train data done.
2020-07-25 17:33:22,590 P50682 INFO Start training: 3668 batches/epoch
2020-07-25 17:33:22,590 P50682 INFO ************ Epoch=1 start ************
2020-07-25 18:03:51,595 P50682 INFO [Metrics] logloss: 0.445220 - AUC: 0.806394
2020-07-25 18:03:51,597 P50682 INFO Save best model: monitor(max): 0.361174
2020-07-25 18:03:51,823 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 18:03:51,868 P50682 INFO Train loss: 0.458252
2020-07-25 18:03:51,868 P50682 INFO ************ Epoch=1 end ************
2020-07-25 18:34:23,801 P50682 INFO [Metrics] logloss: 0.443267 - AUC: 0.808655
2020-07-25 18:34:23,803 P50682 INFO Save best model: monitor(max): 0.365388
2020-07-25 18:34:24,217 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 18:34:24,264 P50682 INFO Train loss: 0.452600
2020-07-25 18:34:24,264 P50682 INFO ************ Epoch=2 end ************
2020-07-25 19:04:52,152 P50682 INFO [Metrics] logloss: 0.441816 - AUC: 0.809902
2020-07-25 19:04:52,153 P50682 INFO Save best model: monitor(max): 0.368086
2020-07-25 19:04:52,555 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 19:04:52,602 P50682 INFO Train loss: 0.450986
2020-07-25 19:04:52,602 P50682 INFO ************ Epoch=3 end ************
2020-07-25 19:35:25,662 P50682 INFO [Metrics] logloss: 0.440879 - AUC: 0.810772
2020-07-25 19:35:25,664 P50682 INFO Save best model: monitor(max): 0.369893
2020-07-25 19:35:26,148 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 19:35:26,193 P50682 INFO Train loss: 0.449817
2020-07-25 19:35:26,193 P50682 INFO ************ Epoch=4 end ************
2020-07-25 20:05:54,067 P50682 INFO [Metrics] logloss: 0.440662 - AUC: 0.811143
2020-07-25 20:05:54,069 P50682 INFO Save best model: monitor(max): 0.370481
2020-07-25 20:05:54,477 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 20:05:54,522 P50682 INFO Train loss: 0.448938
2020-07-25 20:05:54,522 P50682 INFO ************ Epoch=5 end ************
2020-07-25 20:36:28,849 P50682 INFO [Metrics] logloss: 0.440622 - AUC: 0.811057
2020-07-25 20:36:28,851 P50682 INFO Monitor(max) STOP: 0.370435 !
2020-07-25 20:36:28,851 P50682 INFO Reduce learning rate on plateau: 0.000100
2020-07-25 20:36:28,851 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 20:36:28,896 P50682 INFO Train loss: 0.448227
2020-07-25 20:36:28,896 P50682 INFO ************ Epoch=6 end ************
2020-07-25 21:06:55,914 P50682 INFO [Metrics] logloss: 0.438810 - AUC: 0.813331
2020-07-25 21:06:55,915 P50682 INFO Save best model: monitor(max): 0.374521
2020-07-25 21:06:56,333 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 21:06:56,379 P50682 INFO Train loss: 0.435194
2020-07-25 21:06:56,379 P50682 INFO ************ Epoch=7 end ************
2020-07-25 21:37:41,430 P50682 INFO [Metrics] logloss: 0.439672 - AUC: 0.812808
2020-07-25 21:37:41,434 P50682 INFO Monitor(max) STOP: 0.373136 !
2020-07-25 21:37:41,435 P50682 INFO Reduce learning rate on plateau: 0.000010
2020-07-25 21:37:41,435 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 21:37:41,480 P50682 INFO Train loss: 0.429477
2020-07-25 21:37:41,480 P50682 INFO ************ Epoch=8 end ************
2020-07-25 22:08:11,597 P50682 INFO [Metrics] logloss: 0.441629 - AUC: 0.811652
2020-07-25 22:08:11,598 P50682 INFO Monitor(max) STOP: 0.370023 !
2020-07-25 22:08:11,598 P50682 INFO Reduce learning rate on plateau: 0.000001
2020-07-25 22:08:11,599 P50682 INFO Early stopping at epoch=9
2020-07-25 22:08:11,599 P50682 INFO --- 3668/3668 batches finished ---
2020-07-25 22:08:11,641 P50682 INFO Train loss: 0.423638
2020-07-25 22:08:11,642 P50682 INFO Training finished.
2020-07-25 22:08:11,642 P50682 INFO Load best model: /home/zhujieming/xxx/OpenCTR1030/benchmarks/Criteo/AFN_criteo/min10/criteo_x4_5c863b0f/AFN_criteo_x4_5c863b0f_004_81f459ad_model.ckpt
2020-07-25 22:08:11,934 P50682 INFO ****** Train/validation evaluation ******
2020-07-25 22:16:49,943 P50682 INFO [Metrics] logloss: 0.421408 - AUC: 0.831099
2020-07-25 22:17:51,316 P50682 INFO [Metrics] logloss: 0.438810 - AUC: 0.813331
2020-07-25 22:17:51,372 P50682 INFO ******** Test evaluation ********
2020-07-25 22:17:51,373 P50682 INFO Loading data...
2020-07-25 22:17:51,373 P50682 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-07-25 22:17:52,237 P50682 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-07-25 22:17:52,237 P50682 INFO Loading test data done.
2020-07-25 22:18:53,723 P50682 INFO [Metrics] logloss: 0.438404 - AUC: 0.813804

```
