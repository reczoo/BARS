## FFM_Criteo_x4_001

A notebook to benchmark FFM on Criteo_x4_001 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.
### Code
1. Install FuxiCTR
  
    Install FuxiCTR via `pip install fuxictr==1.0` to get all dependencies ready. Then download [the FuxiCTR repository](https://github.com/huawei-noah/benchmark/archive/53e314461c19dbc7f462b42bf0f0bfae020dc398.zip) to your local path.

2. Downalod the dataset and run [the preprocessing script](https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Criteo_x4/split_criteo_x4.py) for data splitting. 

3. Download the hyper-parameter configuration file: [FFM_criteo_x4_tuner_config_01.yaml](./FFM_criteo_x4_tuner_config_01.yaml)

4. Run the following script to reproduce the result. 
  + --config: The config file that defines the tuning space
  + --tag: Specify which expid to run (each expid corresponds to a specific setting of hyper-parameters in the tunner space)
  + --gpu: The available gpus for parameters tuning.

  ```bash
  cd FuxiCTR/benchmarks
  python run_param_tuner.py --config YOUR_PATH/FFM_criteo_x4_tuner_config_01.yaml --tag 010 --gpu 0
  ```


### Results
```python
[Metrics] logloss: 0.440745 - AUC: 0.811263
```


### Logs
```python
2020-06-26 23:58:44,499 P2560 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "4",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "3",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FFM",
    "model_id": "FFM_criteo_x4_5c863b0f_010_6f65737a",
    "model_root": "./Criteo/FFM_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-06",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-26 23:58:44,507 P2560 INFO Set up feature encoder...
2020-06-26 23:58:44,507 P2560 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-26 23:58:44,507 P2560 INFO Loading data...
2020-06-26 23:58:44,517 P2560 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-26 23:58:51,055 P2560 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-26 23:58:52,878 P2560 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-26 23:58:53,002 P2560 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-26 23:58:53,002 P2560 INFO Loading train data done.
2020-06-26 23:59:00,484 P2560 INFO Start training: 3668 batches/epoch
2020-06-26 23:59:00,484 P2560 INFO ************ Epoch=1 start ************
2020-06-27 02:15:48,526 P2560 INFO [Metrics] logloss: 0.443578 - AUC: 0.807967
2020-06-27 02:15:48,528 P2560 INFO Save best model: monitor(max): 0.364389
2020-06-27 02:15:49,335 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 02:15:49,383 P2560 INFO Train loss: 0.457038
2020-06-27 02:15:49,384 P2560 INFO ************ Epoch=1 end ************
2020-06-27 04:32:28,946 P2560 INFO [Metrics] logloss: 0.442052 - AUC: 0.809687
2020-06-27 04:32:28,948 P2560 INFO Save best model: monitor(max): 0.367634
2020-06-27 04:32:30,044 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 04:32:30,108 P2560 INFO Train loss: 0.450131
2020-06-27 04:32:30,108 P2560 INFO ************ Epoch=2 end ************
2020-06-27 06:48:38,069 P2560 INFO [Metrics] logloss: 0.441775 - AUC: 0.809998
2020-06-27 06:48:38,070 P2560 INFO Save best model: monitor(max): 0.368223
2020-06-27 06:48:39,250 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 06:48:39,308 P2560 INFO Train loss: 0.448668
2020-06-27 06:48:39,309 P2560 INFO ************ Epoch=3 end ************
2020-06-27 09:04:36,430 P2560 INFO [Metrics] logloss: 0.442046 - AUC: 0.809730
2020-06-27 09:04:36,431 P2560 INFO Monitor(max) STOP: 0.367684 !
2020-06-27 09:04:36,432 P2560 INFO Reduce learning rate on plateau: 0.000100
2020-06-27 09:04:36,432 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 09:04:36,495 P2560 INFO Train loss: 0.447682
2020-06-27 09:04:36,495 P2560 INFO ************ Epoch=4 end ************
2020-06-27 11:21:23,020 P2560 INFO [Metrics] logloss: 0.441177 - AUC: 0.810739
2020-06-27 11:21:23,027 P2560 INFO Save best model: monitor(max): 0.369561
2020-06-27 11:21:24,149 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 11:21:24,203 P2560 INFO Train loss: 0.432505
2020-06-27 11:21:24,204 P2560 INFO ************ Epoch=5 end ************
2020-06-27 13:38:04,763 P2560 INFO [Metrics] logloss: 0.441768 - AUC: 0.810203
2020-06-27 13:38:04,763 P2560 INFO Monitor(max) STOP: 0.368435 !
2020-06-27 13:38:04,764 P2560 INFO Reduce learning rate on plateau: 0.000010
2020-06-27 13:38:04,764 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 13:38:04,811 P2560 INFO Train loss: 0.428703
2020-06-27 13:38:04,812 P2560 INFO ************ Epoch=6 end ************
2020-06-27 15:54:21,631 P2560 INFO [Metrics] logloss: 0.441821 - AUC: 0.810172
2020-06-27 15:54:21,632 P2560 INFO Monitor(max) STOP: 0.368351 !
2020-06-27 15:54:21,632 P2560 INFO Reduce learning rate on plateau: 0.000001
2020-06-27 15:54:21,632 P2560 INFO Early stopping at epoch=7
2020-06-27 15:54:21,632 P2560 INFO --- 3668/3668 batches finished ---
2020-06-27 15:54:21,691 P2560 INFO Train loss: 0.424549
2020-06-27 15:54:21,691 P2560 INFO Training finished.
2020-06-27 15:54:21,691 P2560 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/FFM_criteo/min10/criteo_x4_5c863b0f/FFM_criteo_x4_5c863b0f_010_6f65737a_model.ckpt
2020-06-27 15:54:23,795 P2560 INFO ****** Train/validation evaluation ******
2020-06-27 16:08:12,533 P2560 INFO [Metrics] logloss: 0.416780 - AUC: 0.836701
2020-06-27 16:09:52,183 P2560 INFO [Metrics] logloss: 0.441177 - AUC: 0.810739
2020-06-27 16:09:52,250 P2560 INFO ******** Test evaluation ********
2020-06-27 16:09:52,250 P2560 INFO Loading data...
2020-06-27 16:09:52,250 P2560 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-27 16:09:53,287 P2560 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-27 16:09:53,287 P2560 INFO Loading test data done.
2020-06-27 16:11:33,352 P2560 INFO [Metrics] logloss: 0.440745 - AUC: 0.811263

```