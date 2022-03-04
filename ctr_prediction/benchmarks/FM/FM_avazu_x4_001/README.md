## FM_avazu_x4_001

A hands-on guide to run the FM model on the Avazu_x4_001 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.0
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Avazu_x4_001](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu/README.md#Avazu_x4_001). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FM_avazu_x4_tuner_config_02](./FM_avazu_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FM_avazu_x4_001
    nohup python run_expid.py --config ./FM_avazu_x4_tuner_config_02 --expid FM_avazu_x4_003_a62c6000 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.375362 | 0.788721  |


### Logs
```python
2020-06-13 23:47:14,598 P11780 INFO {
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FM",
    "model_id": "FM_avazu_x4_3bbbc4c9_003_e720ce1f",
    "model_root": "./Avazu/FM_avazu/min2/",
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
    "test_data": "../data/Avazu/avazu_x4_3bbbc4c9/test.h5",
    "train_data": "../data/Avazu/avazu_x4_3bbbc4c9/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_3bbbc4c9/valid.h5",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2020-06-13 23:47:14,600 P11780 INFO Set up feature encoder...
2020-06-13 23:47:14,600 P11780 INFO Load feature_map from json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2020-06-13 23:47:16,075 P11780 INFO Total number of parameters: 63766576.
2020-06-13 23:47:16,076 P11780 INFO Loading data...
2020-06-13 23:47:16,078 P11780 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2020-06-13 23:47:40,508 P11780 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2020-06-13 23:47:45,798 P11780 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-06-13 23:47:45,975 P11780 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-06-13 23:47:45,975 P11780 INFO Loading train data done.
2020-06-13 23:48:21,870 P11780 INFO Start training: 3235 batches/epoch
2020-06-13 23:48:21,870 P11780 INFO ************ Epoch=1 start ************
2020-06-14 00:05:01,032 P11780 INFO [Metrics] logloss: 0.382903 - AUC: 0.775315
2020-06-14 00:05:01,032 P11780 INFO Save best model: monitor(max): 0.392412
2020-06-14 00:05:01,271 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 00:05:01,370 P11780 INFO Train loss: 0.396598
2020-06-14 00:05:01,370 P11780 INFO ************ Epoch=1 end ************
2020-06-14 00:21:19,379 P11780 INFO [Metrics] logloss: 0.380042 - AUC: 0.780067
2020-06-14 00:21:19,390 P11780 INFO Save best model: monitor(max): 0.400025
2020-06-14 00:21:20,289 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 00:21:20,383 P11780 INFO Train loss: 0.387624
2020-06-14 00:21:20,384 P11780 INFO ************ Epoch=2 end ************
2020-06-14 00:37:59,925 P11780 INFO [Metrics] logloss: 0.378981 - AUC: 0.781997
2020-06-14 00:37:59,936 P11780 INFO Save best model: monitor(max): 0.403016
2020-06-14 00:38:01,696 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 00:38:01,782 P11780 INFO Train loss: 0.384500
2020-06-14 00:38:01,782 P11780 INFO ************ Epoch=3 end ************
2020-06-14 00:54:12,075 P11780 INFO [Metrics] logloss: 0.378432 - AUC: 0.782870
2020-06-14 00:54:12,079 P11780 INFO Save best model: monitor(max): 0.404437
2020-06-14 00:54:12,643 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 00:54:12,733 P11780 INFO Train loss: 0.382012
2020-06-14 00:54:12,733 P11780 INFO ************ Epoch=4 end ************
2020-06-14 01:10:46,302 P11780 INFO [Metrics] logloss: 0.377892 - AUC: 0.783985
2020-06-14 01:10:46,306 P11780 INFO Save best model: monitor(max): 0.406093
2020-06-14 01:10:46,845 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 01:10:46,945 P11780 INFO Train loss: 0.379902
2020-06-14 01:10:46,945 P11780 INFO ************ Epoch=5 end ************
2020-06-14 01:27:00,341 P11780 INFO [Metrics] logloss: 0.377842 - AUC: 0.784509
2020-06-14 01:27:00,343 P11780 INFO Save best model: monitor(max): 0.406667
2020-06-14 01:27:00,909 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 01:27:01,012 P11780 INFO Train loss: 0.378040
2020-06-14 01:27:01,012 P11780 INFO ************ Epoch=6 end ************
2020-06-14 01:43:35,130 P11780 INFO [Metrics] logloss: 0.378087 - AUC: 0.784157
2020-06-14 01:43:35,134 P11780 INFO Monitor(max) STOP: 0.406070 !
2020-06-14 01:43:35,134 P11780 INFO Reduce learning rate on plateau: 0.000100
2020-06-14 01:43:35,134 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 01:43:35,234 P11780 INFO Train loss: 0.376432
2020-06-14 01:43:35,234 P11780 INFO ************ Epoch=7 end ************
2020-06-14 01:59:49,633 P11780 INFO [Metrics] logloss: 0.375589 - AUC: 0.787979
2020-06-14 01:59:49,635 P11780 INFO Save best model: monitor(max): 0.412390
2020-06-14 01:59:50,174 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 01:59:50,276 P11780 INFO Train loss: 0.357965
2020-06-14 01:59:50,276 P11780 INFO ************ Epoch=8 end ************
2020-06-14 02:16:24,251 P11780 INFO [Metrics] logloss: 0.375398 - AUC: 0.788307
2020-06-14 02:16:24,255 P11780 INFO Save best model: monitor(max): 0.412909
2020-06-14 02:16:24,808 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 02:16:24,910 P11780 INFO Train loss: 0.355909
2020-06-14 02:16:24,910 P11780 INFO ************ Epoch=9 end ************
2020-06-14 02:32:39,329 P11780 INFO [Metrics] logloss: 0.375313 - AUC: 0.788470
2020-06-14 02:32:39,331 P11780 INFO Save best model: monitor(max): 0.413157
2020-06-14 02:32:39,901 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 02:32:40,005 P11780 INFO Train loss: 0.354790
2020-06-14 02:32:40,006 P11780 INFO ************ Epoch=10 end ************
2020-06-14 02:49:17,100 P11780 INFO [Metrics] logloss: 0.375339 - AUC: 0.788499
2020-06-14 02:49:17,104 P11780 INFO Save best model: monitor(max): 0.413160
2020-06-14 02:49:17,645 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 02:49:17,746 P11780 INFO Train loss: 0.353957
2020-06-14 02:49:17,746 P11780 INFO ************ Epoch=11 end ************
2020-06-14 03:05:33,048 P11780 INFO [Metrics] logloss: 0.375394 - AUC: 0.788577
2020-06-14 03:05:33,050 P11780 INFO Save best model: monitor(max): 0.413184
2020-06-14 03:05:33,602 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 03:05:33,704 P11780 INFO Train loss: 0.353248
2020-06-14 03:05:33,705 P11780 INFO ************ Epoch=12 end ************
2020-06-14 03:22:06,431 P11780 INFO [Metrics] logloss: 0.375335 - AUC: 0.788680
2020-06-14 03:22:06,434 P11780 INFO Save best model: monitor(max): 0.413345
2020-06-14 03:22:06,960 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 03:22:07,061 P11780 INFO Train loss: 0.352624
2020-06-14 03:22:07,061 P11780 INFO ************ Epoch=13 end ************
2020-06-14 03:38:21,012 P11780 INFO [Metrics] logloss: 0.375417 - AUC: 0.788631
2020-06-14 03:38:21,014 P11780 INFO Monitor(max) STOP: 0.413214 !
2020-06-14 03:38:21,015 P11780 INFO Reduce learning rate on plateau: 0.000010
2020-06-14 03:38:21,015 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 03:38:21,117 P11780 INFO Train loss: 0.352046
2020-06-14 03:38:21,117 P11780 INFO ************ Epoch=14 end ************
2020-06-14 03:54:55,454 P11780 INFO [Metrics] logloss: 0.375269 - AUC: 0.788825
2020-06-14 03:54:55,458 P11780 INFO Save best model: monitor(max): 0.413556
2020-06-14 03:54:56,020 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 03:54:56,130 P11780 INFO Train loss: 0.348690
2020-06-14 03:54:56,130 P11780 INFO ************ Epoch=15 end ************
2020-06-14 04:11:10,531 P11780 INFO [Metrics] logloss: 0.375269 - AUC: 0.788820
2020-06-14 04:11:10,534 P11780 INFO Monitor(max) STOP: 0.413551 !
2020-06-14 04:11:10,534 P11780 INFO Reduce learning rate on plateau: 0.000001
2020-06-14 04:11:10,534 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 04:11:10,642 P11780 INFO Train loss: 0.348598
2020-06-14 04:11:10,642 P11780 INFO ************ Epoch=16 end ************
2020-06-14 04:27:44,986 P11780 INFO [Metrics] logloss: 0.375271 - AUC: 0.788845
2020-06-14 04:27:44,989 P11780 INFO Save best model: monitor(max): 0.413573
2020-06-14 04:27:45,521 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 04:27:45,629 P11780 INFO Train loss: 0.348219
2020-06-14 04:27:45,629 P11780 INFO ************ Epoch=17 end ************
2020-06-14 04:44:01,174 P11780 INFO [Metrics] logloss: 0.375275 - AUC: 0.788843
2020-06-14 04:44:01,177 P11780 INFO Monitor(max) STOP: 0.413569 !
2020-06-14 04:44:01,177 P11780 INFO Reduce learning rate on plateau: 0.000001
2020-06-14 04:44:01,177 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 04:44:01,284 P11780 INFO Train loss: 0.348215
2020-06-14 04:44:01,284 P11780 INFO ************ Epoch=18 end ************
2020-06-14 05:00:38,181 P11780 INFO [Metrics] logloss: 0.375274 - AUC: 0.788847
2020-06-14 05:00:38,184 P11780 INFO Monitor(max) STOP: 0.413573 !
2020-06-14 05:00:38,184 P11780 INFO Reduce learning rate on plateau: 0.000001
2020-06-14 05:00:38,184 P11780 INFO Early stopping at epoch=19
2020-06-14 05:00:38,184 P11780 INFO --- 3235/3235 batches finished ---
2020-06-14 05:00:38,289 P11780 INFO Train loss: 0.348203
2020-06-14 05:00:38,289 P11780 INFO Training finished.
2020-06-14 05:00:38,289 P11780 INFO Load best model: /home/XXX/benchmarks/Avazu/FM_avazu/min2/avazu_x4_3bbbc4c9/FM_avazu_x4_3bbbc4c9_003_e720ce1f_model.ckpt
2020-06-14 05:00:38,743 P11780 INFO ****** Train/validation evaluation ******
2020-06-14 05:01:03,038 P11780 INFO [Metrics] logloss: 0.375271 - AUC: 0.788845
2020-06-14 05:01:03,079 P11780 INFO ******** Test evaluation ********
2020-06-14 05:01:03,079 P11780 INFO Loading data...
2020-06-14 05:01:03,079 P11780 INFO Loading data from h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2020-06-14 05:01:03,561 P11780 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-06-14 05:01:03,561 P11780 INFO Loading test data done.
2020-06-14 05:01:28,116 P11780 INFO [Metrics] logloss: 0.375362 - AUC: 0.788721

```
