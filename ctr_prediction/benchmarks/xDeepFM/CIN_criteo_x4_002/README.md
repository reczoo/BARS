## CIN_criteo_x4_002

A hands-on guide to run the xDeepFM model on the Criteo_x4_002 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 10.2
  python: 3.6.4
  pytorch: 1.0.0
  pandas: 0.22.0
  numpy: 1.19.2
  scipy: 1.5.4
  sklearn: 0.22.1
  pyyaml: 5.4.1
  h5py: 2.8.0
  tqdm: 4.60.0
  fuxictr: 1.0.2
  ```

### Dataset
Dataset ID: [Criteo_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_criteo_x4_tuner_config_11](./CIN_criteo_x4_tuner_config_11). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_criteo_x4_002
    nohup python run_expid.py --config ./CIN_criteo_x4_tuner_config_11 --expid xDeepFM_criteo_x4_003_444046bd --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438674 | 0.813314  |


### Logs
```python
2020-06-03 23:36:45,486 P1294 INFO {
    "batch_norm": "False",
    "batch_size": "2000",
    "cin_layer_units": "[39, 39, 39, 39]",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "hidden_activations": "relu",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "xDeepFM",
    "model_id": "xDeepFM_criteo_x4_003_38a8a0a3",
    "model_root": "./Criteo/xDeepFM_criteo/",
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
    "test_data": "../data/Criteo/criteo_x4_001_be98441d/test.h5",
    "train_data": "../data/Criteo/criteo_x4_001_be98441d/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_001_be98441d/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-03 23:36:45,487 P1294 INFO Set up feature encoder...
2020-06-03 23:36:45,487 P1294 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-06-03 23:36:45,488 P1294 INFO Loading data...
2020-06-03 23:36:45,489 P1294 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-06-03 23:36:51,175 P1294 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-06-03 23:36:53,410 P1294 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-03 23:36:53,625 P1294 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-03 23:36:53,625 P1294 INFO Loading train data done.
2020-06-03 23:37:02,206 P1294 INFO **** Start training: 18337 batches/epoch ****
2020-06-04 00:29:57,956 P1294 INFO [Metrics] logloss: 0.448106 - AUC: 0.802993
2020-06-04 00:29:57,958 P1294 INFO Save best model: monitor(max): 0.354887
2020-06-04 00:29:58,974 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 00:29:59,033 P1294 INFO Train loss: 0.468401
2020-06-04 00:29:59,033 P1294 INFO ************ Epoch=1 end ************
2020-06-04 01:22:49,850 P1294 INFO [Metrics] logloss: 0.446655 - AUC: 0.804574
2020-06-04 01:22:49,851 P1294 INFO Save best model: monitor(max): 0.357919
2020-06-04 01:22:51,148 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 01:22:51,207 P1294 INFO Train loss: 0.464334
2020-06-04 01:22:51,207 P1294 INFO ************ Epoch=2 end ************
2020-06-04 02:15:45,664 P1294 INFO [Metrics] logloss: 0.446156 - AUC: 0.805324
2020-06-04 02:15:45,665 P1294 INFO Save best model: monitor(max): 0.359168
2020-06-04 02:15:46,971 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 02:15:47,076 P1294 INFO Train loss: 0.463295
2020-06-04 02:15:47,077 P1294 INFO ************ Epoch=3 end ************
2020-06-04 03:08:36,444 P1294 INFO [Metrics] logloss: 0.445369 - AUC: 0.805815
2020-06-04 03:08:36,445 P1294 INFO Save best model: monitor(max): 0.360446
2020-06-04 03:08:38,261 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 03:08:38,329 P1294 INFO Train loss: 0.462649
2020-06-04 03:08:38,329 P1294 INFO ************ Epoch=4 end ************
2020-06-04 04:01:24,455 P1294 INFO [Metrics] logloss: 0.445177 - AUC: 0.806047
2020-06-04 04:01:24,456 P1294 INFO Save best model: monitor(max): 0.360870
2020-06-04 04:01:26,110 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 04:01:26,170 P1294 INFO Train loss: 0.462189
2020-06-04 04:01:26,170 P1294 INFO ************ Epoch=5 end ************
2020-06-04 04:54:18,897 P1294 INFO [Metrics] logloss: 0.445046 - AUC: 0.806202
2020-06-04 04:54:18,897 P1294 INFO Save best model: monitor(max): 0.361155
2020-06-04 04:54:20,914 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 04:54:20,973 P1294 INFO Train loss: 0.461841
2020-06-04 04:54:20,973 P1294 INFO ************ Epoch=6 end ************
2020-06-04 05:47:16,558 P1294 INFO [Metrics] logloss: 0.444826 - AUC: 0.806579
2020-06-04 05:47:16,559 P1294 INFO Save best model: monitor(max): 0.361753
2020-06-04 05:47:18,510 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 05:47:18,569 P1294 INFO Train loss: 0.461623
2020-06-04 05:47:18,570 P1294 INFO ************ Epoch=7 end ************
2020-06-04 06:40:10,078 P1294 INFO [Metrics] logloss: 0.444908 - AUC: 0.806497
2020-06-04 06:40:10,079 P1294 INFO Monitor(max) STOP: 0.361589 !
2020-06-04 06:40:10,079 P1294 INFO Reduce learning rate on plateau: 0.000100
2020-06-04 06:40:10,079 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 06:40:10,138 P1294 INFO Train loss: 0.461471
2020-06-04 06:40:10,138 P1294 INFO ************ Epoch=8 end ************
2020-06-04 07:33:02,340 P1294 INFO [Metrics] logloss: 0.439965 - AUC: 0.811777
2020-06-04 07:33:02,341 P1294 INFO Save best model: monitor(max): 0.371812
2020-06-04 07:33:04,220 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 07:33:04,281 P1294 INFO Train loss: 0.445687
2020-06-04 07:33:04,282 P1294 INFO ************ Epoch=9 end ************
2020-06-04 08:25:59,226 P1294 INFO [Metrics] logloss: 0.439368 - AUC: 0.812456
2020-06-04 08:25:59,227 P1294 INFO Save best model: monitor(max): 0.373089
2020-06-04 08:26:01,265 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 08:26:01,325 P1294 INFO Train loss: 0.441604
2020-06-04 08:26:01,325 P1294 INFO ************ Epoch=10 end ************
2020-06-04 09:18:45,404 P1294 INFO [Metrics] logloss: 0.439074 - AUC: 0.812778
2020-06-04 09:18:45,405 P1294 INFO Save best model: monitor(max): 0.373705
2020-06-04 09:18:46,981 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 09:18:47,042 P1294 INFO Train loss: 0.440186
2020-06-04 09:18:47,042 P1294 INFO ************ Epoch=11 end ************
2020-06-04 10:11:05,321 P1294 INFO [Metrics] logloss: 0.439067 - AUC: 0.812865
2020-06-04 10:11:05,322 P1294 INFO Save best model: monitor(max): 0.373798
2020-06-04 10:11:06,609 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 10:11:06,667 P1294 INFO Train loss: 0.439189
2020-06-04 10:11:06,668 P1294 INFO ************ Epoch=12 end ************
2020-06-04 11:03:37,389 P1294 INFO [Metrics] logloss: 0.439261 - AUC: 0.812627
2020-06-04 11:03:37,390 P1294 INFO Monitor(max) STOP: 0.373366 !
2020-06-04 11:03:37,390 P1294 INFO Reduce learning rate on plateau: 0.000010
2020-06-04 11:03:37,390 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 11:03:37,449 P1294 INFO Train loss: 0.438346
2020-06-04 11:03:37,449 P1294 INFO ************ Epoch=13 end ************
2020-06-04 11:56:35,621 P1294 INFO [Metrics] logloss: 0.440481 - AUC: 0.811797
2020-06-04 11:56:35,622 P1294 INFO Monitor(max) STOP: 0.371315 !
2020-06-04 11:56:35,622 P1294 INFO Reduce learning rate on plateau: 0.000001
2020-06-04 11:56:35,622 P1294 INFO Early stopping at epoch=14
2020-06-04 11:56:35,623 P1294 INFO --- 18337/18337 batches finished ---
2020-06-04 11:56:35,687 P1294 INFO Train loss: 0.431586
2020-06-04 11:56:35,687 P1294 INFO Training finished.
2020-06-04 11:56:35,687 P1294 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/xDeepFM_criteo/criteo_x4_001_be98441d/xDeepFM_criteo_x4_003_38a8a0a3_model.ckpt
2020-06-04 11:56:36,907 P1294 INFO ****** Train/validation evaluation ******
2020-06-04 12:04:01,174 P1294 INFO [Metrics] logloss: 0.426443 - AUC: 0.825969
2020-06-04 12:04:56,096 P1294 INFO [Metrics] logloss: 0.439067 - AUC: 0.812865
2020-06-04 12:04:56,218 P1294 INFO ******** Test evaluation ********
2020-06-04 12:04:56,218 P1294 INFO Loading data...
2020-06-04 12:04:56,218 P1294 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-06-04 12:04:56,918 P1294 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-04 12:04:56,919 P1294 INFO Loading test data done.
2020-06-04 12:05:53,152 P1294 INFO [Metrics] logloss: 0.438674 - AUC: 0.813314

```
