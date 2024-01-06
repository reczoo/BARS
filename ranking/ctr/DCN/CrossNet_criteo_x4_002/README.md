## CrossNet_criteo_x4_002

A hands-on guide to run the DCN model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DCN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_criteo_x4_tuner_config_12](./CrossNet_criteo_x4_tuner_config_12). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_criteo_x4_002
    nohup python run_expid.py --config ./CrossNet_criteo_x4_tuner_config_12 --expid DCN_criteo_x4_002_eeeaa5c1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.446832 | 0.804716  |


### Logs
```python
2020-06-04 09:50:27,026 P555 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "8",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "1",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DCN",
    "model_id": "DCN_criteo_x4_002_143b1f51",
    "model_root": "./Criteo/DCN_criteo/",
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
2020-06-04 09:50:27,029 P555 INFO Set up feature encoder...
2020-06-04 09:50:27,029 P555 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-06-04 09:50:27,029 P555 INFO Loading data...
2020-06-04 09:50:27,037 P555 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-06-04 09:50:32,211 P555 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-06-04 09:50:34,375 P555 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-04 09:50:34,529 P555 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-04 09:50:34,529 P555 INFO Loading train data done.
2020-06-04 09:50:48,028 P555 INFO **** Start training: 3668 batches/epoch ****
2020-06-04 09:58:50,688 P555 INFO [Metrics] logloss: 0.456432 - AUC: 0.793701
2020-06-04 09:58:50,695 P555 INFO Save best model: monitor(max): 0.337269
2020-06-04 09:58:52,227 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 09:58:52,306 P555 INFO Train loss: 0.470526
2020-06-04 09:58:52,306 P555 INFO ************ Epoch=1 end ************
2020-06-04 10:06:54,212 P555 INFO [Metrics] logloss: 0.453023 - AUC: 0.797493
2020-06-04 10:06:54,213 P555 INFO Save best model: monitor(max): 0.344470
2020-06-04 10:06:56,071 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 10:06:56,140 P555 INFO Train loss: 0.464637
2020-06-04 10:06:56,140 P555 INFO ************ Epoch=2 end ************
2020-06-04 10:14:52,564 P555 INFO [Metrics] logloss: 0.451532 - AUC: 0.799077
2020-06-04 10:14:52,565 P555 INFO Save best model: monitor(max): 0.347545
2020-06-04 10:14:54,342 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 10:14:54,418 P555 INFO Train loss: 0.462549
2020-06-04 10:14:54,418 P555 INFO ************ Epoch=3 end ************
2020-06-04 10:22:49,295 P555 INFO [Metrics] logloss: 0.450746 - AUC: 0.800161
2020-06-04 10:22:49,297 P555 INFO Save best model: monitor(max): 0.349416
2020-06-04 10:22:51,098 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 10:22:51,169 P555 INFO Train loss: 0.461422
2020-06-04 10:22:51,169 P555 INFO ************ Epoch=4 end ************
2020-06-04 10:30:50,634 P555 INFO [Metrics] logloss: 0.450402 - AUC: 0.800413
2020-06-04 10:30:50,635 P555 INFO Save best model: monitor(max): 0.350011
2020-06-04 10:30:51,756 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 10:30:51,832 P555 INFO Train loss: 0.460928
2020-06-04 10:30:51,832 P555 INFO ************ Epoch=5 end ************
2020-06-04 10:38:49,072 P555 INFO [Metrics] logloss: 0.450289 - AUC: 0.800637
2020-06-04 10:38:49,074 P555 INFO Save best model: monitor(max): 0.350348
2020-06-04 10:38:50,812 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 10:38:50,889 P555 INFO Train loss: 0.460677
2020-06-04 10:38:50,889 P555 INFO ************ Epoch=6 end ************
2020-06-04 10:46:50,525 P555 INFO [Metrics] logloss: 0.449965 - AUC: 0.800864
2020-06-04 10:46:50,527 P555 INFO Save best model: monitor(max): 0.350899
2020-06-04 10:46:52,411 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 10:46:52,476 P555 INFO Train loss: 0.460475
2020-06-04 10:46:52,476 P555 INFO ************ Epoch=7 end ************
2020-06-04 10:54:53,213 P555 INFO [Metrics] logloss: 0.449864 - AUC: 0.801093
2020-06-04 10:54:53,214 P555 INFO Save best model: monitor(max): 0.351228
2020-06-04 10:54:55,043 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 10:54:55,104 P555 INFO Train loss: 0.460244
2020-06-04 10:54:55,104 P555 INFO ************ Epoch=8 end ************
2020-06-04 11:02:54,287 P555 INFO [Metrics] logloss: 0.449659 - AUC: 0.801341
2020-06-04 11:02:54,289 P555 INFO Save best model: monitor(max): 0.351682
2020-06-04 11:02:56,125 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 11:02:56,199 P555 INFO Train loss: 0.460026
2020-06-04 11:02:56,199 P555 INFO ************ Epoch=9 end ************
2020-06-04 11:10:56,139 P555 INFO [Metrics] logloss: 0.449849 - AUC: 0.801328
2020-06-04 11:10:56,141 P555 INFO Monitor(max) STOP: 0.351479 !
2020-06-04 11:10:56,141 P555 INFO Reduce learning rate on plateau: 0.000100
2020-06-04 11:10:56,141 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 11:10:56,201 P555 INFO Train loss: 0.459995
2020-06-04 11:10:56,201 P555 INFO ************ Epoch=10 end ************
2020-06-04 11:18:58,510 P555 INFO [Metrics] logloss: 0.447341 - AUC: 0.803944
2020-06-04 11:18:58,512 P555 INFO Save best model: monitor(max): 0.356603
2020-06-04 11:19:00,361 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 11:19:00,418 P555 INFO Train loss: 0.447941
2020-06-04 11:19:00,418 P555 INFO ************ Epoch=11 end ************
2020-06-04 11:27:01,213 P555 INFO [Metrics] logloss: 0.447224 - AUC: 0.804234
2020-06-04 11:27:01,214 P555 INFO Save best model: monitor(max): 0.357010
2020-06-04 11:27:03,068 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 11:27:03,126 P555 INFO Train loss: 0.444455
2020-06-04 11:27:03,127 P555 INFO ************ Epoch=12 end ************
2020-06-04 11:35:03,144 P555 INFO [Metrics] logloss: 0.447452 - AUC: 0.804072
2020-06-04 11:35:03,151 P555 INFO Monitor(max) STOP: 0.356620 !
2020-06-04 11:35:03,151 P555 INFO Reduce learning rate on plateau: 0.000010
2020-06-04 11:35:03,151 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 11:35:03,217 P555 INFO Train loss: 0.442539
2020-06-04 11:35:03,217 P555 INFO ************ Epoch=13 end ************
2020-06-04 11:43:01,882 P555 INFO [Metrics] logloss: 0.448063 - AUC: 0.803692
2020-06-04 11:43:01,883 P555 INFO Monitor(max) STOP: 0.355629 !
2020-06-04 11:43:01,883 P555 INFO Reduce learning rate on plateau: 0.000001
2020-06-04 11:43:01,883 P555 INFO Early stopping at epoch=14
2020-06-04 11:43:01,884 P555 INFO --- 3668/3668 batches finished ---
2020-06-04 11:43:01,953 P555 INFO Train loss: 0.438149
2020-06-04 11:43:01,953 P555 INFO Training finished.
2020-06-04 11:43:01,954 P555 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/DCN_criteo/criteo_x4_001_be98441d/DCN_criteo_x4_002_143b1f51_model.ckpt
2020-06-04 11:43:03,444 P555 INFO ****** Train/validation evaluation ******
2020-06-04 11:47:39,995 P555 INFO [Metrics] logloss: 0.435052 - AUC: 0.817433
2020-06-04 11:48:10,847 P555 INFO [Metrics] logloss: 0.447224 - AUC: 0.804234
2020-06-04 11:48:10,978 P555 INFO ******** Test evaluation ********
2020-06-04 11:48:10,978 P555 INFO Loading data...
2020-06-04 11:48:10,979 P555 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-06-04 11:48:11,943 P555 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-04 11:48:11,943 P555 INFO Loading test data done.
2020-06-04 11:48:40,947 P555 INFO [Metrics] logloss: 0.446832 - AUC: 0.804716

```
