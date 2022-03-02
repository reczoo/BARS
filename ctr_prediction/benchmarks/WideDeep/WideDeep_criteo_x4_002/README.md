## WideDeep_criteo_x4_002

A hands-on guide to run the WideDeep model on the Criteo_x4_002 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [WideDeep](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_criteo_x4_tuner_config_02](./WideDeep_criteo_x4_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_criteo_x4_002
    nohup python run_expid.py --config ./WideDeep_criteo_x4_tuner_config_02 --expid WideDeep_criteo_x4_007_dbcfd11c --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.438892 | 0.812913  |


### Logs
```python
2020-03-04 12:13:56,397 P1308 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_001_be98441d",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "l2(1.e-6)",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "2",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000, 1000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "WideDeep",
    "model_id": "WideDeep_criteo_x4_007_5305f0a1",
    "model_root": "./Criteo/WideDeep_criteo/",
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
2020-03-04 12:13:56,399 P1308 INFO Set up feature encoder...
2020-03-04 12:13:56,400 P1308 INFO Load feature_map from json: ../data/Criteo/criteo_x4_001_be98441d/feature_map.json
2020-03-04 12:13:56,400 P1308 INFO Loading data...
2020-03-04 12:13:56,422 P1308 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/train.h5
2020-03-04 12:14:03,056 P1308 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/valid.h5
2020-03-04 12:14:04,852 P1308 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-03-04 12:14:04,975 P1308 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-04 12:14:04,975 P1308 INFO Loading train data done.
2020-03-04 12:14:16,494 P1308 INFO **** Start training: 3668 batches/epoch ****
2020-03-04 12:25:02,021 P1308 INFO [Metrics] logloss: 0.443056 - AUC: 0.808503
2020-03-04 12:25:02,084 P1308 INFO Save best model: monitor(max): 0.365447
2020-03-04 12:25:03,715 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:25:03,792 P1308 INFO Train loss: 0.456710
2020-03-04 12:25:03,792 P1308 INFO ************ Epoch=1 end ************
2020-03-04 12:35:40,277 P1308 INFO [Metrics] logloss: 0.440805 - AUC: 0.811012
2020-03-04 12:35:40,337 P1308 INFO Save best model: monitor(max): 0.370207
2020-03-04 12:35:42,167 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:35:42,251 P1308 INFO Train loss: 0.449778
2020-03-04 12:35:42,251 P1308 INFO ************ Epoch=2 end ************
2020-03-04 12:46:18,349 P1308 INFO [Metrics] logloss: 0.439688 - AUC: 0.812136
2020-03-04 12:46:18,418 P1308 INFO Save best model: monitor(max): 0.372448
2020-03-04 12:46:20,265 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:46:20,335 P1308 INFO Train loss: 0.447703
2020-03-04 12:46:20,335 P1308 INFO ************ Epoch=3 end ************
2020-03-04 12:56:57,083 P1308 INFO [Metrics] logloss: 0.439318 - AUC: 0.812417
2020-03-04 12:56:57,145 P1308 INFO Save best model: monitor(max): 0.373100
2020-03-04 12:56:58,993 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 12:56:59,060 P1308 INFO Train loss: 0.446406
2020-03-04 12:56:59,060 P1308 INFO ************ Epoch=4 end ************
2020-03-04 13:07:37,644 P1308 INFO [Metrics] logloss: 0.440403 - AUC: 0.811830
2020-03-04 13:07:37,715 P1308 INFO Monitor(max) STOP: 0.371428 !
2020-03-04 13:07:37,715 P1308 INFO Reduce learning rate on plateau: 0.000100
2020-03-04 13:07:37,715 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 13:07:37,795 P1308 INFO Train loss: 0.445365
2020-03-04 13:07:37,795 P1308 INFO ************ Epoch=5 end ************
2020-03-04 13:18:13,294 P1308 INFO [Metrics] logloss: 0.450589 - AUC: 0.805841
2020-03-04 13:18:13,357 P1308 INFO Monitor(max) STOP: 0.355253 !
2020-03-04 13:18:13,357 P1308 INFO Reduce learning rate on plateau: 0.000010
2020-03-04 13:18:13,357 P1308 INFO Early stopping at epoch=6
2020-03-04 13:18:13,357 P1308 INFO --- 3668/3668 batches finished ---
2020-03-04 13:18:13,426 P1308 INFO Train loss: 0.417406
2020-03-04 13:18:13,426 P1308 INFO Training finished.
2020-03-04 13:18:13,426 P1308 INFO Load best model: /cache/XXX/FuxiCTR/benchmarks/Criteo/WideDeep_criteo/criteo_x4_001_be98441d/WideDeep_criteo_x4_007_5305f0a1_criteo_x4_001_be98441d_model.ckpt
2020-03-04 13:18:15,161 P1308 INFO ****** Train/validation evaluation ******
2020-03-04 13:23:26,752 P1308 INFO [Metrics] logloss: 0.419569 - AUC: 0.833701
2020-03-04 13:24:02,752 P1308 INFO [Metrics] logloss: 0.439318 - AUC: 0.812417
2020-03-04 13:24:02,949 P1308 INFO ******** Test evaluation ********
2020-03-04 13:24:02,949 P1308 INFO Loading data...
2020-03-04 13:24:02,949 P1308 INFO Loading data from h5: ../data/Criteo/criteo_x4_001_be98441d/test.h5
2020-03-04 13:24:04,114 P1308 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-03-04 13:24:04,115 P1308 INFO Loading test data done.
2020-03-04 13:24:36,674 P1308 INFO [Metrics] logloss: 0.438892 - AUC: 0.812913

```
