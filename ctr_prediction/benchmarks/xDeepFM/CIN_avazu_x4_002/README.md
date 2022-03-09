## CIN_avazu_x4_002

A hands-on guide to run the xDeepFM model on the Avazu_x4_002 dataset.

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
Dataset ID: [Avazu_x4_002](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Avazu#Avazu_x4_002). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_avazu_x4_tuner_config_04](./CIN_avazu_x4_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_avazu_x4_002
    nohup python run_expid.py --config ./CIN_avazu_x4_tuner_config_04 --expid xDeepFM_avazu_x4_009_ce557b77 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.372362 | 0.792890  |


### Logs
```python
2022-03-12 00:12:56,980 P58410 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "cin_layer_units": "[250, 250, 250, 250]",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_d102865a",
    "debug": "False",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "5",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_avazu_x4_009_ce557b77",
    "model_root": "./Avazu/xDeepFM_avazu_x4_002/",
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
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2022-03-12 00:12:56,980 P58410 INFO Set up feature encoder...
2022-03-12 00:12:56,980 P58410 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x4_d102865a/feature_encoder.pkl
2022-03-12 00:13:06,539 P58410 INFO Total number of parameters: 347956016.
2022-03-12 00:13:06,539 P58410 INFO Loading data...
2022-03-12 00:13:06,540 P58410 INFO Loading data from h5: ../data/Avazu/avazu_x4_d102865a/train.h5
2022-03-12 00:13:09,216 P58410 INFO Loading data from h5: ../data/Avazu/avazu_x4_d102865a/valid.h5
2022-03-12 00:13:10,346 P58410 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-12 00:13:10,446 P58410 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-12 00:13:10,446 P58410 INFO Loading train data done.
2022-03-12 00:13:13,593 P58410 INFO Start training: 6469 batches/epoch
2022-03-12 00:13:13,593 P58410 INFO ************ Epoch=1 start ************
2022-03-12 01:48:00,938 P58410 INFO [Metrics] logloss: 0.372461 - AUC: 0.792684
2022-03-12 01:48:00,942 P58410 INFO Save best model: monitor(max): 0.420222
2022-03-12 01:48:02,561 P58410 INFO --- 6469/6469 batches finished ---
2022-03-12 01:48:02,622 P58410 INFO Train loss: 0.382451
2022-03-12 01:48:02,622 P58410 INFO ************ Epoch=1 end ************
2022-03-12 03:22:46,607 P58410 INFO [Metrics] logloss: 0.389312 - AUC: 0.782594
2022-03-12 03:22:46,610 P58410 INFO Monitor(max) STOP: 0.393282 !
2022-03-12 03:22:46,610 P58410 INFO Reduce learning rate on plateau: 0.000100
2022-03-12 03:22:46,610 P58410 INFO --- 6469/6469 batches finished ---
2022-03-12 03:22:46,662 P58410 INFO Train loss: 0.314493
2022-03-12 03:22:46,662 P58410 INFO ************ Epoch=2 end ************
2022-03-12 04:57:32,329 P58410 INFO [Metrics] logloss: 0.444651 - AUC: 0.762984
2022-03-12 04:57:32,332 P58410 INFO Monitor(max) STOP: 0.318334 !
2022-03-12 04:57:32,332 P58410 INFO Reduce learning rate on plateau: 0.000010
2022-03-12 04:57:32,332 P58410 INFO Early stopping at epoch=3
2022-03-12 04:57:32,332 P58410 INFO --- 6469/6469 batches finished ---
2022-03-12 04:57:32,383 P58410 INFO Train loss: 0.246544
2022-03-12 04:57:32,383 P58410 INFO Training finished.
2022-03-12 04:57:32,383 P58410 INFO Load best model: /cache/FuxiCTR/benchmarks/Avazu/xDeepFM_avazu_x4_002/avazu_x4_d102865a/xDeepFM_avazu_x4_009_ce557b77_model.ckpt
2022-03-12 04:57:34,283 P58410 INFO ****** Validation evaluation ******
2022-03-12 05:00:03,814 P58410 INFO [Metrics] logloss: 0.372461 - AUC: 0.792684
2022-03-12 05:00:03,884 P58410 INFO ******** Test evaluation ********
2022-03-12 05:00:03,885 P58410 INFO Loading data...
2022-03-12 05:00:03,885 P58410 INFO Loading data from h5: ../data/Avazu/avazu_x4_d102865a/test.h5
2022-03-12 05:00:04,437 P58410 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-12 05:00:04,438 P58410 INFO Loading test data done.
2022-03-12 05:02:34,331 P58410 INFO [Metrics] logloss: 0.372362 - AUC: 0.792890

```
