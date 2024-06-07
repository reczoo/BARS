## CrossNet_avazu_x4_002

A hands-on guide to run the DCN model on the Avazu_x4_002 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Avazu_x4](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x4). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [DCN](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/DCN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CrossNet_avazu_x4_tuner_config_01](./CrossNet_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CrossNet_avazu_x4_002
    nohup python run_expid.py --config ./CrossNet_avazu_x4_tuner_config_01 --expid DCN_avazu_x4_006_32b15b95 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.377348 | 0.783976  |


### Logs
```python
2022-03-01 18:05:22,613 P38299 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "crossing_layers": "6",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_d102865a",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_avazu_x4_006_32b15b95",
    "model_root": "./Avazu/DCN_avazu_x4_002/",
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
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-01 18:05:22,614 P38299 INFO Set up feature encoder...
2022-03-01 18:05:22,614 P38299 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-01 18:07:28,906 P38299 INFO Preprocess feature columns...
2022-03-01 18:13:19,451 P38299 INFO Fit feature encoder...
2022-03-01 18:13:19,452 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}
2022-03-01 18:15:53,184 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C1', 'type': 'categorical'}
2022-03-01 18:15:58,135 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'banner_pos', 'type': 'categorical'}
2022-03-01 18:16:02,466 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_id', 'type': 'categorical'}
2022-03-01 18:16:07,496 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_domain', 'type': 'categorical'}
2022-03-01 18:16:12,634 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_category', 'type': 'categorical'}
2022-03-01 18:16:17,470 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_id', 'type': 'categorical'}
2022-03-01 18:16:22,727 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_domain', 'type': 'categorical'}
2022-03-01 18:16:27,904 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_category', 'type': 'categorical'}
2022-03-01 18:16:33,133 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_id', 'type': 'categorical'}
2022-03-01 18:16:41,593 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_ip', 'type': 'categorical'}
2022-03-01 18:17:03,112 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_model', 'type': 'categorical'}
2022-03-01 18:17:08,995 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_type', 'type': 'categorical'}
2022-03-01 18:17:13,461 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_conn_type', 'type': 'categorical'}
2022-03-01 18:17:18,029 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C14', 'type': 'categorical'}
2022-03-01 18:17:23,464 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C15', 'type': 'categorical'}
2022-03-01 18:17:28,428 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C16', 'type': 'categorical'}
2022-03-01 18:17:33,302 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C17', 'type': 'categorical'}
2022-03-01 18:17:38,487 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C18', 'type': 'categorical'}
2022-03-01 18:17:43,068 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C19', 'type': 'categorical'}
2022-03-01 18:17:48,214 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C20', 'type': 'categorical'}
2022-03-01 18:17:53,529 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C21', 'type': 'categorical'}
2022-03-01 18:17:58,710 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}
2022-03-01 18:20:45,026 P38299 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}
2022-03-01 18:23:36,678 P38299 INFO Set feature index...
2022-03-01 18:23:36,678 P38299 INFO Pickle feature_encode: ../data/Avazu/avazu_x4_d102865a/feature_encoder.pkl
2022-03-01 18:23:41,694 P38299 INFO Save feature_map to json: ../data/Avazu/avazu_x4_d102865a/feature_map.json
2022-03-01 18:23:41,695 P38299 INFO Set feature encoder done.
2022-03-01 18:23:53,545 P38299 INFO Total number of parameters: 334949081.
2022-03-01 18:23:53,546 P38299 INFO Loading data...
2022-03-01 18:23:53,549 P38299 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-01 18:25:58,305 P38299 INFO Preprocess feature columns...
2022-03-01 18:31:54,104 P38299 INFO Transform feature columns...
2022-03-01 18:36:31,662 P38299 INFO Saving data to h5: ../data/Avazu/avazu_x4_d102865a/train.h5
2022-03-01 18:36:40,683 P38299 INFO Reading file: ../data/Avazu/Avazu_x4/valid.csv
2022-03-01 18:36:56,256 P38299 INFO Preprocess feature columns...
2022-03-01 18:37:38,926 P38299 INFO Transform feature columns...
2022-03-01 18:38:13,212 P38299 INFO Saving data to h5: ../data/Avazu/avazu_x4_d102865a/valid.h5
2022-03-01 18:38:16,408 P38299 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-01 18:38:16,523 P38299 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-01 18:38:16,523 P38299 INFO Loading train data done.
2022-03-01 18:38:19,479 P38299 INFO Start training: 3235 batches/epoch
2022-03-01 18:38:19,479 P38299 INFO ************ Epoch=1 start ************
2022-03-01 18:46:44,131 P38299 INFO [Metrics] logloss: 0.377490 - AUC: 0.783711
2022-03-01 18:46:44,132 P38299 INFO Save best model: monitor(max): 0.406221
2022-03-01 18:46:45,594 P38299 INFO --- 3235/3235 batches finished ---
2022-03-01 18:46:45,924 P38299 INFO Train loss: 0.385861
2022-03-01 18:46:45,924 P38299 INFO ************ Epoch=1 end ************
2022-03-01 18:55:10,221 P38299 INFO [Metrics] logloss: 0.396565 - AUC: 0.772951
2022-03-01 18:55:10,221 P38299 INFO Monitor(max) STOP: 0.376386 !
2022-03-01 18:55:10,221 P38299 INFO Reduce learning rate on plateau: 0.000100
2022-03-01 18:55:10,222 P38299 INFO --- 3235/3235 batches finished ---
2022-03-01 18:55:10,548 P38299 INFO Train loss: 0.326940
2022-03-01 18:55:10,548 P38299 INFO ************ Epoch=2 end ************
2022-03-01 19:03:33,937 P38299 INFO [Metrics] logloss: 0.443104 - AUC: 0.757553
2022-03-01 19:03:33,938 P38299 INFO Monitor(max) STOP: 0.314449 !
2022-03-01 19:03:33,938 P38299 INFO Reduce learning rate on plateau: 0.000010
2022-03-01 19:03:33,938 P38299 INFO Early stopping at epoch=3
2022-03-01 19:03:33,938 P38299 INFO --- 3235/3235 batches finished ---
2022-03-01 19:03:34,255 P38299 INFO Train loss: 0.276877
2022-03-01 19:03:34,255 P38299 INFO Training finished.
2022-03-01 19:03:34,256 P38299 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/DCN_avazu_x4_002/avazu_x4_d102865a/DCN_avazu_x4_006_32b15b95_model.ckpt
2022-03-01 19:03:36,220 P38299 INFO ****** Validation evaluation ******
2022-03-01 19:04:01,553 P38299 INFO [Metrics] logloss: 0.377490 - AUC: 0.783711
2022-03-01 19:04:01,889 P38299 INFO ******** Test evaluation ********
2022-03-01 19:04:01,890 P38299 INFO Loading data...
2022-03-01 19:04:01,890 P38299 INFO Reading file: ../data/Avazu/Avazu_x4/test.csv
2022-03-01 19:04:17,088 P38299 INFO Preprocess feature columns...
2022-03-01 19:04:59,865 P38299 INFO Transform feature columns...
2022-03-01 19:05:32,998 P38299 INFO Saving data to h5: ../data/Avazu/avazu_x4_d102865a/test.h5
2022-03-01 19:05:34,160 P38299 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-01 19:05:34,160 P38299 INFO Loading test data done.
2022-03-01 19:05:59,350 P38299 INFO [Metrics] logloss: 0.377348 - AUC: 0.783976

```
