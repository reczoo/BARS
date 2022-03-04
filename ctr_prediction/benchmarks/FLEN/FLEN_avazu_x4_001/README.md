## FLEN_avazu_x4_001

A hands-on guide to run the FLEN model on the Avazu_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FLEN](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FLEN.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FLEN_avazu_x4_tuner_config_01](./FLEN_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FLEN_avazu_x4_001
    nohup python run_expid.py --config ./FLEN_avazu_x4_tuner_config_01 --expid FLEN_avazu_x4_006_1e50e8f0 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371981 | 0.792871  |


### Logs
```python
2022-03-02 12:32:07,706 P55048 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_c9fb310a",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[2000, 2000, 2000, 2000]",
    "embedding_dim": "16",
    "embedding_regularizer": "1e-09",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'source': 'context', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'source': 'context', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'device_id', 'device_ip', 'device_model', 'device_type'], 'source': 'user', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['banner_pos', 'site_id', 'site_domain', 'site_category', 'device_conn_type'], 'source': 'context', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['app_id', 'app_domain', 'app_category'], 'source': 'item', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'source': 'context', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'source': 'context', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "2",
    "model": "FLEN",
    "model_id": "FLEN_avazu_x4_006_1e50e8f0",
    "model_root": "./Avazu/FLEN_avazu_x4_001/",
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
2022-03-02 12:32:07,707 P55048 INFO Set up feature encoder...
2022-03-02 12:32:07,707 P55048 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-02 12:34:07,892 P55048 INFO Preprocess feature columns...
2022-03-02 12:40:02,681 P55048 INFO Fit feature encoder...
2022-03-02 12:40:02,681 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:42:46,181 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C1', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:42:51,406 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C14', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:42:56,825 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C15', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:02,040 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C16', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:07,209 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C17', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:12,670 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C18', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:17,444 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C19', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:23,030 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C20', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:28,624 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C21', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:34,017 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_id', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:41,584 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_ip', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:43:59,377 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_model', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:44:05,306 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_type', 'source': 'user', 'type': 'categorical'}
2022-03-02 12:44:09,910 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'banner_pos', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:44:14,561 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_id', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:44:19,691 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_domain', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:44:24,800 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_category', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:44:29,923 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_conn_type', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:44:34,653 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_id', 'source': 'item', 'type': 'categorical'}
2022-03-02 12:44:39,753 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_domain', 'source': 'item', 'type': 'categorical'}
2022-03-02 12:44:44,919 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_category', 'source': 'item', 'type': 'categorical'}
2022-03-02 12:44:50,034 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:47:28,767 P55048 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'source': 'context', 'type': 'categorical'}
2022-03-02 12:50:08,331 P55048 INFO Set feature index...
2022-03-02 12:50:08,331 P55048 INFO Pickle feature_encode: ../data/Avazu/avazu_x4_c9fb310a/feature_encoder.pkl
2022-03-02 12:50:10,156 P55048 INFO Save feature_map to json: ../data/Avazu/avazu_x4_c9fb310a/feature_map.json
2022-03-02 12:50:10,157 P55048 INFO Set feature encoder done.
2022-03-02 12:50:15,135 P55048 INFO Total number of parameters: 76544888.
2022-03-02 12:50:15,135 P55048 INFO Loading data...
2022-03-02 12:50:15,139 P55048 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-02 12:52:13,918 P55048 INFO Preprocess feature columns...
2022-03-02 12:57:57,170 P55048 INFO Transform feature columns...
2022-03-02 13:02:24,874 P55048 INFO Saving data to h5: ../data/Avazu/avazu_x4_c9fb310a/train.h5
2022-03-02 13:02:34,930 P55048 INFO Reading file: ../data/Avazu/Avazu_x4/valid.csv
2022-03-02 13:02:49,128 P55048 INFO Preprocess feature columns...
2022-03-02 13:03:32,077 P55048 INFO Transform feature columns...
2022-03-02 13:04:05,849 P55048 INFO Saving data to h5: ../data/Avazu/avazu_x4_c9fb310a/valid.h5
2022-03-02 13:04:08,474 P55048 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-02 13:04:08,591 P55048 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-02 13:04:08,591 P55048 INFO Loading train data done.
2022-03-02 13:04:11,579 P55048 INFO Start training: 3235 batches/epoch
2022-03-02 13:04:11,579 P55048 INFO ************ Epoch=1 start ************
2022-03-02 13:20:11,682 P55048 INFO [Metrics] logloss: 0.372067 - AUC: 0.792726
2022-03-02 13:20:11,682 P55048 INFO Save best model: monitor(max): 0.420659
2022-03-02 13:20:11,966 P55048 INFO --- 3235/3235 batches finished ---
2022-03-02 13:20:12,231 P55048 INFO Train loss: 0.380235
2022-03-02 13:20:12,231 P55048 INFO ************ Epoch=1 end ************
2022-03-02 13:36:05,322 P55048 INFO [Metrics] logloss: 0.378966 - AUC: 0.789159
2022-03-02 13:36:05,323 P55048 INFO Monitor(max) STOP: 0.410193 !
2022-03-02 13:36:05,323 P55048 INFO Reduce learning rate on plateau: 0.000100
2022-03-02 13:36:05,323 P55048 INFO --- 3235/3235 batches finished ---
2022-03-02 13:36:05,585 P55048 INFO Train loss: 0.333217
2022-03-02 13:36:05,586 P55048 INFO ************ Epoch=2 end ************
2022-03-02 13:52:03,113 P55048 INFO [Metrics] logloss: 0.425216 - AUC: 0.776519
2022-03-02 13:52:03,114 P55048 INFO Monitor(max) STOP: 0.351303 !
2022-03-02 13:52:03,114 P55048 INFO Reduce learning rate on plateau: 0.000010
2022-03-02 13:52:03,114 P55048 INFO Early stopping at epoch=3
2022-03-02 13:52:03,114 P55048 INFO --- 3235/3235 batches finished ---
2022-03-02 13:52:03,417 P55048 INFO Train loss: 0.291971
2022-03-02 13:52:03,418 P55048 INFO Training finished.
2022-03-02 13:52:03,418 P55048 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/FLEN_avazu_x4_001/avazu_x4_c9fb310a/FLEN_avazu_x4_006_1e50e8f0_model.ckpt
2022-03-02 13:52:03,797 P55048 INFO ****** Validation evaluation ******
2022-03-02 13:52:30,154 P55048 INFO [Metrics] logloss: 0.372067 - AUC: 0.792726
2022-03-02 13:52:30,216 P55048 INFO ******** Test evaluation ********
2022-03-02 13:52:30,216 P55048 INFO Loading data...
2022-03-02 13:52:30,217 P55048 INFO Reading file: ../data/Avazu/Avazu_x4/test.csv
2022-03-02 13:52:45,002 P55048 INFO Preprocess feature columns...
2022-03-02 13:53:27,942 P55048 INFO Transform feature columns...
2022-03-02 13:54:00,808 P55048 INFO Saving data to h5: ../data/Avazu/avazu_x4_c9fb310a/test.h5
2022-03-02 13:54:02,117 P55048 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-02 13:54:02,117 P55048 INFO Loading test data done.
2022-03-02 13:54:29,453 P55048 INFO [Metrics] logloss: 0.371981 - AUC: 0.792871

```
