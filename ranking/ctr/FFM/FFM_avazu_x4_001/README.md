## FFM_avazu_x4_001

A hands-on guide to run the FFM model on the Avazu_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/tree/v1.0.2) for this experiment. See the model code: [FFM](https://github.com/reczoo/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/FFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.0.2.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x4`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FFM_avazu_x4_tuner_config_01](./FFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FFM_avazu_x4_001
    nohup python run_expid.py --config ./FFM_avazu_x4_tuner_config_01 --expid FFM_avazu_x4_003_792ee3b7 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.371960 | 0.793121  |


### Logs
```python
2022-03-01 20:43:43,732 P40741 INFO {
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "8",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': False, 'dtype': 'str', 'name': 'id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "2",
    "model": "FFM",
    "model_id": "FFM_avazu_x4_003_792ee3b7",
    "model_root": "./Avazu/FFM_avazu_x4_001/",
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
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-03-01 20:43:43,733 P40741 INFO Set up feature encoder...
2022-03-01 20:43:43,733 P40741 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-01 20:45:45,309 P40741 INFO Preprocess feature columns...
2022-03-01 20:51:30,469 P40741 INFO Fit feature encoder...
2022-03-01 20:51:30,469 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'hour', 'preprocess': 'convert_hour', 'type': 'categorical'}
2022-03-01 20:54:14,582 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C1', 'type': 'categorical'}
2022-03-01 20:54:19,534 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'banner_pos', 'type': 'categorical'}
2022-03-01 20:54:23,822 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_id', 'type': 'categorical'}
2022-03-01 20:54:28,771 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_domain', 'type': 'categorical'}
2022-03-01 20:54:33,785 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'site_category', 'type': 'categorical'}
2022-03-01 20:54:38,606 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_id', 'type': 'categorical'}
2022-03-01 20:54:43,577 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_domain', 'type': 'categorical'}
2022-03-01 20:54:48,306 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'app_category', 'type': 'categorical'}
2022-03-01 20:54:53,085 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_id', 'type': 'categorical'}
2022-03-01 20:55:00,189 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_ip', 'type': 'categorical'}
2022-03-01 20:55:16,303 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_model', 'type': 'categorical'}
2022-03-01 20:55:21,951 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_type', 'type': 'categorical'}
2022-03-01 20:55:26,324 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'device_conn_type', 'type': 'categorical'}
2022-03-01 20:55:30,653 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C14', 'type': 'categorical'}
2022-03-01 20:55:35,967 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C15', 'type': 'categorical'}
2022-03-01 20:55:40,703 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C16', 'type': 'categorical'}
2022-03-01 20:55:45,720 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C17', 'type': 'categorical'}
2022-03-01 20:55:50,831 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C18', 'type': 'categorical'}
2022-03-01 20:55:55,478 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C19', 'type': 'categorical'}
2022-03-01 20:56:00,662 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C20', 'type': 'categorical'}
2022-03-01 20:56:05,883 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'C21', 'type': 'categorical'}
2022-03-01 20:56:10,777 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekday', 'preprocess': 'convert_weekday', 'type': 'categorical'}
2022-03-01 20:58:52,259 P40741 INFO Processing column: {'active': True, 'dtype': 'str', 'name': 'weekend', 'preprocess': 'convert_weekend', 'type': 'categorical'}
2022-03-01 21:01:35,883 P40741 INFO Set feature index...
2022-03-01 21:01:35,884 P40741 INFO Pickle feature_encode: ../data/Avazu/avazu_x4_3bbbc4c9/feature_encoder.pkl
2022-03-01 21:01:37,574 P40741 INFO Save feature_map to json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2022-03-01 21:01:37,575 P40741 INFO Set feature encoder done.
2022-03-01 21:01:56,049 P40741 INFO Total number of parameters: 693930376.
2022-03-01 21:01:56,050 P40741 INFO Loading data...
2022-03-01 21:01:56,053 P40741 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-03-01 21:03:55,494 P40741 INFO Preprocess feature columns...
2022-03-01 21:09:38,007 P40741 INFO Transform feature columns...
2022-03-01 21:14:04,364 P40741 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2022-03-01 21:14:13,404 P40741 INFO Reading file: ../data/Avazu/Avazu_x4/valid.csv
2022-03-01 21:14:27,698 P40741 INFO Preprocess feature columns...
2022-03-01 21:15:13,568 P40741 INFO Transform feature columns...
2022-03-01 21:15:51,581 P40741 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2022-03-01 21:15:54,274 P40741 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-03-01 21:15:54,662 P40741 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-03-01 21:15:54,662 P40741 INFO Loading train data done.
2022-03-01 21:15:58,015 P40741 INFO Start training: 3235 batches/epoch
2022-03-01 21:15:58,016 P40741 INFO ************ Epoch=1 start ************
2022-03-01 23:13:05,354 P40741 INFO [Metrics] logloss: 0.371829 - AUC: 0.793346
2022-03-01 23:13:05,355 P40741 INFO Save best model: monitor(max): 0.421517
2022-03-01 23:13:07,317 P40741 INFO --- 3235/3235 batches finished ---
2022-03-01 23:13:07,689 P40741 INFO Train loss: 0.381648
2022-03-01 23:13:07,689 P40741 INFO ************ Epoch=1 end ************
2022-03-02 01:10:40,100 P40741 INFO [Metrics] logloss: 0.378712 - AUC: 0.790570
2022-03-02 01:10:40,101 P40741 INFO Monitor(max) STOP: 0.411857 !
2022-03-02 01:10:40,101 P40741 INFO Reduce learning rate on plateau: 0.000100
2022-03-02 01:10:40,101 P40741 INFO --- 3235/3235 batches finished ---
2022-03-02 01:10:40,529 P40741 INFO Train loss: 0.329993
2022-03-02 01:10:40,530 P40741 INFO ************ Epoch=2 end ************
2022-03-02 03:07:46,954 P40741 INFO [Metrics] logloss: 0.400290 - AUC: 0.781329
2022-03-02 03:07:46,954 P40741 INFO Monitor(max) STOP: 0.381039 !
2022-03-02 03:07:46,955 P40741 INFO Reduce learning rate on plateau: 0.000010
2022-03-02 03:07:46,955 P40741 INFO Early stopping at epoch=3
2022-03-02 03:07:46,955 P40741 INFO --- 3235/3235 batches finished ---
2022-03-02 03:07:47,372 P40741 INFO Train loss: 0.281902
2022-03-02 03:07:47,373 P40741 INFO Training finished.
2022-03-02 03:07:47,373 P40741 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Avazu/FFM_avazu_x4_001/avazu_x4_3bbbc4c9/FFM_avazu_x4_003_792ee3b7_model.ckpt
2022-03-02 03:07:51,948 P40741 INFO ****** Validation evaluation ******
2022-03-02 03:08:22,081 P40741 INFO [Metrics] logloss: 0.371829 - AUC: 0.793346
2022-03-02 03:08:22,188 P40741 INFO ******** Test evaluation ********
2022-03-02 03:08:22,188 P40741 INFO Loading data...
2022-03-02 03:08:22,188 P40741 INFO Reading file: ../data/Avazu/Avazu_x4/test.csv
2022-03-02 03:08:38,471 P40741 INFO Preprocess feature columns...
2022-03-02 03:09:21,769 P40741 INFO Transform feature columns...
2022-03-02 03:09:56,647 P40741 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2022-03-02 03:09:57,994 P40741 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-03-02 03:09:57,995 P40741 INFO Loading test data done.
2022-03-02 03:10:31,442 P40741 INFO [Metrics] logloss: 0.371960 - AUC: 0.793121

```
