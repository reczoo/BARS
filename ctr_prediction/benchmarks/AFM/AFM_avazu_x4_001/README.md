## AFM_avazu_x4_001

A hands-on guide to run the AFM model on the Avazu_x4_001 dataset.

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

We use [FuxiCTR-v1.0.2](fuxictr_url) for this experiment. See model code: [AFM](https://github.com/xue-pai/FuxiCTR/blob/v1.0.2/fuxictr/pytorch/models/AFM.py).

Running steps:

1. Download [FuxiCTR-v1.0.2](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AFM_avazu_x4_tuner_config_01](./AFM_avazu_x4_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AFM_avazu_x4_001
    nohup python run_expid.py --config ./AFM_avazu_x4_tuner_config_01 --expid AFM_avazu_x4_3bbbc4c9_008_8bd19e2a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| logloss | AUC  |
|:--------------------:|:--------------------:|
| 0.379283 | 0.782347  |


### Logs
```python
2022-02-24 14:32:53,009 P40956 INFO {
    "attention_dim": "32",
    "attention_dropout": "[0, 0]",
    "batch_size": "10000",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_3bbbc4c9",
    "debug": "False",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-07",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'name': 'id', 'active': False, 'dtype': 'str', 'type': 'categorical'}, {'name': 'hour', 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_hour'}, {'name': ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], 'active': True, 'dtype': 'str', 'type': 'categorical'}, {'name': 'weekday', 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_weekday'}, {'name': 'weekend', 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_weekend'}]",
    "gpu": "0",
    "label_col": "{'name': 'click', 'dtype': 'float'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "2",
    "model": "AFM",
    "model_id": "AFM_avazu_x4_3bbbc4c9_008_8bd19e2a",
    "model_root": "./Avazu/AFM_avazu/min2/",
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
    "test_data": "../data/Avazu/Avazu_x4/test.csv",
    "train_data": "../data/Avazu/Avazu_x4/train.csv",
    "use_attention": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x4/valid.csv",
    "verbose": "1",
    "version": "pytorch",
    "workers": "3"
}
2022-02-24 14:32:53,012 P40956 INFO Set up feature encoder...
2022-02-24 14:32:53,013 P40956 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-02-24 14:40:04,922 P40956 INFO Preprocess feature columns...
2022-02-24 15:02:47,422 P40956 INFO Fit feature encoder...
2022-02-24 15:02:47,423 P40956 INFO Processing column: {'name': 'hour', 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_hour'}
2022-02-24 15:11:01,433 P40956 INFO Processing column: {'name': 'C1', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:11:15,797 P40956 INFO Processing column: {'name': 'banner_pos', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:11:28,769 P40956 INFO Processing column: {'name': 'site_id', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:11:44,110 P40956 INFO Processing column: {'name': 'site_domain', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:11:58,726 P40956 INFO Processing column: {'name': 'site_category', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:12:13,201 P40956 INFO Processing column: {'name': 'app_id', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:12:28,154 P40956 INFO Processing column: {'name': 'app_domain', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:12:41,290 P40956 INFO Processing column: {'name': 'app_category', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:12:55,434 P40956 INFO Processing column: {'name': 'device_id', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:13:21,508 P40956 INFO Processing column: {'name': 'device_ip', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:14:28,959 P40956 INFO Processing column: {'name': 'device_model', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:14:49,715 P40956 INFO Processing column: {'name': 'device_type', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:15:12,775 P40956 INFO Processing column: {'name': 'device_conn_type', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:15:33,417 P40956 INFO Processing column: {'name': 'C14', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:15:58,301 P40956 INFO Processing column: {'name': 'C15', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:16:17,962 P40956 INFO Processing column: {'name': 'C16', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:16:37,503 P40956 INFO Processing column: {'name': 'C17', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:16:59,650 P40956 INFO Processing column: {'name': 'C18', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:17:19,067 P40956 INFO Processing column: {'name': 'C19', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:17:39,872 P40956 INFO Processing column: {'name': 'C20', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:18:01,518 P40956 INFO Processing column: {'name': 'C21', 'active': True, 'dtype': 'str', 'type': 'categorical'}
2022-02-24 15:18:23,704 P40956 INFO Processing column: {'name': 'weekday', 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_weekday'}
2022-02-24 15:25:36,303 P40956 INFO Processing column: {'name': 'weekend', 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_weekend'}
2022-02-24 15:32:52,817 P40956 INFO Set feature index...
2022-02-24 15:32:52,817 P40956 INFO Pickle feature_encode: ../data/Avazu/avazu_x4_3bbbc4c9/feature_encoder.pkl
2022-02-24 15:32:54,683 P40956 INFO Save feature_map to json: ../data/Avazu/avazu_x4_3bbbc4c9/feature_map.json
2022-02-24 15:32:54,684 P40956 INFO Set feature encoder done.
2022-02-24 15:33:01,150 P40956 INFO Total number of parameters: 63767168.
2022-02-24 15:33:01,151 P40956 INFO Loading data...
2022-02-24 15:33:01,156 P40956 INFO Reading file: ../data/Avazu/Avazu_x4/train.csv
2022-02-24 15:36:02,241 P40956 INFO Preprocess feature columns...
2022-02-24 15:48:25,827 P40956 INFO Transform feature columns...
2022-02-24 16:05:39,426 P40956 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/train.h5
2022-02-24 16:06:21,367 P40956 INFO Reading file: ../data/Avazu/Avazu_x4/valid.csv
2022-02-24 16:07:16,155 P40956 INFO Preprocess feature columns...
2022-02-24 16:10:46,662 P40956 INFO Transform feature columns...
2022-02-24 16:12:33,581 P40956 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/valid.h5
2022-02-24 16:12:44,694 P40956 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2022-02-24 16:12:45,003 P40956 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2022-02-24 16:12:45,003 P40956 INFO Loading train data done.
2022-02-24 16:12:48,909 P40956 INFO Start training: 3235 batches/epoch
2022-02-24 16:12:48,910 P40956 INFO ************ Epoch=1 start ************
2022-02-24 16:36:40,976 P40956 INFO [Metrics] logloss: 0.385129 - AUC: 0.771152
2022-02-24 16:36:40,987 P40956 INFO Save best model: monitor(max): 0.386023
2022-02-24 16:36:41,863 P40956 INFO --- 3235/3235 batches finished ---
2022-02-24 16:36:43,778 P40956 INFO Train loss: 0.396802
2022-02-24 16:36:43,778 P40956 INFO ************ Epoch=1 end ************
2022-02-24 16:59:11,283 P40956 INFO [Metrics] logloss: 0.379724 - AUC: 0.780769
2022-02-24 16:59:11,284 P40956 INFO Save best model: monitor(max): 0.401045
2022-02-24 16:59:11,890 P40956 INFO --- 3235/3235 batches finished ---
2022-02-24 16:59:14,172 P40956 INFO Train loss: 0.377435
2022-02-24 16:59:14,172 P40956 INFO ************ Epoch=2 end ************
2022-02-24 17:24:15,075 P40956 INFO [Metrics] logloss: 0.379473 - AUC: 0.782095
2022-02-24 17:24:15,076 P40956 INFO Save best model: monitor(max): 0.402622
2022-02-24 17:24:15,635 P40956 INFO --- 3235/3235 batches finished ---
2022-02-24 17:24:16,192 P40956 INFO Train loss: 0.364517
2022-02-24 17:24:16,192 P40956 INFO ************ Epoch=3 end ************
2022-02-24 17:45:58,737 P40956 INFO [Metrics] logloss: 0.381026 - AUC: 0.781557
2022-02-24 17:45:58,739 P40956 INFO Monitor(max) STOP: 0.400531 !
2022-02-24 17:45:58,740 P40956 INFO Reduce learning rate on plateau: 0.000100
2022-02-24 17:45:58,740 P40956 INFO --- 3235/3235 batches finished ---
2022-02-24 17:46:01,026 P40956 INFO Train loss: 0.356310
2022-02-24 17:46:01,026 P40956 INFO ************ Epoch=4 end ************
2022-02-24 18:10:18,591 P40956 INFO [Metrics] logloss: 0.388550 - AUC: 0.778318
2022-02-24 18:10:18,596 P40956 INFO Monitor(max) STOP: 0.389768 !
2022-02-24 18:10:18,597 P40956 INFO Reduce learning rate on plateau: 0.000010
2022-02-24 18:10:18,598 P40956 INFO Early stopping at epoch=5
2022-02-24 18:10:18,598 P40956 INFO --- 3235/3235 batches finished ---
2022-02-24 18:10:20,404 P40956 INFO Train loss: 0.339103
2022-02-24 18:10:20,405 P40956 INFO Training finished.
2022-02-24 18:10:20,405 P40956 INFO Load best model: /home/XXX/FuxiCTR_v1.0/benchmarks/Avazu/AFM_avazu/min2/avazu_x4_3bbbc4c9/AFM_avazu_x4_3bbbc4c9_008_8bd19e2a_model.ckpt
2022-02-24 18:10:21,298 P40956 INFO ****** Validation evaluation ******
2022-02-24 18:11:15,547 P40956 INFO [Metrics] logloss: 0.379473 - AUC: 0.782095
2022-02-24 18:11:16,800 P40956 INFO ******** Test evaluation ********
2022-02-24 18:11:16,801 P40956 INFO Loading data...
2022-02-24 18:11:16,801 P40956 INFO Reading file: ../data/Avazu/Avazu_x4/test.csv
2022-02-24 18:11:37,616 P40956 INFO Preprocess feature columns...
2022-02-24 18:13:23,835 P40956 INFO Transform feature columns...
2022-02-24 18:15:08,408 P40956 INFO Saving data to h5: ../data/Avazu/avazu_x4_3bbbc4c9/test.h5
2022-02-24 18:15:14,173 P40956 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2022-02-24 18:15:14,174 P40956 INFO Loading test data done.
2022-02-24 18:17:14,326 P40956 INFO [Metrics] logloss: 0.379283 - AUC: 0.782347

```
