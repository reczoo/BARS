## FiBiNET_frappe_x1

A hands-on guide to run the FiBiNET model on the Frappe_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  GPU: Tesla P100 16G
  RAM: 755G

  ```

+ Software

  ```python
  CUDA: 11.4
  python: 3.6.5
  pytorch: 1.0.1.post2
  pandas: 0.23.0
  numpy: 1.18.1
  scipy: 1.1.0
  sklearn: 0.23.1
  pyyaml: 5.1
  h5py: 2.7.1
  tqdm: 4.59.0
  fuxictr: 1.1.0
  ```

### Dataset
Dataset ID: [Frappe_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Frappe/README.md#Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [FiBiNET](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiBiNET.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiBiNET_frappe_x1_tuner_config_03](./FiBiNET_frappe_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiBiNET_frappe_x1
    nohup python run_expid.py --config ./FiBiNET_frappe_x1_tuner_config_03 --expid FiBiNET_frappe_x1_002_aedd06a6 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.983182 | 0.194074  |


### Logs
```python
2022-01-31 17:33:15,272 P11118 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "bilinear_type": "field_all",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FiBiNET",
    "model_id": "FiBiNET_frappe_x1_002_aedd06a6",
    "model_root": "./Frappe/FiBiNET_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.3",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "2",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Frappe/Frappe_x1/test.csv",
    "train_data": "../data/Frappe/Frappe_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-31 17:33:15,273 P11118 INFO Set up feature encoder...
2022-01-31 17:33:15,273 P11118 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-31 17:33:15,273 P11118 INFO Loading data...
2022-01-31 17:33:15,275 P11118 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-31 17:33:15,286 P11118 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-31 17:33:15,290 P11118 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-31 17:33:15,290 P11118 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-31 17:33:15,290 P11118 INFO Loading train data done.
2022-01-31 17:33:18,996 P11118 INFO Total number of parameters: 743480.
2022-01-31 17:33:18,997 P11118 INFO Start training: 50 batches/epoch
2022-01-31 17:33:18,997 P11118 INFO ************ Epoch=1 start ************
2022-01-31 17:33:26,724 P11118 INFO [Metrics] AUC: 0.932828 - logloss: 0.640742
2022-01-31 17:33:26,724 P11118 INFO Save best model: monitor(max): 0.932828
2022-01-31 17:33:26,730 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:33:26,772 P11118 INFO Train loss: 0.366241
2022-01-31 17:33:26,772 P11118 INFO ************ Epoch=1 end ************
2022-01-31 17:33:34,315 P11118 INFO [Metrics] AUC: 0.960552 - logloss: 0.414991
2022-01-31 17:33:34,315 P11118 INFO Save best model: monitor(max): 0.960552
2022-01-31 17:33:34,322 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:33:34,381 P11118 INFO Train loss: 0.271461
2022-01-31 17:33:34,381 P11118 INFO ************ Epoch=2 end ************
2022-01-31 17:33:41,894 P11118 INFO [Metrics] AUC: 0.968844 - logloss: 0.227315
2022-01-31 17:33:41,895 P11118 INFO Save best model: monitor(max): 0.968844
2022-01-31 17:33:41,901 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:33:41,943 P11118 INFO Train loss: 0.228772
2022-01-31 17:33:41,943 P11118 INFO ************ Epoch=3 end ************
2022-01-31 17:33:49,518 P11118 INFO [Metrics] AUC: 0.970508 - logloss: 0.904317
2022-01-31 17:33:49,518 P11118 INFO Save best model: monitor(max): 0.970508
2022-01-31 17:33:49,524 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:33:49,563 P11118 INFO Train loss: 0.203686
2022-01-31 17:33:49,563 P11118 INFO ************ Epoch=4 end ************
2022-01-31 17:33:57,083 P11118 INFO [Metrics] AUC: 0.971387 - logloss: 0.898646
2022-01-31 17:33:57,083 P11118 INFO Save best model: monitor(max): 0.971387
2022-01-31 17:33:57,090 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:33:57,129 P11118 INFO Train loss: 0.191795
2022-01-31 17:33:57,129 P11118 INFO ************ Epoch=5 end ************
2022-01-31 17:34:04,859 P11118 INFO [Metrics] AUC: 0.971795 - logloss: 2.609997
2022-01-31 17:34:04,860 P11118 INFO Save best model: monitor(max): 0.971795
2022-01-31 17:34:04,866 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:34:04,909 P11118 INFO Train loss: 0.185367
2022-01-31 17:34:04,909 P11118 INFO ************ Epoch=6 end ************
2022-01-31 17:34:12,610 P11118 INFO [Metrics] AUC: 0.966920 - logloss: 4.645360
2022-01-31 17:34:12,611 P11118 INFO Monitor(max) STOP: 0.966920 !
2022-01-31 17:34:12,611 P11118 INFO Reduce learning rate on plateau: 0.000100
2022-01-31 17:34:12,611 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:34:12,666 P11118 INFO Train loss: 0.180811
2022-01-31 17:34:12,666 P11118 INFO ************ Epoch=7 end ************
2022-01-31 17:34:20,185 P11118 INFO [Metrics] AUC: 0.982165 - logloss: 0.157065
2022-01-31 17:34:20,185 P11118 INFO Save best model: monitor(max): 0.982165
2022-01-31 17:34:20,192 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:34:20,228 P11118 INFO Train loss: 0.145691
2022-01-31 17:34:20,228 P11118 INFO ************ Epoch=8 end ************
2022-01-31 17:34:27,778 P11118 INFO [Metrics] AUC: 0.982529 - logloss: 0.196734
2022-01-31 17:34:27,779 P11118 INFO Save best model: monitor(max): 0.982529
2022-01-31 17:34:27,785 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:34:27,827 P11118 INFO Train loss: 0.121074
2022-01-31 17:34:27,827 P11118 INFO ************ Epoch=9 end ************
2022-01-31 17:34:35,422 P11118 INFO [Metrics] AUC: 0.982882 - logloss: 0.192746
2022-01-31 17:34:35,422 P11118 INFO Save best model: monitor(max): 0.982882
2022-01-31 17:34:35,428 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:34:35,464 P11118 INFO Train loss: 0.106040
2022-01-31 17:34:35,464 P11118 INFO ************ Epoch=10 end ************
2022-01-31 17:34:40,905 P11118 INFO [Metrics] AUC: 0.981138 - logloss: 0.402629
2022-01-31 17:34:40,905 P11118 INFO Monitor(max) STOP: 0.981138 !
2022-01-31 17:34:40,905 P11118 INFO Reduce learning rate on plateau: 0.000010
2022-01-31 17:34:40,905 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:34:40,942 P11118 INFO Train loss: 0.095479
2022-01-31 17:34:40,942 P11118 INFO ************ Epoch=11 end ************
2022-01-31 17:34:46,204 P11118 INFO [Metrics] AUC: 0.982314 - logloss: 0.166057
2022-01-31 17:34:46,205 P11118 INFO Monitor(max) STOP: 0.982314 !
2022-01-31 17:34:46,205 P11118 INFO Reduce learning rate on plateau: 0.000001
2022-01-31 17:34:46,205 P11118 INFO Early stopping at epoch=12
2022-01-31 17:34:46,205 P11118 INFO --- 50/50 batches finished ---
2022-01-31 17:34:46,238 P11118 INFO Train loss: 0.086467
2022-01-31 17:34:46,238 P11118 INFO Training finished.
2022-01-31 17:34:46,238 P11118 INFO Load best model: /home/XXX/benchmarks/Frappe/FiBiNET_frappe_x1/frappe_x1_04e961e9/FiBiNET_frappe_x1_002_aedd06a6.model
2022-01-31 17:34:49,366 P11118 INFO ****** Validation evaluation ******
2022-01-31 17:34:49,814 P11118 INFO [Metrics] AUC: 0.982882 - logloss: 0.192746
2022-01-31 17:34:49,848 P11118 INFO ******** Test evaluation ********
2022-01-31 17:34:49,848 P11118 INFO Loading data...
2022-01-31 17:34:49,848 P11118 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-31 17:34:49,851 P11118 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-31 17:34:49,851 P11118 INFO Loading test data done.
2022-01-31 17:34:50,163 P11118 INFO [Metrics] AUC: 0.983182 - logloss: 0.194074

```
