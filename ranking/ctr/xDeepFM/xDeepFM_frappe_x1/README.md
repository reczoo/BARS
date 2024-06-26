## xDeepFM_frappe_x1

A hands-on guide to run the xDeepFM model on the Frappe_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
Dataset ID: [Frappe_x1](https://github.com/reczoo/Datasets/tree/main/Frappe/Frappe_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [xDeepFM](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [xDeepFM_frappe_x1_tuner_config_02](./xDeepFM_frappe_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd xDeepFM_frappe_x1
    nohup python run_expid.py --config ./xDeepFM_frappe_x1_tuner_config_02 --expid xDeepFM_frappe_x1_005_447fa536 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.984507 | 0.146604  |
| 2 | 0.984446 | 0.143949  |
| 3 | 0.983468 | 0.149751  |
| 4 | 0.984208 | 0.146196  |
| 5 | 0.984059 | 0.144127  |
| Avg | 0.984138 | 0.146125 |
| Std | &#177;0.00037177 | &#177;0.00210282 |


### Logs
```python
2022-01-18 11:37:21,960 P2676 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "cin_layer_units": "[64]",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_frappe_x1_005_447fa536",
    "model_root": "./Frappe/xDeepFM_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
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
2022-01-18 11:37:21,961 P2676 INFO Set up feature encoder...
2022-01-18 11:37:21,961 P2676 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-18 11:37:21,961 P2676 INFO Loading data...
2022-01-18 11:37:21,963 P2676 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-18 11:37:21,974 P2676 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-18 11:37:21,979 P2676 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-18 11:37:21,979 P2676 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-18 11:37:21,979 P2676 INFO Loading train data done.
2022-01-18 11:37:25,069 P2676 INFO Total number of parameters: 429809.
2022-01-18 11:37:25,070 P2676 INFO Start training: 50 batches/epoch
2022-01-18 11:37:25,070 P2676 INFO ************ Epoch=1 start ************
2022-01-18 11:37:33,062 P2676 INFO [Metrics] AUC: 0.934883 - logloss: 0.698188
2022-01-18 11:37:33,062 P2676 INFO Save best model: monitor(max): 0.934883
2022-01-18 11:37:33,067 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:37:33,103 P2676 INFO Train loss: 0.384318
2022-01-18 11:37:33,103 P2676 INFO ************ Epoch=1 end ************
2022-01-18 11:37:41,014 P2676 INFO [Metrics] AUC: 0.958553 - logloss: 0.292481
2022-01-18 11:37:41,015 P2676 INFO Save best model: monitor(max): 0.958553
2022-01-18 11:37:41,020 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:37:41,050 P2676 INFO Train loss: 0.287205
2022-01-18 11:37:41,050 P2676 INFO ************ Epoch=2 end ************
2022-01-18 11:37:48,935 P2676 INFO [Metrics] AUC: 0.970371 - logloss: 0.216380
2022-01-18 11:37:48,935 P2676 INFO Save best model: monitor(max): 0.970371
2022-01-18 11:37:48,940 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:37:48,971 P2676 INFO Train loss: 0.254174
2022-01-18 11:37:48,971 P2676 INFO ************ Epoch=3 end ************
2022-01-18 11:37:56,733 P2676 INFO [Metrics] AUC: 0.974030 - logloss: 0.190211
2022-01-18 11:37:56,734 P2676 INFO Save best model: monitor(max): 0.974030
2022-01-18 11:37:56,739 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:37:56,772 P2676 INFO Train loss: 0.238086
2022-01-18 11:37:56,772 P2676 INFO ************ Epoch=4 end ************
2022-01-18 11:38:10,103 P2676 INFO [Metrics] AUC: 0.975718 - logloss: 0.175388
2022-01-18 11:38:10,104 P2676 INFO Save best model: monitor(max): 0.975718
2022-01-18 11:38:10,109 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:38:10,141 P2676 INFO Train loss: 0.227837
2022-01-18 11:38:10,142 P2676 INFO ************ Epoch=5 end ************
2022-01-18 11:38:25,337 P2676 INFO [Metrics] AUC: 0.977502 - logloss: 0.178191
2022-01-18 11:38:25,337 P2676 INFO Save best model: monitor(max): 0.977502
2022-01-18 11:38:25,342 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:38:25,380 P2676 INFO Train loss: 0.222635
2022-01-18 11:38:25,381 P2676 INFO ************ Epoch=6 end ************
2022-01-18 11:38:40,637 P2676 INFO [Metrics] AUC: 0.976834 - logloss: 0.170260
2022-01-18 11:38:40,637 P2676 INFO Monitor(max) STOP: 0.976834 !
2022-01-18 11:38:40,637 P2676 INFO Reduce learning rate on plateau: 0.000100
2022-01-18 11:38:40,638 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:38:40,679 P2676 INFO Train loss: 0.217524
2022-01-18 11:38:40,679 P2676 INFO ************ Epoch=7 end ************
2022-01-18 11:38:55,866 P2676 INFO [Metrics] AUC: 0.982700 - logloss: 0.143619
2022-01-18 11:38:55,867 P2676 INFO Save best model: monitor(max): 0.982700
2022-01-18 11:38:55,872 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:38:55,902 P2676 INFO Train loss: 0.184331
2022-01-18 11:38:55,902 P2676 INFO ************ Epoch=8 end ************
2022-01-18 11:39:11,183 P2676 INFO [Metrics] AUC: 0.984177 - logloss: 0.138543
2022-01-18 11:39:11,183 P2676 INFO Save best model: monitor(max): 0.984177
2022-01-18 11:39:11,188 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:39:11,224 P2676 INFO Train loss: 0.147203
2022-01-18 11:39:11,224 P2676 INFO ************ Epoch=9 end ************
2022-01-18 11:39:26,477 P2676 INFO [Metrics] AUC: 0.984648 - logloss: 0.138857
2022-01-18 11:39:26,477 P2676 INFO Save best model: monitor(max): 0.984648
2022-01-18 11:39:26,482 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:39:26,514 P2676 INFO Train loss: 0.124709
2022-01-18 11:39:26,515 P2676 INFO ************ Epoch=10 end ************
2022-01-18 11:39:41,932 P2676 INFO [Metrics] AUC: 0.984883 - logloss: 0.140546
2022-01-18 11:39:41,932 P2676 INFO Save best model: monitor(max): 0.984883
2022-01-18 11:39:41,938 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:39:41,971 P2676 INFO Train loss: 0.108857
2022-01-18 11:39:41,971 P2676 INFO ************ Epoch=11 end ************
2022-01-18 11:39:57,357 P2676 INFO [Metrics] AUC: 0.985116 - logloss: 0.141438
2022-01-18 11:39:57,358 P2676 INFO Save best model: monitor(max): 0.985116
2022-01-18 11:39:57,363 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:39:57,396 P2676 INFO Train loss: 0.097397
2022-01-18 11:39:57,396 P2676 INFO ************ Epoch=12 end ************
2022-01-18 11:40:12,569 P2676 INFO [Metrics] AUC: 0.984755 - logloss: 0.144713
2022-01-18 11:40:12,569 P2676 INFO Monitor(max) STOP: 0.984755 !
2022-01-18 11:40:12,569 P2676 INFO Reduce learning rate on plateau: 0.000010
2022-01-18 11:40:12,569 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:40:12,601 P2676 INFO Train loss: 0.089385
2022-01-18 11:40:12,601 P2676 INFO ************ Epoch=13 end ************
2022-01-18 11:40:27,723 P2676 INFO [Metrics] AUC: 0.984974 - logloss: 0.146651
2022-01-18 11:40:27,723 P2676 INFO Monitor(max) STOP: 0.984974 !
2022-01-18 11:40:27,723 P2676 INFO Reduce learning rate on plateau: 0.000001
2022-01-18 11:40:27,723 P2676 INFO Early stopping at epoch=14
2022-01-18 11:40:27,723 P2676 INFO --- 50/50 batches finished ---
2022-01-18 11:40:27,754 P2676 INFO Train loss: 0.080105
2022-01-18 11:40:27,754 P2676 INFO Training finished.
2022-01-18 11:40:27,754 P2676 INFO Load best model: /home/XXX/benchmarks/Frappe/xDeepFM_frappe_x1/frappe_x1_04e961e9/xDeepFM_frappe_x1_005_447fa536.model
2022-01-18 11:40:27,834 P2676 INFO ****** Validation evaluation ******
2022-01-18 11:40:28,216 P2676 INFO [Metrics] AUC: 0.985116 - logloss: 0.141438
2022-01-18 11:40:28,254 P2676 INFO ******** Test evaluation ********
2022-01-18 11:40:28,254 P2676 INFO Loading data...
2022-01-18 11:40:28,254 P2676 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-18 11:40:28,257 P2676 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-18 11:40:28,257 P2676 INFO Loading test data done.
2022-01-18 11:40:28,565 P2676 INFO [Metrics] AUC: 0.984507 - logloss: 0.146604

```
