## AutoInt_frappe_x1

A hands-on guide to run the AutoInt model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [AutoInt](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/AutoInt.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [AutoInt_frappe_x1_tuner_config_03](./AutoInt_frappe_x1_tuner_config_03). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd AutoInt_frappe_x1
    nohup python run_expid.py --config ./AutoInt_frappe_x1_tuner_config_03 --expid AutoInt_frappe_x1_006_0eb83bd7 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 14 runs:
| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.982263 | 0.169137  |
| 2 | 0.982244 | 0.160713  |
| 3 | 0.981941 | 0.171024  |
| 4 | 0.981738 | 0.171480  |
| 5 | 0.981281 | 0.174519  |
| 6 | 0.980344 | 0.182125  |
| 7 | 0.980277 | 0.183588  |
| 8 | 0.979484 | 0.181713  |
| 9 | 0.979331 | 0.170642  |
| 10 | 0.979049 | 0.197980  |
| 11 | 0.979044 | 0.191088  |
| 12 | 0.978981 | 0.188947  |
| 13 | 0.978515 | 0.183773  |
| 14 | 0.978113 | 0.183440  |
| | | | 
| Avg | 0.9801860714285714 | 0.17929778571428573 |
| Std | &#177;0.0014049059177592945 | &#177;0.009734063864143511 |


### Logs
```python
2022-01-23 18:15:20,166 P15081 INFO {
    "attention_dim": "128",
    "attention_layers": "7",
    "batch_norm": "False",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[]",
    "embedding_dim": "10",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "1",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "AutoInt",
    "model_id": "AutoInt_frappe_x1_006_0eb83bd7",
    "model_root": "./Frappe/AutoInt_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
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
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "valid_data": "../data/Frappe/Frappe_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-23 18:15:20,167 P15081 INFO Set up feature encoder...
2022-01-23 18:15:20,167 P15081 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-01-23 18:15:20,167 P15081 INFO Loading data...
2022-01-23 18:15:20,170 P15081 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-01-23 18:15:20,185 P15081 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-01-23 18:15:20,192 P15081 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-01-23 18:15:20,192 P15081 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-01-23 18:15:20,193 P15081 INFO Loading train data done.
2022-01-23 18:15:25,296 P15081 INFO Total number of parameters: 355203.
2022-01-23 18:15:25,296 P15081 INFO Start training: 50 batches/epoch
2022-01-23 18:15:25,297 P15081 INFO ************ Epoch=1 start ************
2022-01-23 18:15:47,641 P15081 INFO [Metrics] AUC: 0.929242 - logloss: 0.388581
2022-01-23 18:15:47,641 P15081 INFO Save best model: monitor(max): 0.929242
2022-01-23 18:15:47,647 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:15:47,685 P15081 INFO Train loss: 0.565425
2022-01-23 18:15:47,685 P15081 INFO ************ Epoch=1 end ************
2022-01-23 18:16:09,482 P15081 INFO [Metrics] AUC: 0.936353 - logloss: 0.285543
2022-01-23 18:16:09,482 P15081 INFO Save best model: monitor(max): 0.936353
2022-01-23 18:16:09,487 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:16:09,527 P15081 INFO Train loss: 0.330035
2022-01-23 18:16:09,527 P15081 INFO ************ Epoch=2 end ************
2022-01-23 18:16:31,012 P15081 INFO [Metrics] AUC: 0.939177 - logloss: 0.281787
2022-01-23 18:16:31,013 P15081 INFO Save best model: monitor(max): 0.939177
2022-01-23 18:16:31,017 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:16:31,053 P15081 INFO Train loss: 0.305692
2022-01-23 18:16:31,054 P15081 INFO ************ Epoch=3 end ************
2022-01-23 18:16:52,908 P15081 INFO [Metrics] AUC: 0.941802 - logloss: 0.274546
2022-01-23 18:16:52,909 P15081 INFO Save best model: monitor(max): 0.941802
2022-01-23 18:16:52,913 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:16:52,953 P15081 INFO Train loss: 0.296474
2022-01-23 18:16:52,954 P15081 INFO ************ Epoch=4 end ************
2022-01-23 18:17:14,817 P15081 INFO [Metrics] AUC: 0.948097 - logloss: 0.259215
2022-01-23 18:17:14,818 P15081 INFO Save best model: monitor(max): 0.948097
2022-01-23 18:17:14,822 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:17:14,859 P15081 INFO Train loss: 0.288219
2022-01-23 18:17:14,859 P15081 INFO ************ Epoch=5 end ************
2022-01-23 18:17:36,624 P15081 INFO [Metrics] AUC: 0.953848 - logloss: 0.248725
2022-01-23 18:17:36,624 P15081 INFO Save best model: monitor(max): 0.953848
2022-01-23 18:17:36,628 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:17:36,672 P15081 INFO Train loss: 0.277879
2022-01-23 18:17:36,672 P15081 INFO ************ Epoch=6 end ************
2022-01-23 18:17:58,380 P15081 INFO [Metrics] AUC: 0.956707 - logloss: 0.236726
2022-01-23 18:17:58,381 P15081 INFO Save best model: monitor(max): 0.956707
2022-01-23 18:17:58,385 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:17:58,425 P15081 INFO Train loss: 0.268859
2022-01-23 18:17:58,425 P15081 INFO ************ Epoch=7 end ************
2022-01-23 18:18:20,362 P15081 INFO [Metrics] AUC: 0.960537 - logloss: 0.226316
2022-01-23 18:18:20,363 P15081 INFO Save best model: monitor(max): 0.960537
2022-01-23 18:18:20,367 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:18:20,404 P15081 INFO Train loss: 0.258923
2022-01-23 18:18:20,404 P15081 INFO ************ Epoch=8 end ************
2022-01-23 18:18:42,496 P15081 INFO [Metrics] AUC: 0.961970 - logloss: 0.225191
2022-01-23 18:18:42,497 P15081 INFO Save best model: monitor(max): 0.961970
2022-01-23 18:18:42,501 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:18:42,539 P15081 INFO Train loss: 0.254275
2022-01-23 18:18:42,539 P15081 INFO ************ Epoch=9 end ************
2022-01-23 18:19:04,140 P15081 INFO [Metrics] AUC: 0.963819 - logloss: 0.220582
2022-01-23 18:19:04,140 P15081 INFO Save best model: monitor(max): 0.963819
2022-01-23 18:19:04,144 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:19:04,191 P15081 INFO Train loss: 0.248178
2022-01-23 18:19:04,191 P15081 INFO ************ Epoch=10 end ************
2022-01-23 18:19:25,565 P15081 INFO [Metrics] AUC: 0.964403 - logloss: 0.219453
2022-01-23 18:19:25,565 P15081 INFO Save best model: monitor(max): 0.964403
2022-01-23 18:19:25,569 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:19:25,611 P15081 INFO Train loss: 0.244098
2022-01-23 18:19:25,611 P15081 INFO ************ Epoch=11 end ************
2022-01-23 18:19:47,153 P15081 INFO [Metrics] AUC: 0.965947 - logloss: 0.210579
2022-01-23 18:19:47,153 P15081 INFO Save best model: monitor(max): 0.965947
2022-01-23 18:19:47,157 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:19:47,195 P15081 INFO Train loss: 0.238869
2022-01-23 18:19:47,195 P15081 INFO ************ Epoch=12 end ************
2022-01-23 18:20:09,182 P15081 INFO [Metrics] AUC: 0.967149 - logloss: 0.207849
2022-01-23 18:20:09,183 P15081 INFO Save best model: monitor(max): 0.967149
2022-01-23 18:20:09,187 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:20:09,251 P15081 INFO Train loss: 0.237869
2022-01-23 18:20:09,251 P15081 INFO ************ Epoch=13 end ************
2022-01-23 18:20:31,148 P15081 INFO [Metrics] AUC: 0.969113 - logloss: 0.200254
2022-01-23 18:20:31,148 P15081 INFO Save best model: monitor(max): 0.969113
2022-01-23 18:20:31,152 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:20:31,195 P15081 INFO Train loss: 0.233040
2022-01-23 18:20:31,195 P15081 INFO ************ Epoch=14 end ************
2022-01-23 18:20:52,901 P15081 INFO [Metrics] AUC: 0.969190 - logloss: 0.200513
2022-01-23 18:20:52,901 P15081 INFO Save best model: monitor(max): 0.969190
2022-01-23 18:20:52,905 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:20:52,957 P15081 INFO Train loss: 0.231243
2022-01-23 18:20:52,958 P15081 INFO ************ Epoch=15 end ************
2022-01-23 18:21:14,468 P15081 INFO [Metrics] AUC: 0.971145 - logloss: 0.198363
2022-01-23 18:21:14,469 P15081 INFO Save best model: monitor(max): 0.971145
2022-01-23 18:21:14,473 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:21:14,509 P15081 INFO Train loss: 0.227569
2022-01-23 18:21:14,510 P15081 INFO ************ Epoch=16 end ************
2022-01-23 18:21:36,381 P15081 INFO [Metrics] AUC: 0.971635 - logloss: 0.192158
2022-01-23 18:21:36,381 P15081 INFO Save best model: monitor(max): 0.971635
2022-01-23 18:21:36,385 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:21:36,423 P15081 INFO Train loss: 0.225160
2022-01-23 18:21:36,423 P15081 INFO ************ Epoch=17 end ************
2022-01-23 18:21:57,936 P15081 INFO [Metrics] AUC: 0.971937 - logloss: 0.190683
2022-01-23 18:21:57,936 P15081 INFO Save best model: monitor(max): 0.971937
2022-01-23 18:21:57,940 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:21:57,978 P15081 INFO Train loss: 0.223913
2022-01-23 18:21:57,978 P15081 INFO ************ Epoch=18 end ************
2022-01-23 18:22:19,804 P15081 INFO [Metrics] AUC: 0.972454 - logloss: 0.186339
2022-01-23 18:22:19,805 P15081 INFO Save best model: monitor(max): 0.972454
2022-01-23 18:22:19,809 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:22:19,846 P15081 INFO Train loss: 0.220088
2022-01-23 18:22:19,846 P15081 INFO ************ Epoch=19 end ************
2022-01-23 18:22:41,589 P15081 INFO [Metrics] AUC: 0.973932 - logloss: 0.183628
2022-01-23 18:22:41,590 P15081 INFO Save best model: monitor(max): 0.973932
2022-01-23 18:22:41,594 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:22:41,634 P15081 INFO Train loss: 0.217382
2022-01-23 18:22:41,634 P15081 INFO ************ Epoch=20 end ************
2022-01-23 18:23:03,480 P15081 INFO [Metrics] AUC: 0.974394 - logloss: 0.183487
2022-01-23 18:23:03,480 P15081 INFO Save best model: monitor(max): 0.974394
2022-01-23 18:23:03,485 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:23:03,527 P15081 INFO Train loss: 0.213499
2022-01-23 18:23:03,527 P15081 INFO ************ Epoch=21 end ************
2022-01-23 18:23:25,439 P15081 INFO [Metrics] AUC: 0.975374 - logloss: 0.177700
2022-01-23 18:23:25,439 P15081 INFO Save best model: monitor(max): 0.975374
2022-01-23 18:23:25,443 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:23:25,482 P15081 INFO Train loss: 0.211741
2022-01-23 18:23:25,482 P15081 INFO ************ Epoch=22 end ************
2022-01-23 18:23:47,159 P15081 INFO [Metrics] AUC: 0.975495 - logloss: 0.177530
2022-01-23 18:23:47,159 P15081 INFO Save best model: monitor(max): 0.975495
2022-01-23 18:23:47,163 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:23:47,201 P15081 INFO Train loss: 0.211479
2022-01-23 18:23:47,201 P15081 INFO ************ Epoch=23 end ************
2022-01-23 18:24:08,891 P15081 INFO [Metrics] AUC: 0.975911 - logloss: 0.178364
2022-01-23 18:24:08,892 P15081 INFO Save best model: monitor(max): 0.975911
2022-01-23 18:24:08,898 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:24:08,935 P15081 INFO Train loss: 0.208048
2022-01-23 18:24:08,936 P15081 INFO ************ Epoch=24 end ************
2022-01-23 18:24:30,712 P15081 INFO [Metrics] AUC: 0.976383 - logloss: 0.174331
2022-01-23 18:24:30,712 P15081 INFO Save best model: monitor(max): 0.976383
2022-01-23 18:24:30,716 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:24:30,770 P15081 INFO Train loss: 0.208620
2022-01-23 18:24:30,770 P15081 INFO ************ Epoch=25 end ************
2022-01-23 18:24:52,551 P15081 INFO [Metrics] AUC: 0.977241 - logloss: 0.169820
2022-01-23 18:24:52,551 P15081 INFO Save best model: monitor(max): 0.977241
2022-01-23 18:24:52,556 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:24:52,594 P15081 INFO Train loss: 0.206128
2022-01-23 18:24:52,594 P15081 INFO ************ Epoch=26 end ************
2022-01-23 18:25:14,072 P15081 INFO [Metrics] AUC: 0.976883 - logloss: 0.171011
2022-01-23 18:25:14,072 P15081 INFO Monitor(max) STOP: 0.976883 !
2022-01-23 18:25:14,073 P15081 INFO Reduce learning rate on plateau: 0.000100
2022-01-23 18:25:14,073 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:25:14,112 P15081 INFO Train loss: 0.202170
2022-01-23 18:25:14,112 P15081 INFO ************ Epoch=27 end ************
2022-01-23 18:25:32,685 P15081 INFO [Metrics] AUC: 0.980128 - logloss: 0.163108
2022-01-23 18:25:32,685 P15081 INFO Save best model: monitor(max): 0.980128
2022-01-23 18:25:32,689 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:25:32,731 P15081 INFO Train loss: 0.167589
2022-01-23 18:25:32,731 P15081 INFO ************ Epoch=28 end ************
2022-01-23 18:25:54,236 P15081 INFO [Metrics] AUC: 0.981547 - logloss: 0.160892
2022-01-23 18:25:54,237 P15081 INFO Save best model: monitor(max): 0.981547
2022-01-23 18:25:54,241 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:25:54,279 P15081 INFO Train loss: 0.139064
2022-01-23 18:25:54,279 P15081 INFO ************ Epoch=29 end ************
2022-01-23 18:26:16,172 P15081 INFO [Metrics] AUC: 0.981899 - logloss: 0.162492
2022-01-23 18:26:16,173 P15081 INFO Save best model: monitor(max): 0.981899
2022-01-23 18:26:16,177 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:26:16,216 P15081 INFO Train loss: 0.122369
2022-01-23 18:26:16,217 P15081 INFO ************ Epoch=30 end ************
2022-01-23 18:26:38,024 P15081 INFO [Metrics] AUC: 0.981968 - logloss: 0.166729
2022-01-23 18:26:38,024 P15081 INFO Save best model: monitor(max): 0.981968
2022-01-23 18:26:38,028 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:26:38,065 P15081 INFO Train loss: 0.110245
2022-01-23 18:26:38,065 P15081 INFO ************ Epoch=31 end ************
2022-01-23 18:26:59,991 P15081 INFO [Metrics] AUC: 0.982161 - logloss: 0.168569
2022-01-23 18:26:59,991 P15081 INFO Save best model: monitor(max): 0.982161
2022-01-23 18:26:59,995 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:27:00,033 P15081 INFO Train loss: 0.101333
2022-01-23 18:27:00,033 P15081 INFO ************ Epoch=32 end ************
2022-01-23 18:27:21,895 P15081 INFO [Metrics] AUC: 0.982189 - logloss: 0.172185
2022-01-23 18:27:21,895 P15081 INFO Save best model: monitor(max): 0.982189
2022-01-23 18:27:21,899 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:27:21,937 P15081 INFO Train loss: 0.093316
2022-01-23 18:27:21,938 P15081 INFO ************ Epoch=33 end ************
2022-01-23 18:27:43,545 P15081 INFO [Metrics] AUC: 0.982064 - logloss: 0.179178
2022-01-23 18:27:43,545 P15081 INFO Monitor(max) STOP: 0.982064 !
2022-01-23 18:27:43,545 P15081 INFO Reduce learning rate on plateau: 0.000010
2022-01-23 18:27:43,545 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:27:43,592 P15081 INFO Train loss: 0.086915
2022-01-23 18:27:43,592 P15081 INFO ************ Epoch=34 end ************
2022-01-23 18:28:05,170 P15081 INFO [Metrics] AUC: 0.982125 - logloss: 0.180170
2022-01-23 18:28:05,170 P15081 INFO Monitor(max) STOP: 0.982125 !
2022-01-23 18:28:05,170 P15081 INFO Reduce learning rate on plateau: 0.000001
2022-01-23 18:28:05,170 P15081 INFO Early stopping at epoch=35
2022-01-23 18:28:05,171 P15081 INFO --- 50/50 batches finished ---
2022-01-23 18:28:05,207 P15081 INFO Train loss: 0.078901
2022-01-23 18:28:05,207 P15081 INFO Training finished.
2022-01-23 18:28:05,207 P15081 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Frappe/AutoInt_frappe_x1/frappe_x1_04e961e9/AutoInt_frappe_x1_006_0eb83bd7.model
2022-01-23 18:28:11,334 P15081 INFO ****** Validation evaluation ******
2022-01-23 18:28:12,443 P15081 INFO [Metrics] AUC: 0.982189 - logloss: 0.172185
2022-01-23 18:28:12,478 P15081 INFO ******** Test evaluation ********
2022-01-23 18:28:12,478 P15081 INFO Loading data...
2022-01-23 18:28:12,479 P15081 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-01-23 18:28:12,481 P15081 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-01-23 18:28:12,481 P15081 INFO Loading test data done.
2022-01-23 18:28:13,181 P15081 INFO [Metrics] AUC: 0.982263 - logloss: 0.169137

```
