## WideDeep_frappe_x1

A hands-on guide to run the WideDeep model on the Frappe_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [WideDeep](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Frappe/Frappe_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_frappe_x1_tuner_config_01](./WideDeep_frappe_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_frappe_x1
    nohup python run_expid.py --config ./WideDeep_frappe_x1_tuner_config_01 --expid WideDeep_frappe_x1_030_af559975 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.984124 | 0.148974  |
| 2 | 0.983167 | 0.153814  |
| 3 | 0.981936 | 0.158301  |
| 4 | 0.982485 | 0.152619  |
| 5 | 0.983655 | 0.156184  |
| Avg | 0.983073 | 0.153978 |
| Std | &#177;0.00078622 | &#177;0.00317706 |


### Logs
```python
2022-02-07 18:21:45,490 P32887 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Frappe/",
    "dataset_id": "frappe_x1_04e961e9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.1",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['user', 'item', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "WideDeep",
    "model_id": "WideDeep_frappe_x1_030_af559975",
    "model_root": "./Frappe/WideDeep_frappe_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.6",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
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
2022-02-07 18:21:45,491 P32887 INFO Set up feature encoder...
2022-02-07 18:21:45,491 P32887 INFO Load feature_map from json: ../data/Frappe/frappe_x1_04e961e9/feature_map.json
2022-02-07 18:21:45,496 P32887 INFO Loading data...
2022-02-07 18:21:45,500 P32887 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/train.h5
2022-02-07 18:21:45,557 P32887 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/valid.h5
2022-02-07 18:21:45,582 P32887 INFO Train samples: total/202027, pos/67604, neg/134423, ratio/33.46%, blocks/1
2022-02-07 18:21:45,582 P32887 INFO Validation samples: total/57722, pos/19063, neg/38659, ratio/33.03%, blocks/1
2022-02-07 18:21:45,582 P32887 INFO Loading train data done.
2022-02-07 18:21:56,518 P32887 INFO Total number of parameters: 423280.
2022-02-07 18:21:56,522 P32887 INFO Start training: 50 batches/epoch
2022-02-07 18:21:56,522 P32887 INFO ************ Epoch=1 start ************
2022-02-07 18:22:07,256 P32887 INFO [Metrics] AUC: 0.930784 - logloss: 0.567257
2022-02-07 18:22:07,256 P32887 INFO Save best model: monitor(max): 0.930784
2022-02-07 18:22:07,273 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:22:07,458 P32887 INFO Train loss: 0.476120
2022-02-07 18:22:07,458 P32887 INFO ************ Epoch=1 end ************
2022-02-07 18:22:14,894 P32887 INFO [Metrics] AUC: 0.936067 - logloss: 0.298284
2022-02-07 18:22:14,895 P32887 INFO Save best model: monitor(max): 0.936067
2022-02-07 18:22:14,901 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:22:14,995 P32887 INFO Train loss: 0.326179
2022-02-07 18:22:14,996 P32887 INFO ************ Epoch=2 end ************
2022-02-07 18:22:23,284 P32887 INFO [Metrics] AUC: 0.942708 - logloss: 0.274715
2022-02-07 18:22:23,285 P32887 INFO Save best model: monitor(max): 0.942708
2022-02-07 18:22:23,293 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:22:23,459 P32887 INFO Train loss: 0.312202
2022-02-07 18:22:23,459 P32887 INFO ************ Epoch=3 end ************
2022-02-07 18:22:31,528 P32887 INFO [Metrics] AUC: 0.952202 - logloss: 0.248667
2022-02-07 18:22:31,528 P32887 INFO Save best model: monitor(max): 0.952202
2022-02-07 18:22:31,535 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:22:31,654 P32887 INFO Train loss: 0.298374
2022-02-07 18:22:31,654 P32887 INFO ************ Epoch=4 end ************
2022-02-07 18:22:39,997 P32887 INFO [Metrics] AUC: 0.960572 - logloss: 0.234939
2022-02-07 18:22:39,997 P32887 INFO Save best model: monitor(max): 0.960572
2022-02-07 18:22:40,017 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:22:40,240 P32887 INFO Train loss: 0.283117
2022-02-07 18:22:40,241 P32887 INFO ************ Epoch=5 end ************
2022-02-07 18:22:48,512 P32887 INFO [Metrics] AUC: 0.964619 - logloss: 0.255946
2022-02-07 18:22:48,513 P32887 INFO Save best model: monitor(max): 0.964619
2022-02-07 18:22:48,519 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:22:48,598 P32887 INFO Train loss: 0.273553
2022-02-07 18:22:48,598 P32887 INFO ************ Epoch=6 end ************
2022-02-07 18:22:55,236 P32887 INFO [Metrics] AUC: 0.967321 - logloss: 0.256327
2022-02-07 18:22:55,237 P32887 INFO Save best model: monitor(max): 0.967321
2022-02-07 18:22:55,244 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:22:55,337 P32887 INFO Train loss: 0.264835
2022-02-07 18:22:55,337 P32887 INFO ************ Epoch=7 end ************
2022-02-07 18:23:01,900 P32887 INFO [Metrics] AUC: 0.970187 - logloss: 0.202221
2022-02-07 18:23:01,901 P32887 INFO Save best model: monitor(max): 0.970187
2022-02-07 18:23:01,908 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:02,009 P32887 INFO Train loss: 0.260115
2022-02-07 18:23:02,010 P32887 INFO ************ Epoch=8 end ************
2022-02-07 18:23:08,446 P32887 INFO [Metrics] AUC: 0.971985 - logloss: 0.212591
2022-02-07 18:23:08,447 P32887 INFO Save best model: monitor(max): 0.971985
2022-02-07 18:23:08,455 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:08,535 P32887 INFO Train loss: 0.253558
2022-02-07 18:23:08,535 P32887 INFO ************ Epoch=9 end ************
2022-02-07 18:23:14,493 P32887 INFO [Metrics] AUC: 0.973697 - logloss: 0.191398
2022-02-07 18:23:14,493 P32887 INFO Save best model: monitor(max): 0.973697
2022-02-07 18:23:14,504 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:14,583 P32887 INFO Train loss: 0.251684
2022-02-07 18:23:14,583 P32887 INFO ************ Epoch=10 end ************
2022-02-07 18:23:20,912 P32887 INFO [Metrics] AUC: 0.973132 - logloss: 0.207505
2022-02-07 18:23:20,912 P32887 INFO Monitor(max) STOP: 0.973132 !
2022-02-07 18:23:20,912 P32887 INFO Reduce learning rate on plateau: 0.000100
2022-02-07 18:23:20,912 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:21,002 P32887 INFO Train loss: 0.246450
2022-02-07 18:23:21,002 P32887 INFO ************ Epoch=11 end ************
2022-02-07 18:23:26,304 P32887 INFO [Metrics] AUC: 0.980778 - logloss: 0.152452
2022-02-07 18:23:26,304 P32887 INFO Save best model: monitor(max): 0.980778
2022-02-07 18:23:26,312 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:26,395 P32887 INFO Train loss: 0.209377
2022-02-07 18:23:26,395 P32887 INFO ************ Epoch=12 end ************
2022-02-07 18:23:31,173 P32887 INFO [Metrics] AUC: 0.982831 - logloss: 0.144228
2022-02-07 18:23:31,174 P32887 INFO Save best model: monitor(max): 0.982831
2022-02-07 18:23:31,181 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:31,261 P32887 INFO Train loss: 0.174074
2022-02-07 18:23:31,261 P32887 INFO ************ Epoch=13 end ************
2022-02-07 18:23:36,145 P32887 INFO [Metrics] AUC: 0.983853 - logloss: 0.141533
2022-02-07 18:23:36,146 P32887 INFO Save best model: monitor(max): 0.983853
2022-02-07 18:23:36,153 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:36,222 P32887 INFO Train loss: 0.152204
2022-02-07 18:23:36,223 P32887 INFO ************ Epoch=14 end ************
2022-02-07 18:23:40,278 P32887 INFO [Metrics] AUC: 0.984393 - logloss: 0.140579
2022-02-07 18:23:40,279 P32887 INFO Save best model: monitor(max): 0.984393
2022-02-07 18:23:40,286 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:40,373 P32887 INFO Train loss: 0.137278
2022-02-07 18:23:40,374 P32887 INFO ************ Epoch=15 end ************
2022-02-07 18:23:43,800 P32887 INFO [Metrics] AUC: 0.984481 - logloss: 0.142208
2022-02-07 18:23:43,814 P32887 INFO Save best model: monitor(max): 0.984481
2022-02-07 18:23:43,821 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:43,904 P32887 INFO Train loss: 0.127773
2022-02-07 18:23:43,904 P32887 INFO ************ Epoch=16 end ************
2022-02-07 18:23:48,001 P32887 INFO [Metrics] AUC: 0.984750 - logloss: 0.141459
2022-02-07 18:23:48,002 P32887 INFO Save best model: monitor(max): 0.984750
2022-02-07 18:23:48,008 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:48,097 P32887 INFO Train loss: 0.120077
2022-02-07 18:23:48,098 P32887 INFO ************ Epoch=17 end ************
2022-02-07 18:23:51,884 P32887 INFO [Metrics] AUC: 0.984734 - logloss: 0.142879
2022-02-07 18:23:51,884 P32887 INFO Monitor(max) STOP: 0.984734 !
2022-02-07 18:23:51,885 P32887 INFO Reduce learning rate on plateau: 0.000010
2022-02-07 18:23:51,885 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:51,954 P32887 INFO Train loss: 0.114382
2022-02-07 18:23:51,954 P32887 INFO ************ Epoch=18 end ************
2022-02-07 18:23:55,770 P32887 INFO [Metrics] AUC: 0.984797 - logloss: 0.142763
2022-02-07 18:23:55,770 P32887 INFO Save best model: monitor(max): 0.984797
2022-02-07 18:23:55,777 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:55,887 P32887 INFO Train loss: 0.104554
2022-02-07 18:23:55,887 P32887 INFO ************ Epoch=19 end ************
2022-02-07 18:23:59,822 P32887 INFO [Metrics] AUC: 0.984917 - logloss: 0.142174
2022-02-07 18:23:59,822 P32887 INFO Save best model: monitor(max): 0.984917
2022-02-07 18:23:59,829 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:23:59,888 P32887 INFO Train loss: 0.102806
2022-02-07 18:23:59,888 P32887 INFO ************ Epoch=20 end ************
2022-02-07 18:24:03,760 P32887 INFO [Metrics] AUC: 0.985025 - logloss: 0.142019
2022-02-07 18:24:03,761 P32887 INFO Save best model: monitor(max): 0.985025
2022-02-07 18:24:03,768 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:03,830 P32887 INFO Train loss: 0.101527
2022-02-07 18:24:03,830 P32887 INFO ************ Epoch=21 end ************
2022-02-07 18:24:07,318 P32887 INFO [Metrics] AUC: 0.985033 - logloss: 0.141957
2022-02-07 18:24:07,319 P32887 INFO Save best model: monitor(max): 0.985033
2022-02-07 18:24:07,325 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:07,414 P32887 INFO Train loss: 0.099771
2022-02-07 18:24:07,414 P32887 INFO ************ Epoch=22 end ************
2022-02-07 18:24:11,285 P32887 INFO [Metrics] AUC: 0.985073 - logloss: 0.141955
2022-02-07 18:24:11,285 P32887 INFO Save best model: monitor(max): 0.985073
2022-02-07 18:24:11,292 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:11,380 P32887 INFO Train loss: 0.099187
2022-02-07 18:24:11,380 P32887 INFO ************ Epoch=23 end ************
2022-02-07 18:24:15,263 P32887 INFO [Metrics] AUC: 0.985079 - logloss: 0.142229
2022-02-07 18:24:15,264 P32887 INFO Save best model: monitor(max): 0.985079
2022-02-07 18:24:15,271 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:15,349 P32887 INFO Train loss: 0.097205
2022-02-07 18:24:15,349 P32887 INFO ************ Epoch=24 end ************
2022-02-07 18:24:19,205 P32887 INFO [Metrics] AUC: 0.985163 - logloss: 0.142057
2022-02-07 18:24:19,206 P32887 INFO Save best model: monitor(max): 0.985163
2022-02-07 18:24:19,212 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:19,297 P32887 INFO Train loss: 0.096755
2022-02-07 18:24:19,297 P32887 INFO ************ Epoch=25 end ************
2022-02-07 18:24:23,204 P32887 INFO [Metrics] AUC: 0.985225 - logloss: 0.141775
2022-02-07 18:24:23,204 P32887 INFO Save best model: monitor(max): 0.985225
2022-02-07 18:24:23,211 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:23,275 P32887 INFO Train loss: 0.096224
2022-02-07 18:24:23,275 P32887 INFO ************ Epoch=26 end ************
2022-02-07 18:24:27,056 P32887 INFO [Metrics] AUC: 0.985270 - logloss: 0.141953
2022-02-07 18:24:27,057 P32887 INFO Save best model: monitor(max): 0.985270
2022-02-07 18:24:27,064 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:27,159 P32887 INFO Train loss: 0.095850
2022-02-07 18:24:27,160 P32887 INFO ************ Epoch=27 end ************
2022-02-07 18:24:30,967 P32887 INFO [Metrics] AUC: 0.985215 - logloss: 0.142591
2022-02-07 18:24:30,968 P32887 INFO Monitor(max) STOP: 0.985215 !
2022-02-07 18:24:30,968 P32887 INFO Reduce learning rate on plateau: 0.000001
2022-02-07 18:24:30,968 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:31,031 P32887 INFO Train loss: 0.093966
2022-02-07 18:24:31,031 P32887 INFO ************ Epoch=28 end ************
2022-02-07 18:24:34,876 P32887 INFO [Metrics] AUC: 0.985238 - logloss: 0.142498
2022-02-07 18:24:34,876 P32887 INFO Monitor(max) STOP: 0.985238 !
2022-02-07 18:24:34,877 P32887 INFO Reduce learning rate on plateau: 0.000001
2022-02-07 18:24:34,877 P32887 INFO Early stopping at epoch=29
2022-02-07 18:24:34,877 P32887 INFO --- 50/50 batches finished ---
2022-02-07 18:24:34,971 P32887 INFO Train loss: 0.093920
2022-02-07 18:24:34,971 P32887 INFO Training finished.
2022-02-07 18:24:34,971 P32887 INFO Load best model: /home/XXX/FuxiCTR_github/benchmarks/Frappe/WideDeep_frappe_x1/frappe_x1_04e961e9/WideDeep_frappe_x1_030_af559975.model
2022-02-07 18:24:34,982 P32887 INFO ****** Validation evaluation ******
2022-02-07 18:24:35,663 P32887 INFO [Metrics] AUC: 0.985270 - logloss: 0.141953
2022-02-07 18:24:35,712 P32887 INFO ******** Test evaluation ********
2022-02-07 18:24:35,713 P32887 INFO Loading data...
2022-02-07 18:24:35,713 P32887 INFO Loading data from h5: ../data/Frappe/frappe_x1_04e961e9/test.h5
2022-02-07 18:24:35,717 P32887 INFO Test samples: total/28860, pos/9536, neg/19324, ratio/33.04%, blocks/1
2022-02-07 18:24:35,717 P32887 INFO Loading test data done.
2022-02-07 18:24:36,150 P32887 INFO [Metrics] AUC: 0.984124 - logloss: 0.148974

```
