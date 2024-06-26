## WideDeep_criteo_x1

A hands-on guide to run the WideDeep model on the Criteo_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)

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
  fuxictr: 1.1.0

  ```

### Dataset
Dataset ID: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [WideDeep](https://github.com/reczoo/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/WideDeep.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [WideDeep_criteo_x1_tuner_config_01](./WideDeep_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd WideDeep_criteo_x1
    nohup python run_expid.py --config ./WideDeep_criteo_x1_tuner_config_01 --expid WideDeep_criteo_x1_010_3a06c5bc --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

Total 5 runs:

| Runs | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 1 | 0.813860 | 0.438004  |
| 2 | 0.813799 | 0.438034  |
| 3 | 0.813723 | 0.438149  |
| 4 | 0.813885 | 0.438042  |
| 5 | 0.813844 | 0.438000  |
| Avg | 0.813822 | 0.438046 |
| Std | &#177;0.00005697 | &#177;0.00005413 |


### Logs
```python
2022-02-06 13:56:43,046 P65771 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "WideDeep",
    "model_id": "WideDeep_criteo_x1_010_3a06c5bc",
    "model_root": "./Criteo/WideDeep_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-06 13:56:43,047 P65771 INFO Set up feature encoder...
2022-02-06 13:56:43,048 P65771 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-06 13:56:43,048 P65771 INFO Loading data...
2022-02-06 13:56:43,050 P65771 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-06 13:56:47,639 P65771 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-06 13:56:48,963 P65771 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-06 13:56:48,963 P65771 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-06 13:56:48,963 P65771 INFO Loading train data done.
2022-02-06 13:56:55,326 P65771 INFO Total number of parameters: 23429477.
2022-02-06 13:56:55,327 P65771 INFO Start training: 8058 batches/epoch
2022-02-06 13:56:55,327 P65771 INFO ************ Epoch=1 start ************
2022-02-06 14:19:03,262 P65771 INFO [Metrics] AUC: 0.803173 - logloss: 0.448142
2022-02-06 14:19:03,264 P65771 INFO Save best model: monitor(max): 0.803173
2022-02-06 14:19:03,534 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 14:19:03,580 P65771 INFO Train loss: 0.463532
2022-02-06 14:19:03,580 P65771 INFO ************ Epoch=1 end ************
2022-02-06 14:41:08,163 P65771 INFO [Metrics] AUC: 0.805219 - logloss: 0.446217
2022-02-06 14:41:08,164 P65771 INFO Save best model: monitor(max): 0.805219
2022-02-06 14:41:08,273 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 14:41:08,320 P65771 INFO Train loss: 0.457819
2022-02-06 14:41:08,321 P65771 INFO ************ Epoch=2 end ************
2022-02-06 15:03:14,543 P65771 INFO [Metrics] AUC: 0.806546 - logloss: 0.444872
2022-02-06 15:03:14,545 P65771 INFO Save best model: monitor(max): 0.806546
2022-02-06 15:03:14,659 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 15:03:14,708 P65771 INFO Train loss: 0.456272
2022-02-06 15:03:14,708 P65771 INFO ************ Epoch=3 end ************
2022-02-06 15:25:22,459 P65771 INFO [Metrics] AUC: 0.807157 - logloss: 0.444271
2022-02-06 15:25:22,460 P65771 INFO Save best model: monitor(max): 0.807157
2022-02-06 15:25:22,574 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 15:25:22,622 P65771 INFO Train loss: 0.455502
2022-02-06 15:25:22,622 P65771 INFO ************ Epoch=4 end ************
2022-02-06 15:47:30,767 P65771 INFO [Metrics] AUC: 0.807759 - logloss: 0.443797
2022-02-06 15:47:30,769 P65771 INFO Save best model: monitor(max): 0.807759
2022-02-06 15:47:30,880 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 15:47:30,926 P65771 INFO Train loss: 0.455013
2022-02-06 15:47:30,927 P65771 INFO ************ Epoch=5 end ************
2022-02-06 16:09:37,189 P65771 INFO [Metrics] AUC: 0.808040 - logloss: 0.443574
2022-02-06 16:09:37,190 P65771 INFO Save best model: monitor(max): 0.808040
2022-02-06 16:09:37,305 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 16:09:37,352 P65771 INFO Train loss: 0.454645
2022-02-06 16:09:37,353 P65771 INFO ************ Epoch=6 end ************
2022-02-06 16:31:44,023 P65771 INFO [Metrics] AUC: 0.808334 - logloss: 0.443154
2022-02-06 16:31:44,025 P65771 INFO Save best model: monitor(max): 0.808334
2022-02-06 16:31:44,147 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 16:31:44,195 P65771 INFO Train loss: 0.454360
2022-02-06 16:31:44,195 P65771 INFO ************ Epoch=7 end ************
2022-02-06 16:53:51,899 P65771 INFO [Metrics] AUC: 0.808611 - logloss: 0.443122
2022-02-06 16:53:51,901 P65771 INFO Save best model: monitor(max): 0.808611
2022-02-06 16:53:52,009 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 16:53:52,057 P65771 INFO Train loss: 0.454125
2022-02-06 16:53:52,057 P65771 INFO ************ Epoch=8 end ************
2022-02-06 17:15:59,047 P65771 INFO [Metrics] AUC: 0.808811 - logloss: 0.442728
2022-02-06 17:15:59,049 P65771 INFO Save best model: monitor(max): 0.808811
2022-02-06 17:15:59,164 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 17:15:59,214 P65771 INFO Train loss: 0.453964
2022-02-06 17:15:59,214 P65771 INFO ************ Epoch=9 end ************
2022-02-06 17:38:06,141 P65771 INFO [Metrics] AUC: 0.808898 - logloss: 0.442669
2022-02-06 17:38:06,142 P65771 INFO Save best model: monitor(max): 0.808898
2022-02-06 17:38:06,241 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 17:38:06,291 P65771 INFO Train loss: 0.453801
2022-02-06 17:38:06,291 P65771 INFO ************ Epoch=10 end ************
2022-02-06 18:00:12,346 P65771 INFO [Metrics] AUC: 0.808974 - logloss: 0.442536
2022-02-06 18:00:12,348 P65771 INFO Save best model: monitor(max): 0.808974
2022-02-06 18:00:12,459 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 18:00:12,508 P65771 INFO Train loss: 0.453665
2022-02-06 18:00:12,508 P65771 INFO ************ Epoch=11 end ************
2022-02-06 18:22:16,657 P65771 INFO [Metrics] AUC: 0.809164 - logloss: 0.442423
2022-02-06 18:22:16,658 P65771 INFO Save best model: monitor(max): 0.809164
2022-02-06 18:22:16,761 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 18:22:16,808 P65771 INFO Train loss: 0.453543
2022-02-06 18:22:16,808 P65771 INFO ************ Epoch=12 end ************
2022-02-06 18:44:22,184 P65771 INFO [Metrics] AUC: 0.809204 - logloss: 0.442375
2022-02-06 18:44:22,185 P65771 INFO Save best model: monitor(max): 0.809204
2022-02-06 18:44:22,294 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 18:44:22,340 P65771 INFO Train loss: 0.453427
2022-02-06 18:44:22,340 P65771 INFO ************ Epoch=13 end ************
2022-02-06 19:06:25,074 P65771 INFO [Metrics] AUC: 0.809235 - logloss: 0.442334
2022-02-06 19:06:25,076 P65771 INFO Save best model: monitor(max): 0.809235
2022-02-06 19:06:25,202 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 19:06:25,247 P65771 INFO Train loss: 0.453324
2022-02-06 19:06:25,248 P65771 INFO ************ Epoch=14 end ************
2022-02-06 19:28:26,504 P65771 INFO [Metrics] AUC: 0.809377 - logloss: 0.442226
2022-02-06 19:28:26,506 P65771 INFO Save best model: monitor(max): 0.809377
2022-02-06 19:28:26,617 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 19:28:26,663 P65771 INFO Train loss: 0.453203
2022-02-06 19:28:26,663 P65771 INFO ************ Epoch=15 end ************
2022-02-06 19:50:28,210 P65771 INFO [Metrics] AUC: 0.809391 - logloss: 0.442148
2022-02-06 19:50:28,212 P65771 INFO Save best model: monitor(max): 0.809391
2022-02-06 19:50:28,314 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 19:50:28,360 P65771 INFO Train loss: 0.453115
2022-02-06 19:50:28,360 P65771 INFO ************ Epoch=16 end ************
2022-02-06 20:12:24,511 P65771 INFO [Metrics] AUC: 0.809529 - logloss: 0.442108
2022-02-06 20:12:24,513 P65771 INFO Save best model: monitor(max): 0.809529
2022-02-06 20:12:24,627 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 20:12:24,674 P65771 INFO Train loss: 0.453034
2022-02-06 20:12:24,675 P65771 INFO ************ Epoch=17 end ************
2022-02-06 20:34:17,815 P65771 INFO [Metrics] AUC: 0.809580 - logloss: 0.442041
2022-02-06 20:34:17,817 P65771 INFO Save best model: monitor(max): 0.809580
2022-02-06 20:34:17,928 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 20:34:17,978 P65771 INFO Train loss: 0.452960
2022-02-06 20:34:17,978 P65771 INFO ************ Epoch=18 end ************
2022-02-06 20:56:10,148 P65771 INFO [Metrics] AUC: 0.809652 - logloss: 0.441948
2022-02-06 20:56:10,150 P65771 INFO Save best model: monitor(max): 0.809652
2022-02-06 20:56:10,249 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 20:56:10,308 P65771 INFO Train loss: 0.452892
2022-02-06 20:56:10,308 P65771 INFO ************ Epoch=19 end ************
2022-02-06 21:17:56,398 P65771 INFO [Metrics] AUC: 0.809605 - logloss: 0.442073
2022-02-06 21:17:56,399 P65771 INFO Monitor(max) STOP: 0.809605 !
2022-02-06 21:17:56,399 P65771 INFO Reduce learning rate on plateau: 0.000100
2022-02-06 21:17:56,399 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 21:17:56,446 P65771 INFO Train loss: 0.452850
2022-02-06 21:17:56,446 P65771 INFO ************ Epoch=20 end ************
2022-02-06 21:39:42,034 P65771 INFO [Metrics] AUC: 0.812995 - logloss: 0.438890
2022-02-06 21:39:42,035 P65771 INFO Save best model: monitor(max): 0.812995
2022-02-06 21:39:42,143 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 21:39:42,190 P65771 INFO Train loss: 0.441688
2022-02-06 21:39:42,190 P65771 INFO ************ Epoch=21 end ************
2022-02-06 22:01:26,728 P65771 INFO [Metrics] AUC: 0.813427 - logloss: 0.438492
2022-02-06 22:01:26,730 P65771 INFO Save best model: monitor(max): 0.813427
2022-02-06 22:01:26,828 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 22:01:26,874 P65771 INFO Train loss: 0.437351
2022-02-06 22:01:26,875 P65771 INFO ************ Epoch=22 end ************
2022-02-06 22:23:14,286 P65771 INFO [Metrics] AUC: 0.813510 - logloss: 0.438481
2022-02-06 22:23:14,288 P65771 INFO Save best model: monitor(max): 0.813510
2022-02-06 22:23:14,412 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 22:23:14,458 P65771 INFO Train loss: 0.435519
2022-02-06 22:23:14,459 P65771 INFO ************ Epoch=23 end ************
2022-02-06 22:44:57,253 P65771 INFO [Metrics] AUC: 0.813377 - logloss: 0.438634
2022-02-06 22:44:57,254 P65771 INFO Monitor(max) STOP: 0.813377 !
2022-02-06 22:44:57,254 P65771 INFO Reduce learning rate on plateau: 0.000010
2022-02-06 22:44:57,254 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 22:44:57,300 P65771 INFO Train loss: 0.434160
2022-02-06 22:44:57,301 P65771 INFO ************ Epoch=24 end ************
2022-02-06 23:06:37,055 P65771 INFO [Metrics] AUC: 0.813050 - logloss: 0.439233
2022-02-06 23:06:37,057 P65771 INFO Monitor(max) STOP: 0.813050 !
2022-02-06 23:06:37,057 P65771 INFO Reduce learning rate on plateau: 0.000001
2022-02-06 23:06:37,057 P65771 INFO Early stopping at epoch=25
2022-02-06 23:06:37,057 P65771 INFO --- 8058/8058 batches finished ---
2022-02-06 23:06:37,104 P65771 INFO Train loss: 0.430021
2022-02-06 23:06:37,105 P65771 INFO Training finished.
2022-02-06 23:06:37,105 P65771 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/WideDeep_criteo_x1/criteo_x1_7b681156/WideDeep_criteo_x1_010_3a06c5bc.model
2022-02-06 23:06:40,306 P65771 INFO ****** Validation evaluation ******
2022-02-06 23:07:09,856 P65771 INFO [Metrics] AUC: 0.813510 - logloss: 0.438481
2022-02-06 23:07:09,938 P65771 INFO ******** Test evaluation ********
2022-02-06 23:07:09,938 P65771 INFO Loading data...
2022-02-06 23:07:09,939 P65771 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-06 23:07:10,753 P65771 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-06 23:07:10,754 P65771 INFO Loading test data done.
2022-02-06 23:07:27,833 P65771 INFO [Metrics] AUC: 0.813860 - logloss: 0.438004

```
