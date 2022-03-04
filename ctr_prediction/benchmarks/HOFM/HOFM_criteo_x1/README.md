## HOFM_criteo_x1

A hands-on guide to run the HOFM model on the Criteo_x1 dataset.

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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [HOFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HOFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HOFM_criteo_x1_tuner_config_02](./HOFM_criteo_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HOFM_criteo_x1
    nohup python run_expid.py --config ./HOFM_criteo_x1_tuner_config_02 --expid HOFM_criteo_x1_001_1b741ae5 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.805478 | 0.446054  |


### Logs
```python
2022-01-27 17:49:42,381 P52257 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HOFM",
    "model_id": "HOFM_criteo_x1_001_1b741ae5",
    "model_root": "./Criteo/HOFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "order": "3",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-06",
    "reuse_embedding": "False",
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
2022-01-27 17:49:42,382 P52257 INFO Set up feature encoder...
2022-01-27 17:49:42,382 P52257 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-27 17:49:42,382 P52257 INFO Loading data...
2022-01-27 17:49:42,385 P52257 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-27 17:49:47,240 P52257 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-27 17:49:48,426 P52257 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-27 17:49:48,426 P52257 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-27 17:49:48,427 P52257 INFO Loading train data done.
2022-01-27 17:49:53,136 P52257 INFO Total number of parameters: 43812637.
2022-01-27 17:49:53,136 P52257 INFO Start training: 8058 batches/epoch
2022-01-27 17:49:53,136 P52257 INFO ************ Epoch=1 start ************
2022-01-27 18:40:22,848 P52257 INFO [Metrics] AUC: 0.795663 - logloss: 0.454833
2022-01-27 18:40:22,850 P52257 INFO Save best model: monitor(max): 0.795663
2022-01-27 18:40:22,998 P52257 INFO --- 8058/8058 batches finished ---
2022-01-27 18:40:23,056 P52257 INFO Train loss: 0.470948
2022-01-27 18:40:23,057 P52257 INFO ************ Epoch=1 end ************
2022-01-27 19:31:19,229 P52257 INFO [Metrics] AUC: 0.797262 - logloss: 0.453413
2022-01-27 19:31:19,230 P52257 INFO Save best model: monitor(max): 0.797262
2022-01-27 19:31:19,514 P52257 INFO --- 8058/8058 batches finished ---
2022-01-27 19:31:19,562 P52257 INFO Train loss: 0.466213
2022-01-27 19:31:19,563 P52257 INFO ************ Epoch=2 end ************
2022-01-27 20:22:08,552 P52257 INFO [Metrics] AUC: 0.797674 - logloss: 0.453005
2022-01-27 20:22:08,554 P52257 INFO Save best model: monitor(max): 0.797674
2022-01-27 20:22:08,836 P52257 INFO --- 8058/8058 batches finished ---
2022-01-27 20:22:08,892 P52257 INFO Train loss: 0.465533
2022-01-27 20:22:08,892 P52257 INFO ************ Epoch=3 end ************
2022-01-27 21:13:01,909 P52257 INFO [Metrics] AUC: 0.797973 - logloss: 0.452770
2022-01-27 21:13:01,910 P52257 INFO Save best model: monitor(max): 0.797973
2022-01-27 21:13:02,200 P52257 INFO --- 8058/8058 batches finished ---
2022-01-27 21:13:02,250 P52257 INFO Train loss: 0.465288
2022-01-27 21:13:02,250 P52257 INFO ************ Epoch=4 end ************
2022-01-27 22:03:57,841 P52257 INFO [Metrics] AUC: 0.798171 - logloss: 0.452593
2022-01-27 22:03:57,842 P52257 INFO Save best model: monitor(max): 0.798171
2022-01-27 22:03:58,115 P52257 INFO --- 8058/8058 batches finished ---
2022-01-27 22:03:58,166 P52257 INFO Train loss: 0.465151
2022-01-27 22:03:58,166 P52257 INFO ************ Epoch=5 end ************
2022-01-27 22:54:59,518 P52257 INFO [Metrics] AUC: 0.798494 - logloss: 0.452284
2022-01-27 22:54:59,520 P52257 INFO Save best model: monitor(max): 0.798494
2022-01-27 22:54:59,801 P52257 INFO --- 8058/8058 batches finished ---
2022-01-27 22:54:59,851 P52257 INFO Train loss: 0.465096
2022-01-27 22:54:59,851 P52257 INFO ************ Epoch=6 end ************
2022-01-27 23:45:31,005 P52257 INFO [Metrics] AUC: 0.798548 - logloss: 0.452269
2022-01-27 23:45:31,006 P52257 INFO Save best model: monitor(max): 0.798548
2022-01-27 23:45:31,287 P52257 INFO --- 8058/8058 batches finished ---
2022-01-27 23:45:31,338 P52257 INFO Train loss: 0.465046
2022-01-27 23:45:31,338 P52257 INFO ************ Epoch=7 end ************
2022-01-28 00:36:04,246 P52257 INFO [Metrics] AUC: 0.798226 - logloss: 0.452523
2022-01-28 00:36:04,247 P52257 INFO Monitor(max) STOP: 0.798226 !
2022-01-28 00:36:04,248 P52257 INFO Reduce learning rate on plateau: 0.000100
2022-01-28 00:36:04,248 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 00:36:04,297 P52257 INFO Train loss: 0.464996
2022-01-28 00:36:04,297 P52257 INFO ************ Epoch=8 end ************
2022-01-28 01:26:39,340 P52257 INFO [Metrics] AUC: 0.802973 - logloss: 0.448287
2022-01-28 01:26:39,342 P52257 INFO Save best model: monitor(max): 0.802973
2022-01-28 01:26:39,621 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 01:26:39,669 P52257 INFO Train loss: 0.453068
2022-01-28 01:26:39,669 P52257 INFO ************ Epoch=9 end ************
2022-01-28 02:17:21,751 P52257 INFO [Metrics] AUC: 0.803719 - logloss: 0.447639
2022-01-28 02:17:21,753 P52257 INFO Save best model: monitor(max): 0.803719
2022-01-28 02:17:22,045 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 02:17:22,101 P52257 INFO Train loss: 0.449323
2022-01-28 02:17:22,101 P52257 INFO ************ Epoch=10 end ************
2022-01-28 03:08:03,954 P52257 INFO [Metrics] AUC: 0.804088 - logloss: 0.447318
2022-01-28 03:08:03,956 P52257 INFO Save best model: monitor(max): 0.804088
2022-01-28 03:08:04,245 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 03:08:04,294 P52257 INFO Train loss: 0.448259
2022-01-28 03:08:04,294 P52257 INFO ************ Epoch=11 end ************
2022-01-28 03:58:55,906 P52257 INFO [Metrics] AUC: 0.804340 - logloss: 0.447111
2022-01-28 03:58:55,908 P52257 INFO Save best model: monitor(max): 0.804340
2022-01-28 03:58:56,197 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 03:58:56,252 P52257 INFO Train loss: 0.447560
2022-01-28 03:58:56,252 P52257 INFO ************ Epoch=12 end ************
2022-01-28 04:49:53,116 P52257 INFO [Metrics] AUC: 0.804553 - logloss: 0.446950
2022-01-28 04:49:53,117 P52257 INFO Save best model: monitor(max): 0.804553
2022-01-28 04:49:53,405 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 04:49:53,454 P52257 INFO Train loss: 0.446996
2022-01-28 04:49:53,454 P52257 INFO ************ Epoch=13 end ************
2022-01-28 05:40:47,838 P52257 INFO [Metrics] AUC: 0.804669 - logloss: 0.446845
2022-01-28 05:40:47,839 P52257 INFO Save best model: monitor(max): 0.804669
2022-01-28 05:40:48,128 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 05:40:48,180 P52257 INFO Train loss: 0.446517
2022-01-28 05:40:48,180 P52257 INFO ************ Epoch=14 end ************
2022-01-28 06:31:29,817 P52257 INFO [Metrics] AUC: 0.804783 - logloss: 0.446745
2022-01-28 06:31:29,819 P52257 INFO Save best model: monitor(max): 0.804783
2022-01-28 06:31:30,106 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 06:31:30,159 P52257 INFO Train loss: 0.446082
2022-01-28 06:31:30,159 P52257 INFO ************ Epoch=15 end ************
2022-01-28 07:22:04,312 P52257 INFO [Metrics] AUC: 0.804838 - logloss: 0.446712
2022-01-28 07:22:04,313 P52257 INFO Save best model: monitor(max): 0.804838
2022-01-28 07:22:04,585 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 07:22:04,644 P52257 INFO Train loss: 0.445705
2022-01-28 07:22:04,644 P52257 INFO ************ Epoch=16 end ************
2022-01-28 08:12:56,696 P52257 INFO [Metrics] AUC: 0.804817 - logloss: 0.446733
2022-01-28 08:12:56,698 P52257 INFO Monitor(max) STOP: 0.804817 !
2022-01-28 08:12:56,698 P52257 INFO Reduce learning rate on plateau: 0.000010
2022-01-28 08:12:56,698 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 08:12:56,749 P52257 INFO Train loss: 0.445370
2022-01-28 08:12:56,749 P52257 INFO ************ Epoch=17 end ************
2022-01-28 09:03:55,876 P52257 INFO [Metrics] AUC: 0.805137 - logloss: 0.446467
2022-01-28 09:03:55,877 P52257 INFO Save best model: monitor(max): 0.805137
2022-01-28 09:03:56,172 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 09:03:56,221 P52257 INFO Train loss: 0.441536
2022-01-28 09:03:56,221 P52257 INFO ************ Epoch=18 end ************
2022-01-28 09:54:51,977 P52257 INFO [Metrics] AUC: 0.805166 - logloss: 0.446450
2022-01-28 09:54:51,979 P52257 INFO Save best model: monitor(max): 0.805166
2022-01-28 09:54:52,272 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 09:54:52,324 P52257 INFO Train loss: 0.441230
2022-01-28 09:54:52,324 P52257 INFO ************ Epoch=19 end ************
2022-01-28 10:45:31,131 P52257 INFO [Metrics] AUC: 0.805172 - logloss: 0.446455
2022-01-28 10:45:31,133 P52257 INFO Save best model: monitor(max): 0.805172
2022-01-28 10:45:31,410 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 10:45:31,460 P52257 INFO Train loss: 0.441069
2022-01-28 10:45:31,460 P52257 INFO ************ Epoch=20 end ************
2022-01-28 11:36:31,606 P52257 INFO [Metrics] AUC: 0.805170 - logloss: 0.446468
2022-01-28 11:36:31,608 P52257 INFO Monitor(max) STOP: 0.805170 !
2022-01-28 11:36:31,608 P52257 INFO Reduce learning rate on plateau: 0.000001
2022-01-28 11:36:31,608 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 11:36:31,658 P52257 INFO Train loss: 0.440943
2022-01-28 11:36:31,658 P52257 INFO ************ Epoch=21 end ************
2022-01-28 12:27:20,058 P52257 INFO [Metrics] AUC: 0.805172 - logloss: 0.446466
2022-01-28 12:27:20,059 P52257 INFO Monitor(max) STOP: 0.805172 !
2022-01-28 12:27:20,059 P52257 INFO Reduce learning rate on plateau: 0.000001
2022-01-28 12:27:20,059 P52257 INFO Early stopping at epoch=22
2022-01-28 12:27:20,059 P52257 INFO --- 8058/8058 batches finished ---
2022-01-28 12:27:20,109 P52257 INFO Train loss: 0.440329
2022-01-28 12:27:20,109 P52257 INFO Training finished.
2022-01-28 12:27:20,109 P52257 INFO Load best model: /home/XXX/FuxiCTR/benchmarks/Criteo/HOFM_criteo_x1/criteo_x1_7b681156/HOFM_criteo_x1_001_1b741ae5.model
2022-01-28 12:27:20,303 P52257 INFO ****** Validation evaluation ******
2022-01-28 12:31:59,881 P52257 INFO [Metrics] AUC: 0.805172 - logloss: 0.446455
2022-01-28 12:31:59,943 P52257 INFO ******** Test evaluation ********
2022-01-28 12:31:59,943 P52257 INFO Loading data...
2022-01-28 12:31:59,944 P52257 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-28 12:32:00,608 P52257 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-28 12:32:00,608 P52257 INFO Loading test data done.
2022-01-28 12:34:35,500 P52257 INFO [Metrics] AUC: 0.805478 - logloss: 0.446054

```
