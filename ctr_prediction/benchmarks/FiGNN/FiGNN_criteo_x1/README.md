## FiGNN_criteo_x1

A hands-on guide to run the FiGNN model on the Criteo_x1 dataset.

Author: [XUEPAI](https://github.com/xue-pai)

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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/README.md#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [FiGNN](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/FiGNN.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [FiGNN_criteo_x1_tuner_config_01](./FiGNN_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd FiGNN_criteo_x1
    nohup python run_expid.py --config ./FiGNN_criteo_x1_tuner_config_01 --expid FiGNN_criteo_x1_003_c6718583 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.813448 | 0.438525  |


### Logs
```python
2022-01-26 20:52:52,680 P24959 INFO {
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
    "gnn_layers": "6",
    "gpu": "2",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FiGNN",
    "model_id": "FiGNN_criteo_x1_003_c6718583",
    "model_root": "./Criteo/FiGNN_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reuse_graph_layer": "False",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_gru": "True",
    "use_hdf5": "True",
    "use_residual": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-26 20:52:52,681 P24959 INFO Set up feature encoder...
2022-01-26 20:52:52,681 P24959 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-26 20:52:52,681 P24959 INFO Loading data...
2022-01-26 20:52:52,683 P24959 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-26 20:52:57,840 P24959 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-26 20:52:59,208 P24959 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-26 20:52:59,209 P24959 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-26 20:52:59,209 P24959 INFO Loading train data done.
2022-01-26 20:53:05,455 P24959 INFO Total number of parameters: 20925920.
2022-01-26 20:53:05,455 P24959 INFO Start training: 8058 batches/epoch
2022-01-26 20:53:05,455 P24959 INFO ************ Epoch=1 start ************
2022-01-26 21:52:10,468 P24959 INFO [Metrics] AUC: 0.801635 - logloss: 0.449404
2022-01-26 21:52:10,470 P24959 INFO Save best model: monitor(max): 0.801635
2022-01-26 21:52:10,899 P24959 INFO --- 8058/8058 batches finished ---
2022-01-26 21:52:10,941 P24959 INFO Train loss: 0.467987
2022-01-26 21:52:10,941 P24959 INFO ************ Epoch=1 end ************
2022-01-26 22:51:18,729 P24959 INFO [Metrics] AUC: 0.804663 - logloss: 0.447404
2022-01-26 22:51:18,730 P24959 INFO Save best model: monitor(max): 0.804663
2022-01-26 22:51:18,822 P24959 INFO --- 8058/8058 batches finished ---
2022-01-26 22:51:18,873 P24959 INFO Train loss: 0.456734
2022-01-26 22:51:18,873 P24959 INFO ************ Epoch=2 end ************
2022-01-26 23:50:25,186 P24959 INFO [Metrics] AUC: 0.805923 - logloss: 0.445463
2022-01-26 23:50:25,187 P24959 INFO Save best model: monitor(max): 0.805923
2022-01-26 23:50:25,278 P24959 INFO --- 8058/8058 batches finished ---
2022-01-26 23:50:25,331 P24959 INFO Train loss: 0.454843
2022-01-26 23:50:25,331 P24959 INFO ************ Epoch=3 end ************
2022-01-27 00:49:30,512 P24959 INFO [Metrics] AUC: 0.806641 - logloss: 0.444796
2022-01-27 00:49:30,513 P24959 INFO Save best model: monitor(max): 0.806641
2022-01-27 00:49:30,606 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 00:49:30,655 P24959 INFO Train loss: 0.453880
2022-01-27 00:49:30,655 P24959 INFO ************ Epoch=4 end ************
2022-01-27 01:48:36,542 P24959 INFO [Metrics] AUC: 0.807202 - logloss: 0.444285
2022-01-27 01:48:36,544 P24959 INFO Save best model: monitor(max): 0.807202
2022-01-27 01:48:36,631 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 01:48:36,675 P24959 INFO Train loss: 0.453271
2022-01-27 01:48:36,675 P24959 INFO ************ Epoch=5 end ************
2022-01-27 02:47:41,052 P24959 INFO [Metrics] AUC: 0.807574 - logloss: 0.444071
2022-01-27 02:47:41,053 P24959 INFO Save best model: monitor(max): 0.807574
2022-01-27 02:47:41,169 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 02:47:41,213 P24959 INFO Train loss: 0.452800
2022-01-27 02:47:41,213 P24959 INFO ************ Epoch=6 end ************
2022-01-27 03:46:44,678 P24959 INFO [Metrics] AUC: 0.807786 - logloss: 0.443784
2022-01-27 03:46:44,680 P24959 INFO Save best model: monitor(max): 0.807786
2022-01-27 03:46:44,777 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 03:46:44,820 P24959 INFO Train loss: 0.452475
2022-01-27 03:46:44,820 P24959 INFO ************ Epoch=7 end ************
2022-01-27 04:45:47,448 P24959 INFO [Metrics] AUC: 0.808132 - logloss: 0.443627
2022-01-27 04:45:47,449 P24959 INFO Save best model: monitor(max): 0.808132
2022-01-27 04:45:47,536 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 04:45:47,579 P24959 INFO Train loss: 0.452160
2022-01-27 04:45:47,580 P24959 INFO ************ Epoch=8 end ************
2022-01-27 05:44:49,979 P24959 INFO [Metrics] AUC: 0.808292 - logloss: 0.443383
2022-01-27 05:44:49,980 P24959 INFO Save best model: monitor(max): 0.808292
2022-01-27 05:44:50,082 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 05:44:50,127 P24959 INFO Train loss: 0.451905
2022-01-27 05:44:50,127 P24959 INFO ************ Epoch=9 end ************
2022-01-27 06:43:51,248 P24959 INFO [Metrics] AUC: 0.808397 - logloss: 0.443148
2022-01-27 06:43:51,250 P24959 INFO Save best model: monitor(max): 0.808397
2022-01-27 06:43:51,337 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 06:43:51,385 P24959 INFO Train loss: 0.451705
2022-01-27 06:43:51,385 P24959 INFO ************ Epoch=10 end ************
2022-01-27 07:42:48,114 P24959 INFO [Metrics] AUC: 0.808575 - logloss: 0.443157
2022-01-27 07:42:48,116 P24959 INFO Save best model: monitor(max): 0.808575
2022-01-27 07:42:48,203 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 07:42:48,249 P24959 INFO Train loss: 0.451493
2022-01-27 07:42:48,249 P24959 INFO ************ Epoch=11 end ************
2022-01-27 08:41:41,506 P24959 INFO [Metrics] AUC: 0.808686 - logloss: 0.442840
2022-01-27 08:41:41,507 P24959 INFO Save best model: monitor(max): 0.808686
2022-01-27 08:41:41,605 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 08:41:41,649 P24959 INFO Train loss: 0.451343
2022-01-27 08:41:41,649 P24959 INFO ************ Epoch=12 end ************
2022-01-27 09:40:28,304 P24959 INFO [Metrics] AUC: 0.808914 - logloss: 0.442649
2022-01-27 09:40:28,306 P24959 INFO Save best model: monitor(max): 0.808914
2022-01-27 09:40:28,392 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 09:40:28,440 P24959 INFO Train loss: 0.451202
2022-01-27 09:40:28,440 P24959 INFO ************ Epoch=13 end ************
2022-01-27 10:39:10,532 P24959 INFO [Metrics] AUC: 0.809010 - logloss: 0.442614
2022-01-27 10:39:10,534 P24959 INFO Save best model: monitor(max): 0.809010
2022-01-27 10:39:10,629 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 10:39:10,685 P24959 INFO Train loss: 0.451065
2022-01-27 10:39:10,685 P24959 INFO ************ Epoch=14 end ************
2022-01-27 11:37:52,120 P24959 INFO [Metrics] AUC: 0.809047 - logloss: 0.442499
2022-01-27 11:37:52,121 P24959 INFO Save best model: monitor(max): 0.809047
2022-01-27 11:37:52,207 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 11:37:52,254 P24959 INFO Train loss: 0.450950
2022-01-27 11:37:52,254 P24959 INFO ************ Epoch=15 end ************
2022-01-27 12:36:31,841 P24959 INFO [Metrics] AUC: 0.809128 - logloss: 0.442503
2022-01-27 12:36:31,843 P24959 INFO Save best model: monitor(max): 0.809128
2022-01-27 12:36:31,938 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 12:36:31,984 P24959 INFO Train loss: 0.450843
2022-01-27 12:36:31,984 P24959 INFO ************ Epoch=16 end ************
2022-01-27 13:35:11,803 P24959 INFO [Metrics] AUC: 0.809291 - logloss: 0.442337
2022-01-27 13:35:11,805 P24959 INFO Save best model: monitor(max): 0.809291
2022-01-27 13:35:11,893 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 13:35:11,944 P24959 INFO Train loss: 0.450756
2022-01-27 13:35:11,944 P24959 INFO ************ Epoch=17 end ************
2022-01-27 14:33:44,735 P24959 INFO [Metrics] AUC: 0.809186 - logloss: 0.442461
2022-01-27 14:33:44,736 P24959 INFO Monitor(max) STOP: 0.809186 !
2022-01-27 14:33:44,736 P24959 INFO Reduce learning rate on plateau: 0.000100
2022-01-27 14:33:44,737 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 14:33:44,781 P24959 INFO Train loss: 0.450664
2022-01-27 14:33:44,781 P24959 INFO ************ Epoch=18 end ************
2022-01-27 15:32:17,523 P24959 INFO [Metrics] AUC: 0.812416 - logloss: 0.439459
2022-01-27 15:32:17,524 P24959 INFO Save best model: monitor(max): 0.812416
2022-01-27 15:32:17,612 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 15:32:17,661 P24959 INFO Train loss: 0.441246
2022-01-27 15:32:17,661 P24959 INFO ************ Epoch=19 end ************
2022-01-27 16:30:51,977 P24959 INFO [Metrics] AUC: 0.812903 - logloss: 0.439053
2022-01-27 16:30:51,978 P24959 INFO Save best model: monitor(max): 0.812903
2022-01-27 16:30:52,074 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 16:30:52,117 P24959 INFO Train loss: 0.437891
2022-01-27 16:30:52,118 P24959 INFO ************ Epoch=20 end ************
2022-01-27 17:29:25,505 P24959 INFO [Metrics] AUC: 0.813086 - logloss: 0.438916
2022-01-27 17:29:25,507 P24959 INFO Save best model: monitor(max): 0.813086
2022-01-27 17:29:25,611 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 17:29:25,654 P24959 INFO Train loss: 0.436421
2022-01-27 17:29:25,654 P24959 INFO ************ Epoch=21 end ************
2022-01-27 18:27:58,084 P24959 INFO [Metrics] AUC: 0.813100 - logloss: 0.438968
2022-01-27 18:27:58,085 P24959 INFO Save best model: monitor(max): 0.813100
2022-01-27 18:27:58,169 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 18:27:58,214 P24959 INFO Train loss: 0.435331
2022-01-27 18:27:58,214 P24959 INFO ************ Epoch=22 end ************
2022-01-27 19:26:31,278 P24959 INFO [Metrics] AUC: 0.812996 - logloss: 0.439167
2022-01-27 19:26:31,279 P24959 INFO Monitor(max) STOP: 0.812996 !
2022-01-27 19:26:31,279 P24959 INFO Reduce learning rate on plateau: 0.000010
2022-01-27 19:26:31,279 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 19:26:31,325 P24959 INFO Train loss: 0.434404
2022-01-27 19:26:31,326 P24959 INFO ************ Epoch=23 end ************
2022-01-27 20:25:06,856 P24959 INFO [Metrics] AUC: 0.812726 - logloss: 0.439563
2022-01-27 20:25:06,858 P24959 INFO Monitor(max) STOP: 0.812726 !
2022-01-27 20:25:06,858 P24959 INFO Reduce learning rate on plateau: 0.000001
2022-01-27 20:25:06,858 P24959 INFO Early stopping at epoch=24
2022-01-27 20:25:06,858 P24959 INFO --- 8058/8058 batches finished ---
2022-01-27 20:25:06,903 P24959 INFO Train loss: 0.430375
2022-01-27 20:25:06,903 P24959 INFO Training finished.
2022-01-27 20:25:06,903 P24959 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/FiGNN_criteo_x1/criteo_x1_7b681156/FiGNN_criteo_x1_003_c6718583.model
2022-01-27 20:25:09,686 P24959 INFO ****** Validation evaluation ******
2022-01-27 20:26:48,498 P24959 INFO [Metrics] AUC: 0.813100 - logloss: 0.438968
2022-01-27 20:26:48,578 P24959 INFO ******** Test evaluation ********
2022-01-27 20:26:48,579 P24959 INFO Loading data...
2022-01-27 20:26:48,579 P24959 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-27 20:26:49,383 P24959 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-27 20:26:49,384 P24959 INFO Loading test data done.
2022-01-27 20:27:35,687 P24959 INFO [Metrics] AUC: 0.813448 - logloss: 0.438525

```
