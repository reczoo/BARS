## HFM+_criteo_x1

A hands-on guide to run the HFM model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_criteo_x1_tuner_config_07](./HFM+_criteo_x1_tuner_config_07). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_criteo_x1
    nohup python run_expid.py --config ./HFM+_criteo_x1_tuner_config_07 --expid HFM_criteo_x1_007_e8226627 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.812430 | 0.439360  |


### Logs
```python
2022-02-01 19:10:48,024 P86662 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_regularizer": "0.0005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "6",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HFM",
    "model_id": "HFM_criteo_x1_007_e8226627",
    "model_root": "./Criteo/HFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.5",
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
    "use_dnn": "True",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-02-01 19:10:48,025 P86662 INFO Set up feature encoder...
2022-02-01 19:10:48,025 P86662 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-02-01 19:10:48,025 P86662 INFO Loading data...
2022-02-01 19:10:48,026 P86662 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-02-01 19:10:52,412 P86662 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-02-01 19:10:53,560 P86662 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-02-01 19:10:53,560 P86662 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-02-01 19:10:53,560 P86662 INFO Loading train data done.
2022-02-01 19:10:58,482 P86662 INFO Total number of parameters: 26237478.
2022-02-01 19:10:58,482 P86662 INFO Start training: 8058 batches/epoch
2022-02-01 19:10:58,482 P86662 INFO ************ Epoch=1 start ************
2022-02-01 19:31:05,568 P86662 INFO [Metrics] AUC: 0.794639 - logloss: 0.455469
2022-02-01 19:31:05,570 P86662 INFO Save best model: monitor(max): 0.794639
2022-02-01 19:31:05,878 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 19:31:05,918 P86662 INFO Train loss: 0.489341
2022-02-01 19:31:05,918 P86662 INFO ************ Epoch=1 end ************
2022-02-01 19:51:15,358 P86662 INFO [Metrics] AUC: 0.796517 - logloss: 0.454070
2022-02-01 19:51:15,360 P86662 INFO Save best model: monitor(max): 0.796517
2022-02-01 19:51:15,474 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 19:51:15,515 P86662 INFO Train loss: 0.484538
2022-02-01 19:51:15,515 P86662 INFO ************ Epoch=2 end ************
2022-02-01 20:11:24,777 P86662 INFO [Metrics] AUC: 0.797262 - logloss: 0.453045
2022-02-01 20:11:24,779 P86662 INFO Save best model: monitor(max): 0.797262
2022-02-01 20:11:24,905 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 20:11:24,951 P86662 INFO Train loss: 0.483942
2022-02-01 20:11:24,952 P86662 INFO ************ Epoch=3 end ************
2022-02-01 20:31:39,148 P86662 INFO [Metrics] AUC: 0.797762 - logloss: 0.452672
2022-02-01 20:31:39,150 P86662 INFO Save best model: monitor(max): 0.797762
2022-02-01 20:31:39,272 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 20:31:39,326 P86662 INFO Train loss: 0.483692
2022-02-01 20:31:39,326 P86662 INFO ************ Epoch=4 end ************
2022-02-01 20:51:53,050 P86662 INFO [Metrics] AUC: 0.798305 - logloss: 0.452121
2022-02-01 20:51:53,052 P86662 INFO Save best model: monitor(max): 0.798305
2022-02-01 20:51:53,166 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 20:51:53,215 P86662 INFO Train loss: 0.483561
2022-02-01 20:51:53,215 P86662 INFO ************ Epoch=5 end ************
2022-02-01 21:12:01,631 P86662 INFO [Metrics] AUC: 0.798650 - logloss: 0.451848
2022-02-01 21:12:01,632 P86662 INFO Save best model: monitor(max): 0.798650
2022-02-01 21:12:01,740 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 21:12:01,792 P86662 INFO Train loss: 0.483473
2022-02-01 21:12:01,792 P86662 INFO ************ Epoch=6 end ************
2022-02-01 21:32:11,353 P86662 INFO [Metrics] AUC: 0.799181 - logloss: 0.451471
2022-02-01 21:32:11,355 P86662 INFO Save best model: monitor(max): 0.799181
2022-02-01 21:32:11,481 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 21:32:11,535 P86662 INFO Train loss: 0.483404
2022-02-01 21:32:11,536 P86662 INFO ************ Epoch=7 end ************
2022-02-01 21:52:19,781 P86662 INFO [Metrics] AUC: 0.799160 - logloss: 0.451305
2022-02-01 21:52:19,783 P86662 INFO Monitor(max) STOP: 0.799160 !
2022-02-01 21:52:19,783 P86662 INFO Reduce learning rate on plateau: 0.000100
2022-02-01 21:52:19,783 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 21:52:19,838 P86662 INFO Train loss: 0.483391
2022-02-01 21:52:19,838 P86662 INFO ************ Epoch=8 end ************
2022-02-01 22:12:29,229 P86662 INFO [Metrics] AUC: 0.808208 - logloss: 0.443329
2022-02-01 22:12:29,230 P86662 INFO Save best model: monitor(max): 0.808208
2022-02-01 22:12:29,351 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 22:12:29,404 P86662 INFO Train loss: 0.454463
2022-02-01 22:12:29,404 P86662 INFO ************ Epoch=9 end ************
2022-02-01 22:32:43,068 P86662 INFO [Metrics] AUC: 0.809129 - logloss: 0.442520
2022-02-01 22:32:43,069 P86662 INFO Save best model: monitor(max): 0.809129
2022-02-01 22:32:43,193 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 22:32:43,250 P86662 INFO Train loss: 0.451073
2022-02-01 22:32:43,251 P86662 INFO ************ Epoch=10 end ************
2022-02-01 22:52:57,507 P86662 INFO [Metrics] AUC: 0.809284 - logloss: 0.442353
2022-02-01 22:52:57,508 P86662 INFO Save best model: monitor(max): 0.809284
2022-02-01 22:52:57,632 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 22:52:57,692 P86662 INFO Train loss: 0.450666
2022-02-01 22:52:57,692 P86662 INFO ************ Epoch=11 end ************
2022-02-01 23:13:06,542 P86662 INFO [Metrics] AUC: 0.809409 - logloss: 0.442229
2022-02-01 23:13:06,544 P86662 INFO Save best model: monitor(max): 0.809409
2022-02-01 23:13:06,675 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 23:13:06,729 P86662 INFO Train loss: 0.450549
2022-02-01 23:13:06,729 P86662 INFO ************ Epoch=12 end ************
2022-02-01 23:33:14,140 P86662 INFO [Metrics] AUC: 0.809455 - logloss: 0.442223
2022-02-01 23:33:14,142 P86662 INFO Save best model: monitor(max): 0.809455
2022-02-01 23:33:14,263 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 23:33:14,321 P86662 INFO Train loss: 0.450436
2022-02-01 23:33:14,321 P86662 INFO ************ Epoch=13 end ************
2022-02-01 23:53:22,581 P86662 INFO [Metrics] AUC: 0.809543 - logloss: 0.442127
2022-02-01 23:53:22,582 P86662 INFO Save best model: monitor(max): 0.809543
2022-02-01 23:53:22,722 P86662 INFO --- 8058/8058 batches finished ---
2022-02-01 23:53:22,780 P86662 INFO Train loss: 0.450323
2022-02-01 23:53:22,781 P86662 INFO ************ Epoch=14 end ************
2022-02-02 00:13:32,428 P86662 INFO [Metrics] AUC: 0.809671 - logloss: 0.442000
2022-02-02 00:13:32,429 P86662 INFO Save best model: monitor(max): 0.809671
2022-02-02 00:13:32,553 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 00:13:32,605 P86662 INFO Train loss: 0.450243
2022-02-02 00:13:32,606 P86662 INFO ************ Epoch=15 end ************
2022-02-02 00:33:44,361 P86662 INFO [Metrics] AUC: 0.809625 - logloss: 0.442026
2022-02-02 00:33:44,363 P86662 INFO Monitor(max) STOP: 0.809625 !
2022-02-02 00:33:44,363 P86662 INFO Reduce learning rate on plateau: 0.000010
2022-02-02 00:33:44,363 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 00:33:44,412 P86662 INFO Train loss: 0.450144
2022-02-02 00:33:44,412 P86662 INFO ************ Epoch=16 end ************
2022-02-02 00:53:57,117 P86662 INFO [Metrics] AUC: 0.811450 - logloss: 0.440404
2022-02-02 00:53:57,119 P86662 INFO Save best model: monitor(max): 0.811450
2022-02-02 00:53:57,228 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 00:53:57,277 P86662 INFO Train loss: 0.442419
2022-02-02 00:53:57,278 P86662 INFO ************ Epoch=17 end ************
2022-02-02 01:14:06,743 P86662 INFO [Metrics] AUC: 0.811857 - logloss: 0.440041
2022-02-02 01:14:06,745 P86662 INFO Save best model: monitor(max): 0.811857
2022-02-02 01:14:06,869 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 01:14:06,909 P86662 INFO Train loss: 0.439660
2022-02-02 01:14:06,910 P86662 INFO ************ Epoch=18 end ************
2022-02-02 01:34:23,981 P86662 INFO [Metrics] AUC: 0.812050 - logloss: 0.439879
2022-02-02 01:34:23,982 P86662 INFO Save best model: monitor(max): 0.812050
2022-02-02 01:34:24,104 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 01:34:24,157 P86662 INFO Train loss: 0.438527
2022-02-02 01:34:24,157 P86662 INFO ************ Epoch=19 end ************
2022-02-02 01:54:36,577 P86662 INFO [Metrics] AUC: 0.812108 - logloss: 0.439818
2022-02-02 01:54:36,578 P86662 INFO Save best model: monitor(max): 0.812108
2022-02-02 01:54:36,700 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 01:54:36,742 P86662 INFO Train loss: 0.437720
2022-02-02 01:54:36,742 P86662 INFO ************ Epoch=20 end ************
2022-02-02 02:14:48,469 P86662 INFO [Metrics] AUC: 0.812155 - logloss: 0.439801
2022-02-02 02:14:48,470 P86662 INFO Save best model: monitor(max): 0.812155
2022-02-02 02:14:48,590 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 02:14:48,633 P86662 INFO Train loss: 0.437007
2022-02-02 02:14:48,633 P86662 INFO ************ Epoch=21 end ************
2022-02-02 02:34:59,038 P86662 INFO [Metrics] AUC: 0.812147 - logloss: 0.439812
2022-02-02 02:34:59,039 P86662 INFO Monitor(max) STOP: 0.812147 !
2022-02-02 02:34:59,040 P86662 INFO Reduce learning rate on plateau: 0.000001
2022-02-02 02:34:59,040 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 02:34:59,086 P86662 INFO Train loss: 0.436415
2022-02-02 02:34:59,086 P86662 INFO ************ Epoch=22 end ************
2022-02-02 02:55:09,654 P86662 INFO [Metrics] AUC: 0.812092 - logloss: 0.439928
2022-02-02 02:55:09,656 P86662 INFO Monitor(max) STOP: 0.812092 !
2022-02-02 02:55:09,656 P86662 INFO Reduce learning rate on plateau: 0.000001
2022-02-02 02:55:09,656 P86662 INFO Early stopping at epoch=23
2022-02-02 02:55:09,656 P86662 INFO --- 8058/8058 batches finished ---
2022-02-02 02:55:09,737 P86662 INFO Train loss: 0.434020
2022-02-02 02:55:09,737 P86662 INFO Training finished.
2022-02-02 02:55:09,737 P86662 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/HFM_criteo_x1/criteo_x1_7b681156/HFM_criteo_x1_007_e8226627.model
2022-02-02 02:55:14,333 P86662 INFO ****** Validation evaluation ******
2022-02-02 02:56:29,143 P86662 INFO [Metrics] AUC: 0.812155 - logloss: 0.439801
2022-02-02 02:56:29,235 P86662 INFO ******** Test evaluation ********
2022-02-02 02:56:29,235 P86662 INFO Loading data...
2022-02-02 02:56:29,236 P86662 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-02-02 02:56:29,827 P86662 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-02-02 02:56:29,828 P86662 INFO Loading test data done.
2022-02-02 02:57:11,158 P86662 INFO [Metrics] AUC: 0.812430 - logloss: 0.439360

```
