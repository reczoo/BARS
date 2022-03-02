## CIN_criteo_x1

A hands-on guide to run the xDeepFM model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](fuxictr_url) for this experiment. See model code: [xDeepFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/xDeepFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](fuxictr_url) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Avazu/Avazu_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [CIN_criteo_x1_tuner_config_02](./CIN_criteo_x1_tuner_config_02). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd CIN_criteo_x1
    nohup python run_expid.py --config ./CIN_criteo_x1_tuner_config_02 --expid xDeepFM_criteo_x1_004_6b5adc06 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.811731 | 0.440261  |


### Logs
```python
2022-01-20 23:40:24,682 P809 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "cin_layer_units": "[32, 32, 32, 32, 32]",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "dnn_hidden_units": "None",
    "embedding_dim": "10",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "3",
    "hidden_activations": "relu",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "xDeepFM",
    "model_id": "xDeepFM_criteo_x1_004_6b5adc06",
    "model_root": "./Criteo/xDeepFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
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
    "test_data": "../data/Criteo/Criteo_x1/test.csv",
    "train_data": "../data/Criteo/Criteo_x1/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2022-01-20 23:40:24,683 P809 INFO Set up feature encoder...
2022-01-20 23:40:24,683 P809 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-20 23:40:24,683 P809 INFO Loading data...
2022-01-20 23:40:24,685 P809 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-20 23:40:30,964 P809 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-20 23:40:32,090 P809 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-20 23:40:32,090 P809 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-20 23:40:32,091 P809 INFO Loading train data done.
2022-01-20 23:40:36,182 P809 INFO Total number of parameters: 23158213.
2022-01-20 23:40:36,182 P809 INFO Start training: 8058 batches/epoch
2022-01-20 23:40:36,182 P809 INFO ************ Epoch=1 start ************
2022-01-20 23:59:12,357 P809 INFO [Metrics] AUC: 0.802357 - logloss: 0.448861
2022-01-20 23:59:12,358 P809 INFO Save best model: monitor(max): 0.802357
2022-01-20 23:59:12,599 P809 INFO --- 8058/8058 batches finished ---
2022-01-20 23:59:12,638 P809 INFO Train loss: 0.463596
2022-01-20 23:59:12,638 P809 INFO ************ Epoch=1 end ************
2022-01-21 00:17:46,705 P809 INFO [Metrics] AUC: 0.804488 - logloss: 0.446831
2022-01-21 00:17:46,707 P809 INFO Save best model: monitor(max): 0.804488
2022-01-21 00:17:46,847 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 00:17:46,905 P809 INFO Train loss: 0.458500
2022-01-21 00:17:46,906 P809 INFO ************ Epoch=2 end ************
2022-01-21 00:36:19,514 P809 INFO [Metrics] AUC: 0.805366 - logloss: 0.446015
2022-01-21 00:36:19,516 P809 INFO Save best model: monitor(max): 0.805366
2022-01-21 00:36:19,624 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 00:36:19,673 P809 INFO Train loss: 0.457177
2022-01-21 00:36:19,673 P809 INFO ************ Epoch=3 end ************
2022-01-21 00:54:51,176 P809 INFO [Metrics] AUC: 0.805928 - logloss: 0.445628
2022-01-21 00:54:51,178 P809 INFO Save best model: monitor(max): 0.805928
2022-01-21 00:54:51,314 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 00:54:51,365 P809 INFO Train loss: 0.456447
2022-01-21 00:54:51,365 P809 INFO ************ Epoch=4 end ************
2022-01-21 01:13:24,925 P809 INFO [Metrics] AUC: 0.806333 - logloss: 0.445143
2022-01-21 01:13:24,926 P809 INFO Save best model: monitor(max): 0.806333
2022-01-21 01:13:25,060 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 01:13:25,120 P809 INFO Train loss: 0.455955
2022-01-21 01:13:25,121 P809 INFO ************ Epoch=5 end ************
2022-01-21 01:31:53,427 P809 INFO [Metrics] AUC: 0.806547 - logloss: 0.444999
2022-01-21 01:31:53,429 P809 INFO Save best model: monitor(max): 0.806547
2022-01-21 01:31:53,545 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 01:31:53,596 P809 INFO Train loss: 0.455561
2022-01-21 01:31:53,596 P809 INFO ************ Epoch=6 end ************
2022-01-21 01:50:24,779 P809 INFO [Metrics] AUC: 0.806691 - logloss: 0.444828
2022-01-21 01:50:24,781 P809 INFO Save best model: monitor(max): 0.806691
2022-01-21 01:50:24,885 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 01:50:24,943 P809 INFO Train loss: 0.455276
2022-01-21 01:50:24,943 P809 INFO ************ Epoch=7 end ************
2022-01-21 02:08:56,474 P809 INFO [Metrics] AUC: 0.806990 - logloss: 0.444553
2022-01-21 02:08:56,476 P809 INFO Save best model: monitor(max): 0.806990
2022-01-21 02:08:56,586 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 02:08:56,639 P809 INFO Train loss: 0.455011
2022-01-21 02:08:56,639 P809 INFO ************ Epoch=8 end ************
2022-01-21 02:27:29,476 P809 INFO [Metrics] AUC: 0.807143 - logloss: 0.444413
2022-01-21 02:27:29,477 P809 INFO Save best model: monitor(max): 0.807143
2022-01-21 02:27:29,586 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 02:27:29,639 P809 INFO Train loss: 0.454811
2022-01-21 02:27:29,640 P809 INFO ************ Epoch=9 end ************
2022-01-21 02:46:04,060 P809 INFO [Metrics] AUC: 0.807085 - logloss: 0.444595
2022-01-21 02:46:04,062 P809 INFO Monitor(max) STOP: 0.807085 !
2022-01-21 02:46:04,062 P809 INFO Reduce learning rate on plateau: 0.000100
2022-01-21 02:46:04,062 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 02:46:04,117 P809 INFO Train loss: 0.454605
2022-01-21 02:46:04,117 P809 INFO ************ Epoch=10 end ************
2022-01-21 03:04:34,056 P809 INFO [Metrics] AUC: 0.810872 - logloss: 0.441116
2022-01-21 03:04:34,057 P809 INFO Save best model: monitor(max): 0.810872
2022-01-21 03:04:34,166 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 03:04:34,233 P809 INFO Train loss: 0.443612
2022-01-21 03:04:34,233 P809 INFO ************ Epoch=11 end ************
2022-01-21 03:23:04,419 P809 INFO [Metrics] AUC: 0.811375 - logloss: 0.440755
2022-01-21 03:23:04,420 P809 INFO Save best model: monitor(max): 0.811375
2022-01-21 03:23:04,521 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 03:23:04,581 P809 INFO Train loss: 0.439668
2022-01-21 03:23:04,581 P809 INFO ************ Epoch=12 end ************
2022-01-21 03:41:33,693 P809 INFO [Metrics] AUC: 0.811436 - logloss: 0.440684
2022-01-21 03:41:33,695 P809 INFO Save best model: monitor(max): 0.811436
2022-01-21 03:41:33,810 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 03:41:33,876 P809 INFO Train loss: 0.437951
2022-01-21 03:41:33,876 P809 INFO ************ Epoch=13 end ************
2022-01-21 04:00:03,106 P809 INFO [Metrics] AUC: 0.811363 - logloss: 0.440867
2022-01-21 04:00:03,108 P809 INFO Monitor(max) STOP: 0.811363 !
2022-01-21 04:00:03,108 P809 INFO Reduce learning rate on plateau: 0.000010
2022-01-21 04:00:03,108 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 04:00:03,166 P809 INFO Train loss: 0.436571
2022-01-21 04:00:03,166 P809 INFO ************ Epoch=14 end ************
2022-01-21 04:18:33,909 P809 INFO [Metrics] AUC: 0.810852 - logloss: 0.441550
2022-01-21 04:18:33,911 P809 INFO Monitor(max) STOP: 0.810852 !
2022-01-21 04:18:33,911 P809 INFO Reduce learning rate on plateau: 0.000001
2022-01-21 04:18:33,911 P809 INFO Early stopping at epoch=15
2022-01-21 04:18:33,911 P809 INFO --- 8058/8058 batches finished ---
2022-01-21 04:18:33,965 P809 INFO Train loss: 0.431973
2022-01-21 04:18:33,965 P809 INFO Training finished.
2022-01-21 04:18:33,965 P809 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/xDeepFM_criteo_x1/criteo_x1_7b681156/xDeepFM_criteo_x1_004_6b5adc06.model
2022-01-21 04:18:38,465 P809 INFO ****** Validation evaluation ******
2022-01-21 04:19:13,567 P809 INFO [Metrics] AUC: 0.811436 - logloss: 0.440684
2022-01-21 04:19:13,657 P809 INFO ******** Test evaluation ********
2022-01-21 04:19:13,657 P809 INFO Loading data...
2022-01-21 04:19:13,657 P809 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-21 04:19:14,419 P809 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-21 04:19:14,419 P809 INFO Loading test data done.
2022-01-21 04:19:34,377 P809 INFO [Metrics] AUC: 0.811731 - logloss: 0.440261

```
