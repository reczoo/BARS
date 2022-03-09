## LR_criteo_x1

A hands-on guide to run the LR model on the Criteo_x1 dataset.

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

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [LR](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/LR.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [LR_criteo_x1_tuner_config_01](./LR_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd LR_criteo_x1
    nohup python run_expid.py --config ./LR_criteo_x1_tuner_config_01 --expid LR_criteo_x1_008_9581f586 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.788625 | 0.460914  |


### Logs
```python
2022-01-26 07:19:09,174 P48373 INFO {
    "batch_size": "4096",
    "data_block_size": "-1",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x1_7b681156",
    "debug": "False",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "7",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "LR",
    "model_id": "LR_criteo_x1_008_9581f586",
    "model_root": "./Criteo/LR_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "5e-07",
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
2022-01-26 07:19:09,175 P48373 INFO Set up feature encoder...
2022-01-26 07:19:09,175 P48373 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-26 07:19:09,175 P48373 INFO Loading data...
2022-01-26 07:19:09,177 P48373 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-26 07:19:13,848 P48373 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-26 07:19:14,955 P48373 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-26 07:19:14,955 P48373 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-26 07:19:14,955 P48373 INFO Loading train data done.
2022-01-26 07:19:20,375 P48373 INFO Total number of parameters: 2086317.
2022-01-26 07:19:20,376 P48373 INFO Start training: 8058 batches/epoch
2022-01-26 07:19:20,376 P48373 INFO ************ Epoch=1 start ************
2022-01-26 07:24:00,905 P48373 INFO [Metrics] AUC: 0.786731 - logloss: 0.462657
2022-01-26 07:24:00,907 P48373 INFO Save best model: monitor(max): 0.786731
2022-01-26 07:24:01,076 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:24:01,123 P48373 INFO Train loss: 0.472980
2022-01-26 07:24:01,123 P48373 INFO ************ Epoch=1 end ************
2022-01-26 07:28:36,658 P48373 INFO [Metrics] AUC: 0.787434 - logloss: 0.462107
2022-01-26 07:28:36,660 P48373 INFO Save best model: monitor(max): 0.787434
2022-01-26 07:28:36,672 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:28:36,714 P48373 INFO Train loss: 0.462698
2022-01-26 07:28:36,714 P48373 INFO ************ Epoch=2 end ************
2022-01-26 07:33:14,142 P48373 INFO [Metrics] AUC: 0.787561 - logloss: 0.461932
2022-01-26 07:33:14,143 P48373 INFO Save best model: monitor(max): 0.787561
2022-01-26 07:33:14,157 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:33:14,210 P48373 INFO Train loss: 0.462399
2022-01-26 07:33:14,210 P48373 INFO ************ Epoch=3 end ************
2022-01-26 07:37:53,317 P48373 INFO [Metrics] AUC: 0.787643 - logloss: 0.461889
2022-01-26 07:37:53,318 P48373 INFO Save best model: monitor(max): 0.787643
2022-01-26 07:37:53,330 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:37:53,375 P48373 INFO Train loss: 0.462328
2022-01-26 07:37:53,376 P48373 INFO ************ Epoch=4 end ************
2022-01-26 07:42:33,008 P48373 INFO [Metrics] AUC: 0.787626 - logloss: 0.461895
2022-01-26 07:42:33,009 P48373 INFO Monitor(max) STOP: 0.787626 !
2022-01-26 07:42:33,009 P48373 INFO Reduce learning rate on plateau: 0.000100
2022-01-26 07:42:33,009 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:42:33,057 P48373 INFO Train loss: 0.462297
2022-01-26 07:42:33,057 P48373 INFO ************ Epoch=5 end ************
2022-01-26 07:47:15,847 P48373 INFO [Metrics] AUC: 0.788262 - logloss: 0.461363
2022-01-26 07:47:15,848 P48373 INFO Save best model: monitor(max): 0.788262
2022-01-26 07:47:15,860 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:47:15,911 P48373 INFO Train loss: 0.459793
2022-01-26 07:47:15,911 P48373 INFO ************ Epoch=6 end ************
2022-01-26 07:51:51,853 P48373 INFO [Metrics] AUC: 0.788349 - logloss: 0.461293
2022-01-26 07:51:51,854 P48373 INFO Save best model: monitor(max): 0.788349
2022-01-26 07:51:51,866 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:51:51,920 P48373 INFO Train loss: 0.459596
2022-01-26 07:51:51,920 P48373 INFO ************ Epoch=7 end ************
2022-01-26 07:56:31,020 P48373 INFO [Metrics] AUC: 0.788404 - logloss: 0.461251
2022-01-26 07:56:31,021 P48373 INFO Save best model: monitor(max): 0.788404
2022-01-26 07:56:31,033 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 07:56:31,084 P48373 INFO Train loss: 0.459537
2022-01-26 07:56:31,084 P48373 INFO ************ Epoch=8 end ************
2022-01-26 08:01:06,677 P48373 INFO [Metrics] AUC: 0.788426 - logloss: 0.461229
2022-01-26 08:01:06,678 P48373 INFO Save best model: monitor(max): 0.788426
2022-01-26 08:01:06,690 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:01:06,733 P48373 INFO Train loss: 0.459514
2022-01-26 08:01:06,734 P48373 INFO ************ Epoch=9 end ************
2022-01-26 08:05:46,048 P48373 INFO [Metrics] AUC: 0.788442 - logloss: 0.461223
2022-01-26 08:05:46,049 P48373 INFO Save best model: monitor(max): 0.788442
2022-01-26 08:05:46,062 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:05:46,113 P48373 INFO Train loss: 0.459503
2022-01-26 08:05:46,113 P48373 INFO ************ Epoch=10 end ************
2022-01-26 08:10:22,286 P48373 INFO [Metrics] AUC: 0.788442 - logloss: 0.461213
2022-01-26 08:10:22,288 P48373 INFO Monitor(max) STOP: 0.788442 !
2022-01-26 08:10:22,288 P48373 INFO Reduce learning rate on plateau: 0.000010
2022-01-26 08:10:22,288 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:10:22,330 P48373 INFO Train loss: 0.459496
2022-01-26 08:10:22,330 P48373 INFO ************ Epoch=11 end ************
2022-01-26 08:15:03,498 P48373 INFO [Metrics] AUC: 0.788457 - logloss: 0.461204
2022-01-26 08:15:03,499 P48373 INFO Save best model: monitor(max): 0.788457
2022-01-26 08:15:03,511 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:15:03,570 P48373 INFO Train loss: 0.459115
2022-01-26 08:15:03,570 P48373 INFO ************ Epoch=12 end ************
2022-01-26 08:19:38,464 P48373 INFO [Metrics] AUC: 0.788461 - logloss: 0.461200
2022-01-26 08:19:38,465 P48373 INFO Save best model: monitor(max): 0.788461
2022-01-26 08:19:38,479 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:19:38,531 P48373 INFO Train loss: 0.459113
2022-01-26 08:19:38,531 P48373 INFO ************ Epoch=13 end ************
2022-01-26 08:24:09,246 P48373 INFO [Metrics] AUC: 0.788463 - logloss: 0.461199
2022-01-26 08:24:09,248 P48373 INFO Save best model: monitor(max): 0.788463
2022-01-26 08:24:09,262 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:24:09,313 P48373 INFO Train loss: 0.459111
2022-01-26 08:24:09,313 P48373 INFO ************ Epoch=14 end ************
2022-01-26 08:28:38,716 P48373 INFO [Metrics] AUC: 0.788464 - logloss: 0.461198
2022-01-26 08:28:38,718 P48373 INFO Save best model: monitor(max): 0.788464
2022-01-26 08:28:38,730 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:28:38,779 P48373 INFO Train loss: 0.459109
2022-01-26 08:28:38,779 P48373 INFO ************ Epoch=15 end ************
2022-01-26 08:33:10,426 P48373 INFO [Metrics] AUC: 0.788465 - logloss: 0.461198
2022-01-26 08:33:10,427 P48373 INFO Monitor(max) STOP: 0.788465 !
2022-01-26 08:33:10,427 P48373 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 08:33:10,427 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:33:10,479 P48373 INFO Train loss: 0.459111
2022-01-26 08:33:10,479 P48373 INFO ************ Epoch=16 end ************
2022-01-26 08:37:41,937 P48373 INFO [Metrics] AUC: 0.788466 - logloss: 0.461198
2022-01-26 08:37:41,938 P48373 INFO Save best model: monitor(max): 0.788466
2022-01-26 08:37:41,950 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:37:42,002 P48373 INFO Train loss: 0.459069
2022-01-26 08:37:42,002 P48373 INFO ************ Epoch=17 end ************
2022-01-26 08:42:10,624 P48373 INFO [Metrics] AUC: 0.788466 - logloss: 0.461198
2022-01-26 08:42:10,626 P48373 INFO Monitor(max) STOP: 0.788466 !
2022-01-26 08:42:10,626 P48373 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 08:42:10,626 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:42:10,678 P48373 INFO Train loss: 0.459068
2022-01-26 08:42:10,678 P48373 INFO ************ Epoch=18 end ************
2022-01-26 08:46:41,861 P48373 INFO [Metrics] AUC: 0.788466 - logloss: 0.461198
2022-01-26 08:46:41,863 P48373 INFO Monitor(max) STOP: 0.788466 !
2022-01-26 08:46:41,863 P48373 INFO Reduce learning rate on plateau: 0.000001
2022-01-26 08:46:41,863 P48373 INFO Early stopping at epoch=19
2022-01-26 08:46:41,863 P48373 INFO --- 8058/8058 batches finished ---
2022-01-26 08:46:41,909 P48373 INFO Train loss: 0.459067
2022-01-26 08:46:41,909 P48373 INFO Training finished.
2022-01-26 08:46:41,909 P48373 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/LR_criteo_x1/criteo_x1_7b681156/LR_criteo_x1_008_9581f586.model
2022-01-26 08:46:44,651 P48373 INFO ****** Validation evaluation ******
2022-01-26 08:47:09,002 P48373 INFO [Metrics] AUC: 0.788466 - logloss: 0.461198
2022-01-26 08:47:09,089 P48373 INFO ******** Test evaluation ********
2022-01-26 08:47:09,089 P48373 INFO Loading data...
2022-01-26 08:47:09,089 P48373 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-26 08:47:09,912 P48373 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-26 08:47:09,913 P48373 INFO Loading test data done.
2022-01-26 08:47:23,355 P48373 INFO [Metrics] AUC: 0.788625 - logloss: 0.460914

```
