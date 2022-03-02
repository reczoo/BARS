## HFM_criteo_x1

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

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM_criteo_x1_tuner_config_01](./HFM_criteo_x1_tuner_config_01). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM_criteo_x1
    nohup python run_expid.py --config ./HFM_criteo_x1_tuner_config_01 --expid HFM_criteo_x1_006_8906e1e1 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.803411 | 0.447976  |


### Logs
```python
2022-01-27 16:53:48,394 P8469 INFO {
    "batch_norm": "False",
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
    "gpu": "5",
    "hidden_activations": "relu",
    "hidden_units": "[64, 64, 64]",
    "interaction_type": "circular_correlation",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HFM",
    "model_id": "HFM_criteo_x1_006_8906e1e1",
    "model_root": "./Criteo/HFM_criteo_x1/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0",
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
    "use_dnn": "False",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x1/valid.csv",
    "verbose": "0",
    "version": "pytorch"
}
2022-01-27 16:53:48,394 P8469 INFO Set up feature encoder...
2022-01-27 16:53:48,394 P8469 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-27 16:53:48,395 P8469 INFO Loading data...
2022-01-27 16:53:48,395 P8469 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-27 16:53:52,797 P8469 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-27 16:53:53,881 P8469 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-27 16:53:53,881 P8469 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-27 16:53:53,881 P8469 INFO Loading train data done.
2022-01-27 16:53:58,633 P8469 INFO Total number of parameters: 22949487.
2022-01-27 16:53:58,633 P8469 INFO Start training: 8058 batches/epoch
2022-01-27 16:53:58,634 P8469 INFO ************ Epoch=1 start ************
2022-01-27 17:11:51,482 P8469 INFO [Metrics] AUC: 0.793590 - logloss: 0.456584
2022-01-27 17:11:51,484 P8469 INFO Save best model: monitor(max): 0.793590
2022-01-27 17:11:51,731 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 17:11:51,777 P8469 INFO Train loss: 0.471077
2022-01-27 17:11:51,778 P8469 INFO ************ Epoch=1 end ************
2022-01-27 17:29:41,265 P8469 INFO [Metrics] AUC: 0.794736 - logloss: 0.455559
2022-01-27 17:29:41,266 P8469 INFO Save best model: monitor(max): 0.794736
2022-01-27 17:29:41,376 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 17:29:41,421 P8469 INFO Train loss: 0.467609
2022-01-27 17:29:41,421 P8469 INFO ************ Epoch=2 end ************
2022-01-27 17:47:31,299 P8469 INFO [Metrics] AUC: 0.795161 - logloss: 0.455223
2022-01-27 17:47:31,300 P8469 INFO Save best model: monitor(max): 0.795161
2022-01-27 17:47:31,402 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 17:47:31,445 P8469 INFO Train loss: 0.467141
2022-01-27 17:47:31,446 P8469 INFO ************ Epoch=3 end ************
2022-01-27 18:05:22,537 P8469 INFO [Metrics] AUC: 0.795490 - logloss: 0.454923
2022-01-27 18:05:22,538 P8469 INFO Save best model: monitor(max): 0.795490
2022-01-27 18:05:22,637 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 18:05:22,687 P8469 INFO Train loss: 0.466931
2022-01-27 18:05:22,687 P8469 INFO ************ Epoch=4 end ************
2022-01-27 18:23:11,841 P8469 INFO [Metrics] AUC: 0.795585 - logloss: 0.454799
2022-01-27 18:23:11,842 P8469 INFO Save best model: monitor(max): 0.795585
2022-01-27 18:23:11,958 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 18:23:12,006 P8469 INFO Train loss: 0.466790
2022-01-27 18:23:12,007 P8469 INFO ************ Epoch=5 end ************
2022-01-27 18:41:01,651 P8469 INFO [Metrics] AUC: 0.795613 - logloss: 0.454763
2022-01-27 18:41:01,653 P8469 INFO Save best model: monitor(max): 0.795613
2022-01-27 18:41:01,768 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 18:41:01,820 P8469 INFO Train loss: 0.466709
2022-01-27 18:41:01,820 P8469 INFO ************ Epoch=6 end ************
2022-01-27 18:58:52,083 P8469 INFO [Metrics] AUC: 0.795878 - logloss: 0.454551
2022-01-27 18:58:52,084 P8469 INFO Save best model: monitor(max): 0.795878
2022-01-27 18:58:52,195 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 18:58:52,250 P8469 INFO Train loss: 0.466649
2022-01-27 18:58:52,251 P8469 INFO ************ Epoch=7 end ************
2022-01-27 19:16:41,986 P8469 INFO [Metrics] AUC: 0.795892 - logloss: 0.454547
2022-01-27 19:16:41,987 P8469 INFO Save best model: monitor(max): 0.795892
2022-01-27 19:16:42,099 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 19:16:42,149 P8469 INFO Train loss: 0.466606
2022-01-27 19:16:42,150 P8469 INFO ************ Epoch=8 end ************
2022-01-27 19:34:31,519 P8469 INFO [Metrics] AUC: 0.796028 - logloss: 0.454471
2022-01-27 19:34:31,520 P8469 INFO Save best model: monitor(max): 0.796028
2022-01-27 19:34:31,630 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 19:34:31,694 P8469 INFO Train loss: 0.466561
2022-01-27 19:34:31,694 P8469 INFO ************ Epoch=9 end ************
2022-01-27 19:52:20,268 P8469 INFO [Metrics] AUC: 0.795888 - logloss: 0.454576
2022-01-27 19:52:20,270 P8469 INFO Monitor(max) STOP: 0.795888 !
2022-01-27 19:52:20,270 P8469 INFO Reduce learning rate on plateau: 0.000100
2022-01-27 19:52:20,270 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 19:52:20,323 P8469 INFO Train loss: 0.466547
2022-01-27 19:52:20,323 P8469 INFO ************ Epoch=10 end ************
2022-01-27 20:10:04,934 P8469 INFO [Metrics] AUC: 0.800451 - logloss: 0.450581
2022-01-27 20:10:04,936 P8469 INFO Save best model: monitor(max): 0.800451
2022-01-27 20:10:05,042 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 20:10:05,092 P8469 INFO Train loss: 0.456116
2022-01-27 20:10:05,093 P8469 INFO ************ Epoch=11 end ************
2022-01-27 20:27:46,872 P8469 INFO [Metrics] AUC: 0.801229 - logloss: 0.449961
2022-01-27 20:27:46,873 P8469 INFO Save best model: monitor(max): 0.801229
2022-01-27 20:27:46,985 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 20:27:47,030 P8469 INFO Train loss: 0.452599
2022-01-27 20:27:47,031 P8469 INFO ************ Epoch=12 end ************
2022-01-27 20:45:27,814 P8469 INFO [Metrics] AUC: 0.801719 - logloss: 0.449478
2022-01-27 20:45:27,815 P8469 INFO Save best model: monitor(max): 0.801719
2022-01-27 20:45:27,918 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 20:45:27,965 P8469 INFO Train loss: 0.451410
2022-01-27 20:45:27,965 P8469 INFO ************ Epoch=13 end ************
2022-01-27 21:03:09,587 P8469 INFO [Metrics] AUC: 0.802022 - logloss: 0.449236
2022-01-27 21:03:09,589 P8469 INFO Save best model: monitor(max): 0.802022
2022-01-27 21:03:09,698 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 21:03:09,747 P8469 INFO Train loss: 0.450572
2022-01-27 21:03:09,747 P8469 INFO ************ Epoch=14 end ************
2022-01-27 21:20:53,690 P8469 INFO [Metrics] AUC: 0.802248 - logloss: 0.449045
2022-01-27 21:20:53,691 P8469 INFO Save best model: monitor(max): 0.802248
2022-01-27 21:20:53,791 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 21:20:53,844 P8469 INFO Train loss: 0.449896
2022-01-27 21:20:53,844 P8469 INFO ************ Epoch=15 end ************
2022-01-27 21:38:33,544 P8469 INFO [Metrics] AUC: 0.802352 - logloss: 0.448998
2022-01-27 21:38:33,546 P8469 INFO Save best model: monitor(max): 0.802352
2022-01-27 21:38:33,655 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 21:38:33,711 P8469 INFO Train loss: 0.449338
2022-01-27 21:38:33,711 P8469 INFO ************ Epoch=16 end ************
2022-01-27 21:56:20,578 P8469 INFO [Metrics] AUC: 0.802465 - logloss: 0.448870
2022-01-27 21:56:20,580 P8469 INFO Save best model: monitor(max): 0.802465
2022-01-27 21:56:20,682 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 21:56:20,739 P8469 INFO Train loss: 0.448874
2022-01-27 21:56:20,739 P8469 INFO ************ Epoch=17 end ************
2022-01-27 22:14:11,372 P8469 INFO [Metrics] AUC: 0.802489 - logloss: 0.448873
2022-01-27 22:14:11,374 P8469 INFO Save best model: monitor(max): 0.802489
2022-01-27 22:14:11,485 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 22:14:11,536 P8469 INFO Train loss: 0.448473
2022-01-27 22:14:11,536 P8469 INFO ************ Epoch=18 end ************
2022-01-27 22:32:00,209 P8469 INFO [Metrics] AUC: 0.802563 - logloss: 0.448821
2022-01-27 22:32:00,210 P8469 INFO Save best model: monitor(max): 0.802563
2022-01-27 22:32:00,309 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 22:32:00,361 P8469 INFO Train loss: 0.448130
2022-01-27 22:32:00,361 P8469 INFO ************ Epoch=19 end ************
2022-01-27 22:49:49,685 P8469 INFO [Metrics] AUC: 0.802543 - logloss: 0.448835
2022-01-27 22:49:49,687 P8469 INFO Monitor(max) STOP: 0.802543 !
2022-01-27 22:49:49,687 P8469 INFO Reduce learning rate on plateau: 0.000010
2022-01-27 22:49:49,687 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 22:49:49,738 P8469 INFO Train loss: 0.447831
2022-01-27 22:49:49,738 P8469 INFO ************ Epoch=20 end ************
2022-01-27 23:07:38,733 P8469 INFO [Metrics] AUC: 0.802993 - logloss: 0.448449
2022-01-27 23:07:38,734 P8469 INFO Save best model: monitor(max): 0.802993
2022-01-27 23:07:38,835 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 23:07:38,879 P8469 INFO Train loss: 0.443764
2022-01-27 23:07:38,879 P8469 INFO ************ Epoch=21 end ************
2022-01-27 23:25:27,512 P8469 INFO [Metrics] AUC: 0.803056 - logloss: 0.448421
2022-01-27 23:25:27,513 P8469 INFO Save best model: monitor(max): 0.803056
2022-01-27 23:25:27,616 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 23:25:27,670 P8469 INFO Train loss: 0.443379
2022-01-27 23:25:27,670 P8469 INFO ************ Epoch=22 end ************
2022-01-27 23:43:06,260 P8469 INFO [Metrics] AUC: 0.803051 - logloss: 0.448428
2022-01-27 23:43:06,261 P8469 INFO Monitor(max) STOP: 0.803051 !
2022-01-27 23:43:06,261 P8469 INFO Reduce learning rate on plateau: 0.000001
2022-01-27 23:43:06,261 P8469 INFO --- 8058/8058 batches finished ---
2022-01-27 23:43:06,313 P8469 INFO Train loss: 0.443180
2022-01-27 23:43:06,313 P8469 INFO ************ Epoch=23 end ************
2022-01-28 00:00:46,714 P8469 INFO [Metrics] AUC: 0.803069 - logloss: 0.448418
2022-01-28 00:00:46,716 P8469 INFO Save best model: monitor(max): 0.803069
2022-01-28 00:00:46,825 P8469 INFO --- 8058/8058 batches finished ---
2022-01-28 00:00:46,873 P8469 INFO Train loss: 0.442489
2022-01-28 00:00:46,874 P8469 INFO ************ Epoch=24 end ************
2022-01-28 00:18:25,402 P8469 INFO [Metrics] AUC: 0.803070 - logloss: 0.448419
2022-01-28 00:18:25,404 P8469 INFO Save best model: monitor(max): 0.803070
2022-01-28 00:18:25,516 P8469 INFO --- 8058/8058 batches finished ---
2022-01-28 00:18:25,566 P8469 INFO Train loss: 0.442468
2022-01-28 00:18:25,566 P8469 INFO ************ Epoch=25 end ************
2022-01-28 00:36:04,166 P8469 INFO [Metrics] AUC: 0.803068 - logloss: 0.448422
2022-01-28 00:36:04,167 P8469 INFO Monitor(max) STOP: 0.803068 !
2022-01-28 00:36:04,167 P8469 INFO Reduce learning rate on plateau: 0.000001
2022-01-28 00:36:04,167 P8469 INFO --- 8058/8058 batches finished ---
2022-01-28 00:36:04,213 P8469 INFO Train loss: 0.442452
2022-01-28 00:36:04,214 P8469 INFO ************ Epoch=26 end ************
2022-01-28 00:53:42,307 P8469 INFO [Metrics] AUC: 0.803067 - logloss: 0.448424
2022-01-28 00:53:42,309 P8469 INFO Monitor(max) STOP: 0.803067 !
2022-01-28 00:53:42,309 P8469 INFO Reduce learning rate on plateau: 0.000001
2022-01-28 00:53:42,309 P8469 INFO Early stopping at epoch=27
2022-01-28 00:53:42,309 P8469 INFO --- 8058/8058 batches finished ---
2022-01-28 00:53:42,356 P8469 INFO Train loss: 0.442435
2022-01-28 00:53:42,356 P8469 INFO Training finished.
2022-01-28 00:53:42,356 P8469 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/HFM_criteo_x1/criteo_x1_7b681156/HFM_criteo_x1_006_8906e1e1.model
2022-01-28 00:53:45,610 P8469 INFO ****** Validation evaluation ******
2022-01-28 00:54:54,542 P8469 INFO [Metrics] AUC: 0.803070 - logloss: 0.448419
2022-01-28 00:54:54,637 P8469 INFO ******** Test evaluation ********
2022-01-28 00:54:54,638 P8469 INFO Loading data...
2022-01-28 00:54:54,638 P8469 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-28 00:54:55,441 P8469 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-28 00:54:55,441 P8469 INFO Loading test data done.
2022-01-28 00:55:33,400 P8469 INFO [Metrics] AUC: 0.803411 - logloss: 0.447976

```
