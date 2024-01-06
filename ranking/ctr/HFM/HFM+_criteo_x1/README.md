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
Dataset ID: [Criteo_x1](https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo#Criteo_x1). Please refer to the dataset details to get data ready.

### Code

We use [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/tree/v1.1.0) for this experiment. See the model code: [HFM](https://github.com/xue-pai/FuxiCTR/blob/v1.1.0/fuxictr/pytorch/models/HFM.py).

Running steps:

1. Download [FuxiCTR-v1.1.0](https://github.com/xue-pai/FuxiCTR/archive/refs/tags/v1.1.0.zip) and install all the dependencies listed in the [environments](#environments). Then modify [run_expid.py](./run_expid.py#L5) to add the FuxiCTR library to system path
    
    ```python
    sys.path.append('YOUR_PATH_TO_FuxiCTR/')
    ```

2. Create a data directory and put the downloaded csv files in `../data/Criteo/Criteo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [HFM+_criteo_x1_tuner_config_04](./HFM+_criteo_x1_tuner_config_04). Make sure the data paths in `dataset_config.yaml` are correctly set to what we create in the last step.

4. Run the following script to start.

    ```bash
    cd HFM+_criteo_x1
    nohup python run_expid.py --config ./HFM+_criteo_x1_tuner_config_04 --expid HFM_criteo_x1_009_f964ad0a --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| AUC | logloss  |
|:--------------------:|:--------------------:|
| 0.812476 | 0.439354  |


### Logs
```python
2022-01-29 07:17:03,720 P15796 INFO {
    "batch_norm": "False",
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
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "interaction_type": "circular_convolution",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "HFM",
    "model_id": "HFM_criteo_x1_009_f964ad0a",
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
2022-01-29 07:17:03,721 P15796 INFO Set up feature encoder...
2022-01-29 07:17:03,721 P15796 INFO Load feature_map from json: ../data/Criteo/criteo_x1_7b681156/feature_map.json
2022-01-29 07:17:03,721 P15796 INFO Loading data...
2022-01-29 07:17:03,723 P15796 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/train.h5
2022-01-29 07:17:08,545 P15796 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/valid.h5
2022-01-29 07:17:09,773 P15796 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2022-01-29 07:17:09,773 P15796 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2022-01-29 07:17:09,773 P15796 INFO Loading train data done.
2022-01-29 07:17:14,833 P15796 INFO Total number of parameters: 26235078.
2022-01-29 07:17:14,833 P15796 INFO Start training: 8058 batches/epoch
2022-01-29 07:17:14,833 P15796 INFO ************ Epoch=1 start ************
2022-01-29 08:00:22,013 P15796 INFO [Metrics] AUC: 0.794431 - logloss: 0.455498
2022-01-29 08:00:22,015 P15796 INFO Save best model: monitor(max): 0.794431
2022-01-29 08:00:22,117 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 08:00:22,165 P15796 INFO Train loss: 0.488792
2022-01-29 08:00:22,165 P15796 INFO ************ Epoch=1 end ************
2022-01-29 08:43:28,259 P15796 INFO [Metrics] AUC: 0.795888 - logloss: 0.454244
2022-01-29 08:43:28,261 P15796 INFO Save best model: monitor(max): 0.795888
2022-01-29 08:43:28,396 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 08:43:28,443 P15796 INFO Train loss: 0.484133
2022-01-29 08:43:28,443 P15796 INFO ************ Epoch=2 end ************
2022-01-29 09:26:34,103 P15796 INFO [Metrics] AUC: 0.796629 - logloss: 0.454102
2022-01-29 09:26:34,104 P15796 INFO Save best model: monitor(max): 0.796629
2022-01-29 09:26:34,215 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 09:26:34,266 P15796 INFO Train loss: 0.483709
2022-01-29 09:26:34,266 P15796 INFO ************ Epoch=3 end ************
2022-01-29 10:09:39,919 P15796 INFO [Metrics] AUC: 0.797122 - logloss: 0.453256
2022-01-29 10:09:39,920 P15796 INFO Save best model: monitor(max): 0.797122
2022-01-29 10:09:40,106 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 10:09:40,158 P15796 INFO Train loss: 0.483645
2022-01-29 10:09:40,158 P15796 INFO ************ Epoch=4 end ************
2022-01-29 10:52:47,450 P15796 INFO [Metrics] AUC: 0.797692 - logloss: 0.454744
2022-01-29 10:52:47,451 P15796 INFO Save best model: monitor(max): 0.797692
2022-01-29 10:52:47,581 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 10:52:47,626 P15796 INFO Train loss: 0.483775
2022-01-29 10:52:47,627 P15796 INFO ************ Epoch=5 end ************
2022-01-29 11:35:55,126 P15796 INFO [Metrics] AUC: 0.798113 - logloss: 0.452593
2022-01-29 11:35:55,127 P15796 INFO Save best model: monitor(max): 0.798113
2022-01-29 11:35:55,241 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 11:35:55,286 P15796 INFO Train loss: 0.483832
2022-01-29 11:35:55,287 P15796 INFO ************ Epoch=6 end ************
2022-01-29 12:19:00,781 P15796 INFO [Metrics] AUC: 0.797824 - logloss: 0.452939
2022-01-29 12:19:00,782 P15796 INFO Monitor(max) STOP: 0.797824 !
2022-01-29 12:19:00,782 P15796 INFO Reduce learning rate on plateau: 0.000100
2022-01-29 12:19:00,782 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 12:19:00,836 P15796 INFO Train loss: 0.483941
2022-01-29 12:19:00,837 P15796 INFO ************ Epoch=7 end ************
2022-01-29 13:02:07,261 P15796 INFO [Metrics] AUC: 0.807279 - logloss: 0.444483
2022-01-29 13:02:07,263 P15796 INFO Save best model: monitor(max): 0.807279
2022-01-29 13:02:07,403 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 13:02:07,454 P15796 INFO Train loss: 0.455806
2022-01-29 13:02:07,454 P15796 INFO ************ Epoch=8 end ************
2022-01-29 13:45:13,841 P15796 INFO [Metrics] AUC: 0.808245 - logloss: 0.443706
2022-01-29 13:45:13,842 P15796 INFO Save best model: monitor(max): 0.808245
2022-01-29 13:45:13,977 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 13:45:14,023 P15796 INFO Train loss: 0.452184
2022-01-29 13:45:14,023 P15796 INFO ************ Epoch=9 end ************
2022-01-29 14:28:21,308 P15796 INFO [Metrics] AUC: 0.808653 - logloss: 0.443092
2022-01-29 14:28:21,309 P15796 INFO Save best model: monitor(max): 0.808653
2022-01-29 14:28:21,461 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 14:28:21,513 P15796 INFO Train loss: 0.451564
2022-01-29 14:28:21,513 P15796 INFO ************ Epoch=10 end ************
2022-01-29 15:11:28,215 P15796 INFO [Metrics] AUC: 0.808937 - logloss: 0.442840
2022-01-29 15:11:28,216 P15796 INFO Save best model: monitor(max): 0.808937
2022-01-29 15:11:28,328 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 15:11:28,375 P15796 INFO Train loss: 0.451356
2022-01-29 15:11:28,375 P15796 INFO ************ Epoch=11 end ************
2022-01-29 15:54:35,347 P15796 INFO [Metrics] AUC: 0.809070 - logloss: 0.442711
2022-01-29 15:54:35,349 P15796 INFO Save best model: monitor(max): 0.809070
2022-01-29 15:54:35,470 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 15:54:35,516 P15796 INFO Train loss: 0.451238
2022-01-29 15:54:35,516 P15796 INFO ************ Epoch=12 end ************
2022-01-29 16:37:39,663 P15796 INFO [Metrics] AUC: 0.809118 - logloss: 0.442616
2022-01-29 16:37:39,665 P15796 INFO Save best model: monitor(max): 0.809118
2022-01-29 16:37:39,785 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 16:37:39,831 P15796 INFO Train loss: 0.451132
2022-01-29 16:37:39,831 P15796 INFO ************ Epoch=13 end ************
2022-01-29 17:20:41,830 P15796 INFO [Metrics] AUC: 0.809273 - logloss: 0.442544
2022-01-29 17:20:41,833 P15796 INFO Save best model: monitor(max): 0.809273
2022-01-29 17:20:41,956 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 17:20:42,008 P15796 INFO Train loss: 0.451035
2022-01-29 17:20:42,008 P15796 INFO ************ Epoch=14 end ************
2022-01-29 18:03:41,018 P15796 INFO [Metrics] AUC: 0.809298 - logloss: 0.442543
2022-01-29 18:03:41,020 P15796 INFO Save best model: monitor(max): 0.809298
2022-01-29 18:03:41,164 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 18:03:41,206 P15796 INFO Train loss: 0.450959
2022-01-29 18:03:41,207 P15796 INFO ************ Epoch=15 end ************
2022-01-29 18:46:39,826 P15796 INFO [Metrics] AUC: 0.809359 - logloss: 0.442372
2022-01-29 18:46:39,828 P15796 INFO Save best model: monitor(max): 0.809359
2022-01-29 18:46:39,951 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 18:46:40,004 P15796 INFO Train loss: 0.450855
2022-01-29 18:46:40,004 P15796 INFO ************ Epoch=16 end ************
2022-01-29 19:29:36,038 P15796 INFO [Metrics] AUC: 0.809389 - logloss: 0.442486
2022-01-29 19:29:36,040 P15796 INFO Save best model: monitor(max): 0.809389
2022-01-29 19:29:36,177 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 19:29:36,226 P15796 INFO Train loss: 0.450776
2022-01-29 19:29:36,227 P15796 INFO ************ Epoch=17 end ************
2022-01-29 20:12:30,063 P15796 INFO [Metrics] AUC: 0.809321 - logloss: 0.442367
2022-01-29 20:12:30,065 P15796 INFO Monitor(max) STOP: 0.809321 !
2022-01-29 20:12:30,065 P15796 INFO Reduce learning rate on plateau: 0.000010
2022-01-29 20:12:30,065 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 20:12:30,115 P15796 INFO Train loss: 0.450705
2022-01-29 20:12:30,115 P15796 INFO ************ Epoch=18 end ************
2022-01-29 20:55:24,435 P15796 INFO [Metrics] AUC: 0.811307 - logloss: 0.440597
2022-01-29 20:55:24,437 P15796 INFO Save best model: monitor(max): 0.811307
2022-01-29 20:55:24,560 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 20:55:24,610 P15796 INFO Train loss: 0.443180
2022-01-29 20:55:24,610 P15796 INFO ************ Epoch=19 end ************
2022-01-29 21:38:16,897 P15796 INFO [Metrics] AUC: 0.811721 - logloss: 0.440171
2022-01-29 21:38:16,899 P15796 INFO Save best model: monitor(max): 0.811721
2022-01-29 21:38:17,028 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 21:38:17,076 P15796 INFO Train loss: 0.440641
2022-01-29 21:38:17,076 P15796 INFO ************ Epoch=20 end ************
2022-01-29 22:21:08,513 P15796 INFO [Metrics] AUC: 0.811926 - logloss: 0.440002
2022-01-29 22:21:08,516 P15796 INFO Save best model: monitor(max): 0.811926
2022-01-29 22:21:08,633 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 22:21:08,679 P15796 INFO Train loss: 0.439618
2022-01-29 22:21:08,680 P15796 INFO ************ Epoch=21 end ************
2022-01-29 23:03:58,081 P15796 INFO [Metrics] AUC: 0.812045 - logloss: 0.439862
2022-01-29 23:03:58,083 P15796 INFO Save best model: monitor(max): 0.812045
2022-01-29 23:03:58,214 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 23:03:58,267 P15796 INFO Train loss: 0.438936
2022-01-29 23:03:58,267 P15796 INFO ************ Epoch=22 end ************
2022-01-29 23:46:41,251 P15796 INFO [Metrics] AUC: 0.812125 - logloss: 0.439802
2022-01-29 23:46:41,253 P15796 INFO Save best model: monitor(max): 0.812125
2022-01-29 23:46:41,380 P15796 INFO --- 8058/8058 batches finished ---
2022-01-29 23:46:41,432 P15796 INFO Train loss: 0.438320
2022-01-29 23:46:41,432 P15796 INFO ************ Epoch=23 end ************
2022-01-30 00:29:14,817 P15796 INFO [Metrics] AUC: 0.812168 - logloss: 0.439773
2022-01-30 00:29:14,819 P15796 INFO Save best model: monitor(max): 0.812168
2022-01-30 00:29:14,951 P15796 INFO --- 8058/8058 batches finished ---
2022-01-30 00:29:15,002 P15796 INFO Train loss: 0.437806
2022-01-30 00:29:15,002 P15796 INFO ************ Epoch=24 end ************
2022-01-30 01:11:46,837 P15796 INFO [Metrics] AUC: 0.812178 - logloss: 0.439817
2022-01-30 01:11:46,839 P15796 INFO Save best model: monitor(max): 0.812178
2022-01-30 01:11:46,989 P15796 INFO --- 8058/8058 batches finished ---
2022-01-30 01:11:47,037 P15796 INFO Train loss: 0.437344
2022-01-30 01:11:47,037 P15796 INFO ************ Epoch=25 end ************
2022-01-30 01:54:18,983 P15796 INFO [Metrics] AUC: 0.812134 - logloss: 0.439816
2022-01-30 01:54:18,985 P15796 INFO Monitor(max) STOP: 0.812134 !
2022-01-30 01:54:18,985 P15796 INFO Reduce learning rate on plateau: 0.000001
2022-01-30 01:54:18,985 P15796 INFO --- 8058/8058 batches finished ---
2022-01-30 01:54:19,039 P15796 INFO Train loss: 0.436911
2022-01-30 01:54:19,039 P15796 INFO ************ Epoch=26 end ************
2022-01-30 02:36:49,241 P15796 INFO [Metrics] AUC: 0.812020 - logloss: 0.440040
2022-01-30 02:36:49,243 P15796 INFO Monitor(max) STOP: 0.812020 !
2022-01-30 02:36:49,243 P15796 INFO Reduce learning rate on plateau: 0.000001
2022-01-30 02:36:49,243 P15796 INFO Early stopping at epoch=27
2022-01-30 02:36:49,243 P15796 INFO --- 8058/8058 batches finished ---
2022-01-30 02:36:49,294 P15796 INFO Train loss: 0.434335
2022-01-30 02:36:49,295 P15796 INFO Training finished.
2022-01-30 02:36:49,295 P15796 INFO Load best model: /cache/FuxiCTR/benchmarks/Criteo/HFM_criteo_x1/criteo_x1_7b681156/HFM_criteo_x1_009_f964ad0a.model
2022-01-30 02:36:49,470 P15796 INFO ****** Validation evaluation ******
2022-01-30 02:38:47,674 P15796 INFO [Metrics] AUC: 0.812178 - logloss: 0.439817
2022-01-30 02:38:47,764 P15796 INFO ******** Test evaluation ********
2022-01-30 02:38:47,765 P15796 INFO Loading data...
2022-01-30 02:38:47,765 P15796 INFO Loading data from h5: ../data/Criteo/criteo_x1_7b681156/test.h5
2022-01-30 02:38:48,604 P15796 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2022-01-30 02:38:48,604 P15796 INFO Loading test data done.
2022-01-30 02:39:48,320 P15796 INFO [Metrics] AUC: 0.812476 - logloss: 0.439354

```
