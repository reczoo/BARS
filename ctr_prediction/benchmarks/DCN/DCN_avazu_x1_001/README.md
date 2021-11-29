## DCN_Avazu_x0_001

A notebook to benchmark DCN on Avazu_x0_001 dataset.

Author: [XUEPAI Team](https://github.com/xue-pai)


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.6GHz
  RAM: 500G+
  ```
+ Software

  ```python
  python: 3.6.5
  pandas: 1.0.0
  numpy: 1.18.1
  ```

### Dataset
This dataset split follows the setting in the AFN work. That is, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. The data preprocessing script is provided on Github and we directly download the preprocessed data.
In this setting, we follow the AFN work to fix embedding_dim=16, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons.


### Code




### Results
```python
[Metrics] AUC: 0.764880 - logloss: 0.366279
```


### Logs
```python
2021-08-13 16:44:21,846 P30750 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x0_83355fc7",
    "debug": "False",
    "dnn_activations": "relu",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "low_rank": "32",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DCN_v2",
    "model_id": "DCN_v2_avazu_x0_003_1b8122b0",
    "model_root": "./Avazu/DCN_avazu_x0/",
    "model_structure": "parallel",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.4",
    "net_regularizer": "0",
    "num_cross_layers": "3",
    "num_experts": "4",
    "num_workers": "3",
    "optimizer": "adam",
    "parallel_dnn_hidden_units": "[400, 400, 400]",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2021",
    "shuffle": "True",
    "stacked_dnn_hidden_units": "[500, 500, 500]",
    "task": "binary_classification",
    "test_data": "../data/Avazu/Avazu_x0/test.csv",
    "train_data": "../data/Avazu/Avazu_x0/train.csv",
    "use_hdf5": "True",
    "use_low_rank_mixture": "False",
    "valid_data": "../data/Avazu/Avazu_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-08-13 16:44:21,846 P30750 INFO Set up feature encoder...
2021-08-13 16:44:21,847 P30750 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x0_83355fc7/feature_encoder.pkl
2021-08-13 16:44:23,335 P30750 INFO Total number of parameters: 13544071.
2021-08-13 16:44:23,335 P30750 INFO Loading data...
2021-08-13 16:44:23,338 P30750 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/train.h5
2021-08-13 16:45:11,387 P30750 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/valid.h5
2021-08-13 16:45:19,612 P30750 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-08-13 16:45:19,612 P30750 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-08-13 16:45:19,612 P30750 INFO Loading train data done.
2021-08-13 16:45:26,861 P30750 INFO Start training: 6910 batches/epoch
2021-08-13 16:45:26,861 P30750 INFO ************ Epoch=1 start ************
2021-08-13 17:03:47,278 P30750 INFO [Metrics] AUC: 0.745429 - logloss: 0.396650
2021-08-13 17:03:47,294 P30750 INFO Save best model: monitor(max): 0.745429
2021-08-13 17:03:47,373 P30750 INFO --- 6910/6910 batches finished ---
2021-08-13 17:03:47,607 P30750 INFO Train loss: 0.422170
2021-08-13 17:03:47,607 P30750 INFO ************ Epoch=1 end ************
2021-08-13 17:21:47,030 P30750 INFO [Metrics] AUC: 0.742704 - logloss: 0.398837
2021-08-13 17:21:47,041 P30750 INFO Monitor(max) STOP: 0.742704 !
2021-08-13 17:21:47,041 P30750 INFO Reduce learning rate on plateau: 0.000100
2021-08-13 17:21:47,041 P30750 INFO --- 6910/6910 batches finished ---
2021-08-13 17:21:47,233 P30750 INFO Train loss: 0.419784
2021-08-13 17:21:47,233 P30750 INFO ************ Epoch=2 end ************
2021-08-13 17:39:38,061 P30750 INFO [Metrics] AUC: 0.746213 - logloss: 0.395836
2021-08-13 17:39:38,066 P30750 INFO Save best model: monitor(max): 0.746213
2021-08-13 17:39:38,162 P30750 INFO --- 6910/6910 batches finished ---
2021-08-13 17:39:38,352 P30750 INFO Train loss: 0.398965
2021-08-13 17:39:38,352 P30750 INFO ************ Epoch=3 end ************
2021-08-13 17:57:32,992 P30750 INFO [Metrics] AUC: 0.743226 - logloss: 0.398230
2021-08-13 17:57:33,007 P30750 INFO Monitor(max) STOP: 0.743226 !
2021-08-13 17:57:33,007 P30750 INFO Reduce learning rate on plateau: 0.000010
2021-08-13 17:57:33,007 P30750 INFO --- 6910/6910 batches finished ---
2021-08-13 17:57:33,181 P30750 INFO Train loss: 0.395656
2021-08-13 17:57:33,181 P30750 INFO ************ Epoch=4 end ************
2021-08-13 18:15:26,421 P30750 INFO [Metrics] AUC: 0.736039 - logloss: 0.402718
2021-08-13 18:15:26,440 P30750 INFO Monitor(max) STOP: 0.736039 !
2021-08-13 18:15:26,440 P30750 INFO Reduce learning rate on plateau: 0.000001
2021-08-13 18:15:26,440 P30750 INFO Early stopping at epoch=5
2021-08-13 18:15:26,440 P30750 INFO --- 6910/6910 batches finished ---
2021-08-13 18:15:26,626 P30750 INFO Train loss: 0.385052
2021-08-13 18:15:26,626 P30750 INFO Training finished.
2021-08-13 18:15:26,626 P30750 INFO Load best model: /home/xxx/xxx/FuxiCTR_v2/FuxiCTR/benchmarks/Avazu/DCN_avazu_x0/avazu_x0_83355fc7/DCN_v2_avazu_x0_003_1b8122b0.model
2021-08-13 18:15:26,707 P30750 INFO ****** Train/validation evaluation ******
2021-08-13 18:15:50,019 P30750 INFO [Metrics] AUC: 0.746213 - logloss: 0.395836
2021-08-13 18:15:50,580 P30750 INFO ******** Test evaluation ********
2021-08-13 18:15:50,580 P30750 INFO Loading data...
2021-08-13 18:15:50,581 P30750 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/test.h5
2021-08-13 18:15:53,439 P30750 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-08-13 18:15:53,439 P30750 INFO Loading test data done.
2021-08-13 18:16:46,506 P30750 INFO [Metrics] AUC: 0.764880 - logloss: 0.366279


```
