## DeepFM_Avazu_x0_001

A notebook to benchmark DeepFM on Avazu_x0_001 dataset.

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
[Metrics] AUC: 0.765736 - logloss: 0.366209
```


### Logs
```python
2021-01-09 08:16:30,313 P20256 INFO {
    "batch_norm": "True",
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x0_83355fc7",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepFM",
    "model_id": "DeepFM_avazu_x0_002_0e0eb50c",
    "model_root": "./Avazu/DeepFM_avazu_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_dropout": "0.2",
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
    "test_data": "../data/Avazu/Avazu_x0/test.csv",
    "train_data": "../data/Avazu/Avazu_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-01-09 08:16:30,314 P20256 INFO Set up feature encoder...
2021-01-09 08:16:30,314 P20256 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x0_83355fc7/feature_encoder.pkl
2021-01-09 08:16:31,383 P20256 INFO Total number of parameters: 14696590.
2021-01-09 08:16:31,383 P20256 INFO Loading data...
2021-01-09 08:16:31,387 P20256 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/train.h5
2021-01-09 08:16:39,820 P20256 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/valid.h5
2021-01-09 08:16:41,890 P20256 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-01-09 08:16:41,890 P20256 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-01-09 08:16:41,890 P20256 INFO Loading train data done.
2021-01-09 08:16:46,248 P20256 INFO Start training: 6910 batches/epoch
2021-01-09 08:16:46,248 P20256 INFO ************ Epoch=1 start ************
2021-01-09 08:56:25,232 P20256 INFO [Metrics] AUC: 0.737062 - logloss: 0.401829
2021-01-09 08:56:25,234 P20256 INFO Save best model: monitor(max): 0.737062
2021-01-09 08:56:25,300 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 08:56:25,401 P20256 INFO Train loss: 0.440793
2021-01-09 08:56:25,401 P20256 INFO ************ Epoch=1 end ************
2021-01-09 09:36:03,950 P20256 INFO [Metrics] AUC: 0.737725 - logloss: 0.399957
2021-01-09 09:36:03,963 P20256 INFO Save best model: monitor(max): 0.737725
2021-01-09 09:36:04,202 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 09:36:04,322 P20256 INFO Train loss: 0.438596
2021-01-09 09:36:04,322 P20256 INFO ************ Epoch=2 end ************
2021-01-09 10:15:38,595 P20256 INFO [Metrics] AUC: 0.735084 - logloss: 0.402820
2021-01-09 10:15:38,598 P20256 INFO Monitor(max) STOP: 0.735084 !
2021-01-09 10:15:38,598 P20256 INFO Reduce learning rate on plateau: 0.000100
2021-01-09 10:15:38,598 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 10:15:38,727 P20256 INFO Train loss: 0.438153
2021-01-09 10:15:38,727 P20256 INFO ************ Epoch=3 end ************
2021-01-09 10:55:00,551 P20256 INFO [Metrics] AUC: 0.746627 - logloss: 0.395953
2021-01-09 10:55:00,554 P20256 INFO Save best model: monitor(max): 0.746627
2021-01-09 10:55:00,653 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 10:55:00,992 P20256 INFO Train loss: 0.409575
2021-01-09 10:55:00,992 P20256 INFO ************ Epoch=4 end ************
2021-01-09 11:34:15,148 P20256 INFO [Metrics] AUC: 0.746102 - logloss: 0.396331
2021-01-09 11:34:15,151 P20256 INFO Monitor(max) STOP: 0.746102 !
2021-01-09 11:34:15,151 P20256 INFO Reduce learning rate on plateau: 0.000010
2021-01-09 11:34:15,151 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 11:34:15,269 P20256 INFO Train loss: 0.411395
2021-01-09 11:34:15,269 P20256 INFO ************ Epoch=5 end ************
2021-01-09 12:23:06,494 P20256 INFO [Metrics] AUC: 0.747280 - logloss: 0.395113
2021-01-09 12:23:06,498 P20256 INFO Save best model: monitor(max): 0.747280
2021-01-09 12:23:06,654 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 12:23:06,755 P20256 INFO Train loss: 0.397640
2021-01-09 12:23:06,755 P20256 INFO ************ Epoch=6 end ************
2021-01-09 13:39:57,048 P20256 INFO [Metrics] AUC: 0.744998 - logloss: 0.396533
2021-01-09 13:39:57,077 P20256 INFO Monitor(max) STOP: 0.744998 !
2021-01-09 13:39:57,077 P20256 INFO Reduce learning rate on plateau: 0.000001
2021-01-09 13:39:57,103 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 13:39:58,130 P20256 INFO Train loss: 0.395257
2021-01-09 13:39:58,130 P20256 INFO ************ Epoch=7 end ************
2021-01-09 15:00:30,642 P20256 INFO [Metrics] AUC: 0.739567 - logloss: 0.399661
2021-01-09 15:00:30,692 P20256 INFO Monitor(max) STOP: 0.739567 !
2021-01-09 15:00:30,693 P20256 INFO Reduce learning rate on plateau: 0.000001
2021-01-09 15:00:30,693 P20256 INFO Early stopping at epoch=8
2021-01-09 15:00:30,693 P20256 INFO --- 6910/6910 batches finished ---
2021-01-09 15:00:32,170 P20256 INFO Train loss: 0.388371
2021-01-09 15:00:32,170 P20256 INFO Training finished.
2021-01-09 15:00:32,170 P20256 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Avazu/DeepFM_avazu_x0/avazu_x0_83355fc7/DeepFM_avazu_x0_002_0e0eb50c_model.ckpt
2021-01-09 15:00:33,597 P20256 INFO ****** Train/validation evaluation ******
2021-01-09 15:03:38,978 P20256 INFO [Metrics] AUC: 0.747280 - logloss: 0.395113
2021-01-09 15:03:40,165 P20256 INFO ******** Test evaluation ********
2021-01-09 15:03:40,166 P20256 INFO Loading data...
2021-01-09 15:03:40,166 P20256 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/test.h5
2021-01-09 15:03:50,208 P20256 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-01-09 15:03:50,209 P20256 INFO Loading test data done.
2021-01-09 15:10:04,120 P20256 INFO [Metrics] AUC: 0.765736 - logloss: 0.366209


```
