## FmFM_Avazu_x0_001

A notebook to benchmark FmFM on Avazu_x0_001 dataset.

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
[Metrics] AUC: 0.760326 - logloss: 0.368482
```


### Logs
```python
2021-04-17 15:36:31,453 P12370 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x0_83355fc7",
    "debug": "False",
    "embedding_dim": "10",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "field_interaction_type": "matrixed",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FmFM",
    "model_id": "FmFM_avazu_x0_002_5d3dda38",
    "model_root": "./Avazu/FmFM_avazu_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "num_workers": "3",
    "optimizer": "adam",
    "partition_block_size": "-1",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "1e-05",
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
2021-04-17 15:36:31,453 P12370 INFO Set up feature encoder...
2021-04-17 15:36:31,453 P12370 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x0_83355fc7/feature_encoder.pkl
2021-04-17 15:36:36,126 P12370 INFO Total number of parameters: 14307690.
2021-04-17 15:36:36,126 P12370 INFO Loading data...
2021-04-17 15:36:36,129 P12370 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/train.h5
2021-04-17 15:36:38,604 P12370 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/valid.h5
2021-04-17 15:36:38,954 P12370 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-04-17 15:36:38,954 P12370 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-04-17 15:36:38,954 P12370 INFO Loading train data done.
2021-04-17 15:36:38,966 P12370 INFO Start training: 6910 batches/epoch
2021-04-17 15:36:38,966 P12370 INFO ************ Epoch=1 start ************
2021-04-17 15:51:07,823 P12370 INFO [Metrics] AUC: 0.742191 - logloss: 0.401361
2021-04-17 15:51:07,824 P12370 INFO Save best model: monitor(max): 0.742191
2021-04-17 15:51:07,884 P12370 INFO --- 6910/6910 batches finished ---
2021-04-17 15:51:07,956 P12370 INFO Train loss: 0.406828
2021-04-17 15:51:07,956 P12370 INFO ************ Epoch=1 end ************
2021-04-17 16:05:36,015 P12370 INFO [Metrics] AUC: 0.743363 - logloss: 0.397812
2021-04-17 16:05:36,018 P12370 INFO Save best model: monitor(max): 0.743363
2021-04-17 16:05:36,125 P12370 INFO --- 6910/6910 batches finished ---
2021-04-17 16:05:36,185 P12370 INFO Train loss: 0.400795
2021-04-17 16:05:36,185 P12370 INFO ************ Epoch=2 end ************
2021-04-17 16:20:03,467 P12370 INFO [Metrics] AUC: 0.743544 - logloss: 0.398433
2021-04-17 16:20:03,472 P12370 INFO Save best model: monitor(max): 0.743544
2021-04-17 16:20:03,598 P12370 INFO --- 6910/6910 batches finished ---
2021-04-17 16:20:03,649 P12370 INFO Train loss: 0.398320
2021-04-17 16:20:03,649 P12370 INFO ************ Epoch=3 end ************
2021-04-17 16:34:28,699 P12370 INFO [Metrics] AUC: 0.744064 - logloss: 0.397040
2021-04-17 16:34:28,703 P12370 INFO Save best model: monitor(max): 0.744064
2021-04-17 16:34:28,846 P12370 INFO --- 6910/6910 batches finished ---
2021-04-17 16:34:28,914 P12370 INFO Train loss: 0.397246
2021-04-17 16:34:28,914 P12370 INFO ************ Epoch=4 end ************
2021-04-17 16:48:55,083 P12370 INFO [Metrics] AUC: 0.743626 - logloss: 0.398996
2021-04-17 16:48:55,087 P12370 INFO Monitor(max) STOP: 0.743626 !
2021-04-17 16:48:55,087 P12370 INFO Reduce learning rate on plateau: 0.000100
2021-04-17 16:48:55,087 P12370 INFO --- 6910/6910 batches finished ---
2021-04-17 16:48:55,165 P12370 INFO Train loss: 0.396680
2021-04-17 16:48:55,166 P12370 INFO ************ Epoch=5 end ************
2021-04-17 17:03:24,766 P12370 INFO [Metrics] AUC: 0.737642 - logloss: 0.402059
2021-04-17 17:03:24,769 P12370 INFO Monitor(max) STOP: 0.737642 !
2021-04-17 17:03:24,769 P12370 INFO Reduce learning rate on plateau: 0.000010
2021-04-17 17:03:24,769 P12370 INFO Early stopping at epoch=6
2021-04-17 17:03:24,769 P12370 INFO --- 6910/6910 batches finished ---
2021-04-17 17:03:24,848 P12370 INFO Train loss: 0.386142
2021-04-17 17:03:24,848 P12370 INFO Training finished.
2021-04-17 17:03:24,848 P12370 INFO Load best model: /home/xxx/xxx/FuxiCTR_v2/FuxiCTR/benchmarks/Avazu/FmFM_avazu_x0/avazu_x0_83355fc7/FmFM_avazu_x0_002_5d3dda38.model
2021-04-17 17:03:24,908 P12370 INFO ****** Train/validation evaluation ******
2021-04-17 17:03:52,156 P12370 INFO [Metrics] AUC: 0.744064 - logloss: 0.397040
2021-04-17 17:03:52,195 P12370 INFO ******** Test evaluation ********
2021-04-17 17:03:52,195 P12370 INFO Loading data...
2021-04-17 17:03:52,196 P12370 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/test.h5
2021-04-17 17:03:52,908 P12370 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-04-17 17:03:52,909 P12370 INFO Loading test data done.
2021-04-17 17:04:47,903 P12370 INFO [Metrics] AUC: 0.760326 - logloss: 0.368482

```
