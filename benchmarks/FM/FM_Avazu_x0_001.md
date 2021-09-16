## FM_Avazu_x0_001

A notebook to benchmark FM on Avazu_x0_001 dataset.

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
[Metrics] AUC: 0.761253 - logloss: 0.367738
```


### Logs
```python
2021-01-09 23:51:17,926 P45106 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x0_83355fc7",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "FM",
    "model_id": "FM_avazu_x0_004_d20f45d4",
    "model_root": "./Avazu/FM_avazu_x0/",
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
2021-01-09 23:51:17,927 P45106 INFO Set up feature encoder...
2021-01-09 23:51:17,927 P45106 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x0_83355fc7/feature_encoder.pkl
2021-01-09 23:51:19,142 P45106 INFO Total number of parameters: 14284590.
2021-01-09 23:51:19,143 P45106 INFO Loading data...
2021-01-09 23:51:19,146 P45106 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/train.h5
2021-01-09 23:51:22,579 P45106 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/valid.h5
2021-01-09 23:51:23,134 P45106 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-01-09 23:51:23,135 P45106 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-01-09 23:51:23,135 P45106 INFO Loading train data done.
2021-01-09 23:51:26,717 P45106 INFO Start training: 6910 batches/epoch
2021-01-09 23:51:26,717 P45106 INFO ************ Epoch=1 start ************
2021-01-10 00:27:09,476 P45106 INFO [Metrics] AUC: 0.733208 - logloss: 0.405659
2021-01-10 00:27:09,480 P45106 INFO Save best model: monitor(max): 0.733208
2021-01-10 00:27:09,567 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 00:27:09,730 P45106 INFO Train loss: 0.406628
2021-01-10 00:27:09,731 P45106 INFO ************ Epoch=1 end ************
2021-01-10 01:03:03,946 P45106 INFO [Metrics] AUC: 0.733837 - logloss: 0.405526
2021-01-10 01:03:03,949 P45106 INFO Save best model: monitor(max): 0.733837
2021-01-10 01:03:04,086 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 01:03:04,259 P45106 INFO Train loss: 0.402833
2021-01-10 01:03:04,259 P45106 INFO ************ Epoch=2 end ************
2021-01-10 01:33:46,473 P45106 INFO [Metrics] AUC: 0.735008 - logloss: 0.406557
2021-01-10 01:33:46,477 P45106 INFO Save best model: monitor(max): 0.735008
2021-01-10 01:33:46,609 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 01:33:46,779 P45106 INFO Train loss: 0.402322
2021-01-10 01:33:46,780 P45106 INFO ************ Epoch=3 end ************
2021-01-10 01:45:16,645 P45106 INFO [Metrics] AUC: 0.738006 - logloss: 0.400987
2021-01-10 01:45:16,648 P45106 INFO Save best model: monitor(max): 0.738006
2021-01-10 01:45:16,740 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 01:45:16,855 P45106 INFO Train loss: 0.401940
2021-01-10 01:45:16,855 P45106 INFO ************ Epoch=4 end ************
2021-01-10 01:56:31,259 P45106 INFO [Metrics] AUC: 0.736819 - logloss: 0.402480
2021-01-10 01:56:31,263 P45106 INFO Monitor(max) STOP: 0.736819 !
2021-01-10 01:56:31,263 P45106 INFO Reduce learning rate on plateau: 0.000100
2021-01-10 01:56:31,263 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 01:56:31,355 P45106 INFO Train loss: 0.401683
2021-01-10 01:56:31,355 P45106 INFO ************ Epoch=5 end ************
2021-01-10 02:07:45,136 P45106 INFO [Metrics] AUC: 0.739279 - logloss: 0.400449
2021-01-10 02:07:45,138 P45106 INFO Save best model: monitor(max): 0.739279
2021-01-10 02:07:45,241 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 02:07:45,322 P45106 INFO Train loss: 0.396098
2021-01-10 02:07:45,322 P45106 INFO ************ Epoch=6 end ************
2021-01-10 02:18:59,233 P45106 INFO [Metrics] AUC: 0.738827 - logloss: 0.400667
2021-01-10 02:18:59,237 P45106 INFO Monitor(max) STOP: 0.738827 !
2021-01-10 02:18:59,238 P45106 INFO Reduce learning rate on plateau: 0.000010
2021-01-10 02:18:59,238 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 02:18:59,382 P45106 INFO Train loss: 0.394704
2021-01-10 02:18:59,382 P45106 INFO ************ Epoch=7 end ************
2021-01-10 02:30:33,228 P45106 INFO [Metrics] AUC: 0.738472 - logloss: 0.401330
2021-01-10 02:30:33,232 P45106 INFO Monitor(max) STOP: 0.738472 !
2021-01-10 02:30:33,232 P45106 INFO Reduce learning rate on plateau: 0.000001
2021-01-10 02:30:33,232 P45106 INFO Early stopping at epoch=8
2021-01-10 02:30:33,232 P45106 INFO --- 6910/6910 batches finished ---
2021-01-10 02:30:33,420 P45106 INFO Train loss: 0.393510
2021-01-10 02:30:33,421 P45106 INFO Training finished.
2021-01-10 02:30:33,421 P45106 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Avazu/FM_avazu_x0/avazu_x0_83355fc7/FM_avazu_x0_004_d20f45d4_model.ckpt
2021-01-10 02:30:33,556 P45106 INFO ****** Train/validation evaluation ******
2021-01-10 02:31:01,424 P45106 INFO [Metrics] AUC: 0.739279 - logloss: 0.400449
2021-01-10 02:31:01,578 P45106 INFO ******** Test evaluation ********
2021-01-10 02:31:01,578 P45106 INFO Loading data...
2021-01-10 02:31:01,579 P45106 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/test.h5
2021-01-10 02:31:02,570 P45106 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-01-10 02:31:02,570 P45106 INFO Loading test data done.
2021-01-10 02:31:46,656 P45106 INFO [Metrics] AUC: 0.761253 - logloss: 0.367738


```
