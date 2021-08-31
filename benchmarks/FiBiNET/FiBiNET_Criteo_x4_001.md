## FiBiNET_Criteo_x4_001

A notebook to benchmark FiBiNET on Criteo_x4_001 dataset.

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
In this setting, we follow the winner's solution of the Criteo challenge to discretize each integer value x to ⌊log2 (x)⌋, if x > 2; and x = 1 otherwise. For all categorical fields, we replace infrequent features with a default <OOV> token by setting the threshold min_category_count=10. Note that we do not follow the exact preprocessing steps in AutoInt, because this preprocessing performs much better.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.
### Code




### Results
```python
[Metrics] logloss: 0.438314 - AUC: 0.813607
```


### Logs
```python
2020-06-25 21:52:49,804 P2488 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "bilinear_type": "field_interaction",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-06",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "2",
    "hidden_activations": "relu",
    "hidden_units": "[4096, 2048, 1024, 512]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "FiBiNET",
    "model_id": "FiBiNET_criteo_x4_5c863b0f_010_2b723348",
    "model_root": "./Criteo/FiBiNET_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "reduction_ratio": "3",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-25 21:52:49,805 P2488 INFO Set up feature encoder...
2020-06-25 21:52:49,805 P2488 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-25 21:52:49,805 P2488 INFO Loading data...
2020-06-25 21:52:49,807 P2488 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-25 21:52:55,826 P2488 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-25 21:52:57,869 P2488 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-25 21:52:57,992 P2488 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-25 21:52:57,992 P2488 INFO Loading train data done.
2020-06-25 21:53:07,615 P2488 INFO Start training: 3668 batches/epoch
2020-06-25 21:53:07,615 P2488 INFO ************ Epoch=1 start ************
2020-06-26 00:01:28,127 P2488 INFO [Metrics] logloss: 0.440876 - AUC: 0.811099
2020-06-26 00:01:28,131 P2488 INFO Save best model: monitor(max): 0.370223
2020-06-26 00:01:30,387 P2488 INFO --- 3668/3668 batches finished ---
2020-06-26 00:01:30,455 P2488 INFO Train loss: 0.452338
2020-06-26 00:01:30,457 P2488 INFO ************ Epoch=1 end ************
2020-06-26 02:10:27,155 P2488 INFO [Metrics] logloss: 0.438709 - AUC: 0.813169
2020-06-26 02:10:27,156 P2488 INFO Save best model: monitor(max): 0.374460
2020-06-26 02:10:28,897 P2488 INFO --- 3668/3668 batches finished ---
2020-06-26 02:10:28,970 P2488 INFO Train loss: 0.443842
2020-06-26 02:10:28,976 P2488 INFO ************ Epoch=2 end ************
2020-06-26 04:18:46,202 P2488 INFO [Metrics] logloss: 0.441972 - AUC: 0.810466
2020-06-26 04:18:46,203 P2488 INFO Monitor(max) STOP: 0.368494 !
2020-06-26 04:18:46,204 P2488 INFO Reduce learning rate on plateau: 0.000100
2020-06-26 04:18:46,204 P2488 INFO --- 3668/3668 batches finished ---
2020-06-26 04:18:46,321 P2488 INFO Train loss: 0.437679
2020-06-26 04:18:46,324 P2488 INFO ************ Epoch=3 end ************
2020-06-26 06:27:18,281 P2488 INFO [Metrics] logloss: 0.527664 - AUC: 0.775889
2020-06-26 06:27:18,283 P2488 INFO Monitor(max) STOP: 0.248225 !
2020-06-26 06:27:18,283 P2488 INFO Reduce learning rate on plateau: 0.000010
2020-06-26 06:27:18,283 P2488 INFO Early stopping at epoch=4
2020-06-26 06:27:18,283 P2488 INFO --- 3668/3668 batches finished ---
2020-06-26 06:27:18,349 P2488 INFO Train loss: 0.352262
2020-06-26 06:27:18,352 P2488 INFO Training finished.
2020-06-26 06:27:18,352 P2488 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/FiBiNET_criteo/min10/criteo_x4_5c863b0f/FiBiNET_criteo_x4_5c863b0f_010_2b723348_model.ckpt
2020-06-26 06:27:20,061 P2488 INFO ****** Train/validation evaluation ******
2020-06-26 07:05:57,286 P2488 INFO [Metrics] logloss: 0.419918 - AUC: 0.833587
2020-06-26 07:10:43,758 P2488 INFO [Metrics] logloss: 0.438709 - AUC: 0.813169
2020-06-26 07:10:43,862 P2488 INFO ******** Test evaluation ********
2020-06-26 07:10:43,863 P2488 INFO Loading data...
2020-06-26 07:10:43,863 P2488 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-26 07:10:44,944 P2488 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-26 07:10:44,944 P2488 INFO Loading test data done.
2020-06-26 07:15:30,644 P2488 INFO [Metrics] logloss: 0.438314 - AUC: 0.813607

```