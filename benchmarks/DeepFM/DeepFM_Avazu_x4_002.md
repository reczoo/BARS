## DeepFM_Avazu_x4_002

A notebook to benchmark DeepFM on Avazu_x4_002 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=1 and replace them with a default <OOV> token. Note that we found that min_category_count=1 performs the best, which is surprising.

We fix embedding_dim=40 following the existing FGCNN work.
### Code




### Results
```python
[Metrics] logloss: 0.370178 - AUC: 0.796177
```


### Logs
```python
2020-03-03 17:25:15,479 P825 INFO {
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[3000, 3000, 3000, 3000]",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "DeepFM",
    "model_id": "DeepFM_avazu_x4_003_f11d0986",
    "model_root": "./Avazu/DeepFM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "use_hdf5": "True",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "0"
}
2020-03-03 17:25:15,481 P825 INFO Set up feature encoder...
2020-03-03 17:25:15,481 P825 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-03-03 17:25:15,482 P825 INFO Loading data...
2020-03-03 17:25:15,486 P825 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-03-03 17:25:20,154 P825 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-03-03 17:25:22,053 P825 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-03-03 17:25:22,236 P825 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-03-03 17:25:22,236 P825 INFO Loading train data done.
2020-03-03 17:25:36,334 P825 INFO **** Start training: 3235 batches/epoch ****
2020-03-03 17:41:13,475 P825 INFO [Metrics] logloss: 0.370368 - AUC: 0.795802
2020-03-03 17:41:13,570 P825 INFO Save best model: monitor(max): 0.425434
2020-03-03 17:41:15,790 P825 INFO --- 3235/3235 batches finished ---
2020-03-03 17:41:15,867 P825 INFO Train loss: 0.379846
2020-03-03 17:41:15,868 P825 INFO ************ Epoch=1 end ************
2020-03-03 17:56:50,749 P825 INFO [Metrics] logloss: 0.450234 - AUC: 0.761621
2020-03-03 17:56:50,801 P825 INFO Monitor(max) STOP: 0.311386 !
2020-03-03 17:56:50,802 P825 INFO Reduce learning rate on plateau: 0.000100
2020-03-03 17:56:50,802 P825 INFO --- 3235/3235 batches finished ---
2020-03-03 17:56:50,866 P825 INFO Train loss: 0.286010
2020-03-03 17:56:50,866 P825 INFO ************ Epoch=2 end ************
2020-03-03 18:12:24,407 P825 INFO [Metrics] logloss: 0.512873 - AUC: 0.757223
2020-03-03 18:12:24,473 P825 INFO Monitor(max) STOP: 0.244350 !
2020-03-03 18:12:24,473 P825 INFO Reduce learning rate on plateau: 0.000010
2020-03-03 18:12:24,473 P825 INFO Early stopping at epoch=3
2020-03-03 18:12:24,473 P825 INFO --- 3235/3235 batches finished ---
2020-03-03 18:12:24,535 P825 INFO Train loss: 0.243807
2020-03-03 18:12:24,536 P825 INFO Training finished.
2020-03-03 18:12:24,536 P825 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Avazu/DeepFM_avazu/avazu_x4_001_d45ad60e/DeepFM_avazu_x4_003_f11d0986_avazu_x4_001_d45ad60e_model.ckpt
2020-03-03 18:12:27,391 P825 INFO ****** Train/validation evaluation ******
2020-03-03 18:16:58,624 P825 INFO [Metrics] logloss: 0.318769 - AUC: 0.868529
2020-03-03 18:17:32,060 P825 INFO [Metrics] logloss: 0.370368 - AUC: 0.795802
2020-03-03 18:17:32,215 P825 INFO ******** Test evaluation ********
2020-03-03 18:17:32,215 P825 INFO Loading data...
2020-03-03 18:17:32,216 P825 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-03-03 18:17:32,996 P825 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-03-03 18:17:32,997 P825 INFO Loading test data done.
2020-03-03 18:18:06,729 P825 INFO [Metrics] logloss: 0.370178 - AUC: 0.796177

```
