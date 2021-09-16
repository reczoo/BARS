## HFM+_Avazu_x4_002

A notebook to benchmark HFM+ on Avazu_x4_002 dataset.

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
[Metrics] logloss: 0.368309 - AUC: 0.799247
```


### Logs
```python
2020-05-09 15:48:47,141 P16570 INFO {
    "batch_norm": "False",
    "batch_size": "5000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "hidden_activations": "relu",
    "hidden_units": "[1000, 1000]",
    "interaction_type": "circular_correlation",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HFM",
    "model_id": "HFM_avazu_x4_001_ca72704b",
    "model_root": "./Avazu/HFM_avazu/",
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
    "use_dnn": "True",
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
2020-05-09 15:48:47,142 P16570 INFO Set up feature encoder...
2020-05-09 15:48:47,142 P16570 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-05-09 15:48:47,143 P16570 INFO Loading data...
2020-05-09 15:48:47,145 P16570 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-05-09 15:48:49,437 P16570 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-05-09 15:48:50,684 P16570 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-05-09 15:48:50,805 P16570 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-05-09 15:48:50,805 P16570 INFO Loading train data done.
2020-05-09 15:49:03,819 P16570 INFO **** Start training: 6469 batches/epoch ****
2020-05-09 16:27:09,822 P16570 INFO [Metrics] logloss: 0.368465 - AUC: 0.798969
2020-05-09 16:27:09,910 P16570 INFO Save best model: monitor(max): 0.430504
2020-05-09 16:27:11,215 P16570 INFO --- 6469/6469 batches finished ---
2020-05-09 16:27:11,265 P16570 INFO Train loss: 0.377912
2020-05-09 16:27:11,265 P16570 INFO ************ Epoch=1 end ************
2020-05-09 17:05:14,468 P16570 INFO [Metrics] logloss: 0.398056 - AUC: 0.778096
2020-05-09 17:05:14,550 P16570 INFO Monitor(max) STOP: 0.380040 !
2020-05-09 17:05:14,550 P16570 INFO Reduce learning rate on plateau: 0.000100
2020-05-09 17:05:14,550 P16570 INFO --- 6469/6469 batches finished ---
2020-05-09 17:05:14,647 P16570 INFO Train loss: 0.282584
2020-05-09 17:05:14,647 P16570 INFO ************ Epoch=2 end ************
2020-05-09 17:43:17,220 P16570 INFO [Metrics] logloss: 0.477289 - AUC: 0.761235
2020-05-09 17:43:17,339 P16570 INFO Monitor(max) STOP: 0.283946 !
2020-05-09 17:43:17,339 P16570 INFO Reduce learning rate on plateau: 0.000010
2020-05-09 17:43:17,339 P16570 INFO Early stopping at epoch=3
2020-05-09 17:43:17,340 P16570 INFO --- 6469/6469 batches finished ---
2020-05-09 17:43:17,426 P16570 INFO Train loss: 0.232089
2020-05-09 17:43:17,426 P16570 INFO Training finished.
2020-05-09 17:43:17,426 P16570 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Avazu/HFM_avazu/avazu_x4_001_d45ad60e/HFM_avazu_x4_001_ca72704b_avazu_x4_001_d45ad60e_model.ckpt
2020-05-09 17:43:19,214 P16570 INFO ****** Train/validation evaluation ******
2020-05-09 17:52:20,999 P16570 INFO [Metrics] logloss: 0.313452 - AUC: 0.877523
2020-05-09 17:53:28,393 P16570 INFO [Metrics] logloss: 0.368465 - AUC: 0.798969
2020-05-09 17:53:28,605 P16570 INFO ******** Test evaluation ********
2020-05-09 17:53:28,605 P16570 INFO Loading data...
2020-05-09 17:53:28,605 P16570 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-05-09 17:53:29,131 P16570 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-05-09 17:53:29,131 P16570 INFO Loading test data done.
2020-05-09 17:54:35,698 P16570 INFO [Metrics] logloss: 0.368309 - AUC: 0.799247

```
