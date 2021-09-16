## HOFM_Avazu_x4_002

A notebook to benchmark HOFM on Avazu_x4_002 dataset.

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
[Metrics] logloss: 0.373289 - AUC: 0.791393
```


### Logs
```python
2020-03-06 23:30:08,593 P17122 INFO {
    "batch_size": "5000",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "embedding_dim": "[40, 5]",
    "embedding_dropout": "0",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "HOFM",
    "model_id": "HOFM_avazu_x4_001_63b17d4d",
    "model_root": "./Avazu/HOFM_avazu/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "optimizer": "adam",
    "order": "3",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "regularizer": "0",
    "reuse_embedding": "False",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-03-06 23:30:08,594 P17122 INFO Set up feature encoder...
2020-03-06 23:30:08,594 P17122 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-03-06 23:30:08,595 P17122 INFO Loading data...
2020-03-06 23:30:08,597 P17122 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-03-06 23:30:13,606 P17122 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-03-06 23:30:20,541 P17122 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-03-06 23:30:20,722 P17122 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-03-06 23:30:20,722 P17122 INFO Loading train data done.
2020-03-06 23:30:56,193 P17122 INFO **** Start training: 6469 batches/epoch ****
2020-03-07 00:45:43,884 P17122 INFO [Metrics] logloss: 0.373340 - AUC: 0.791282
2020-03-07 00:45:43,995 P17122 INFO Save best model: monitor(max): 0.417942
2020-03-07 00:46:09,413 P17122 INFO --- 6469/6469 batches finished ---
2020-03-07 00:46:09,474 P17122 INFO Train loss: 0.383256
2020-03-07 00:46:09,474 P17122 INFO ************ Epoch=1 end ************
2020-03-07 02:00:46,445 P17122 INFO [Metrics] logloss: 0.384590 - AUC: 0.786303
2020-03-07 02:00:46,537 P17122 INFO Monitor(max) STOP: 0.401713 !
2020-03-07 02:00:46,538 P17122 INFO Reduce learning rate on plateau: 0.000100
2020-03-07 02:00:46,538 P17122 INFO --- 6469/6469 batches finished ---
2020-03-07 02:00:46,637 P17122 INFO Train loss: 0.324082
2020-03-07 02:00:46,637 P17122 INFO ************ Epoch=2 end ************
2020-03-07 03:15:24,717 P17122 INFO [Metrics] logloss: 0.406113 - AUC: 0.778066
2020-03-07 03:15:24,809 P17122 INFO Monitor(max) STOP: 0.371953 !
2020-03-07 03:15:24,809 P17122 INFO Reduce learning rate on plateau: 0.000010
2020-03-07 03:15:24,809 P17122 INFO Early stopping at epoch=3
2020-03-07 03:15:24,809 P17122 INFO --- 6469/6469 batches finished ---
2020-03-07 03:15:24,939 P17122 INFO Train loss: 0.266659
2020-03-07 03:15:24,939 P17122 INFO Training finished.
2020-03-07 03:15:24,939 P17122 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Avazu/HOFM_avazu/avazu_x4_001_d45ad60e/HOFM_avazu_x4_001_63b17d4d_avazu_x4_001_d45ad60e_model.ckpt
2020-03-07 03:16:00,904 P17122 INFO ****** Train/validation evaluation ******
2020-03-07 03:36:26,898 P17122 INFO [Metrics] logloss: 0.331442 - AUC: 0.854117
2020-03-07 03:38:46,701 P17122 INFO [Metrics] logloss: 0.373340 - AUC: 0.791282
2020-03-07 03:38:46,898 P17122 INFO ******** Test evaluation ********
2020-03-07 03:38:46,898 P17122 INFO Loading data...
2020-03-07 03:38:46,898 P17122 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-03-07 03:38:47,441 P17122 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-03-07 03:38:47,441 P17122 INFO Loading test data done.
2020-03-07 03:41:05,363 P17122 INFO [Metrics] logloss: 0.373289 - AUC: 0.791393

```
