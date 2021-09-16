## DeepIM_Criteo_x0_001

A notebook to benchmark DeepIM on Criteo_x0_001 dataset.

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

Reproducing steps:
Step1: Download the preprocessed data via the [https://github.com/xue-pai/Open-CTR-Benchmark/blob/master/datasets/Criteo/Avazu_x0/download_criteo_x0.py](script).

Criteo_x0_001
In this setting, we follow the AFN work to fix embedding_dim=16, batch_size=4096, and MLP_hidden_units=[400, 400, 400] to make fair comparisons.

### Code




### Results
```python
[Metrics] AUC: 0.814034 - logloss: 0.437877
```


### Logs
```python
2021-06-01 22:15:54,987 P27772 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x0_ace9c1b9",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13'], 'type': 'numeric'}, {'active': True, 'dtype': 'float', 'name': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], 'type': 'categorical'}]",
    "gpu": "0",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "im_batch_norm": "True",
    "im_order": "2",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_criteo_x0_002_90ed257a",
    "model_root": "./Criteo/DeepIM_criteo_x0/",
    "monitor": "AUC",
    "monitor_mode": "max",
    "net_batch_norm": "True",
    "net_dropout": "0.1",
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
    "test_data": "../data/Criteo/Criteo_x0/test.csv",
    "train_data": "../data/Criteo/Criteo_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Criteo/Criteo_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-06-01 22:15:54,988 P27772 INFO Set up feature encoder...
2021-06-01 22:15:54,988 P27772 INFO Load feature_encoder from pickle: ../data/Criteo/criteo_x0_ace9c1b9/feature_encoder.pkl
2021-06-01 22:15:56,562 P27772 INFO Total number of parameters: 21343222.
2021-06-01 22:15:56,563 P27772 INFO Loading data...
2021-06-01 22:15:56,565 P27772 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/train.h5
2021-06-01 22:16:02,779 P27772 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/valid.h5
2021-06-01 22:16:04,517 P27772 INFO Train samples: total/33003326, pos/8456369, neg/24546957, ratio/25.62%, blocks/1
2021-06-01 22:16:04,517 P27772 INFO Validation samples: total/8250124, pos/2114300, neg/6135824, ratio/25.63%, blocks/1
2021-06-01 22:16:04,517 P27772 INFO Loading train data done.
2021-06-01 22:16:08,533 P27772 INFO Start training: 8058 batches/epoch
2021-06-01 22:16:08,533 P27772 INFO ************ Epoch=1 start ************
2021-06-01 22:36:26,600 P27772 INFO [Metrics] AUC: 0.803403 - logloss: 0.447796
2021-06-01 22:36:26,601 P27772 INFO Save best model: monitor(max): 0.803403
2021-06-01 22:36:26,684 P27772 INFO --- 8058/8058 batches finished ---
2021-06-01 22:36:26,790 P27772 INFO Train loss: 0.461975
2021-06-01 22:36:26,790 P27772 INFO ************ Epoch=1 end ************
2021-06-01 22:57:08,972 P27772 INFO [Metrics] AUC: 0.806384 - logloss: 0.445190
2021-06-01 22:57:08,973 P27772 INFO Save best model: monitor(max): 0.806384
2021-06-01 22:57:09,100 P27772 INFO --- 8058/8058 batches finished ---
2021-06-01 22:57:09,167 P27772 INFO Train loss: 0.456497
2021-06-01 22:57:09,167 P27772 INFO ************ Epoch=2 end ************
2021-06-01 23:18:13,077 P27772 INFO [Metrics] AUC: 0.807239 - logloss: 0.444423
2021-06-01 23:18:13,078 P27772 INFO Save best model: monitor(max): 0.807239
2021-06-01 23:18:13,207 P27772 INFO --- 8058/8058 batches finished ---
2021-06-01 23:18:13,278 P27772 INFO Train loss: 0.454991
2021-06-01 23:18:13,278 P27772 INFO ************ Epoch=3 end ************
2021-06-01 23:39:24,566 P27772 INFO [Metrics] AUC: 0.808060 - logloss: 0.443493
2021-06-01 23:39:24,568 P27772 INFO Save best model: monitor(max): 0.808060
2021-06-01 23:39:24,728 P27772 INFO --- 8058/8058 batches finished ---
2021-06-01 23:39:24,811 P27772 INFO Train loss: 0.454272
2021-06-01 23:39:24,812 P27772 INFO ************ Epoch=4 end ************
2021-06-02 00:00:40,305 P27772 INFO [Metrics] AUC: 0.808432 - logloss: 0.443098
2021-06-02 00:00:40,306 P27772 INFO Save best model: monitor(max): 0.808432
2021-06-02 00:00:40,436 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 00:00:40,514 P27772 INFO Train loss: 0.453785
2021-06-02 00:00:40,514 P27772 INFO ************ Epoch=5 end ************
2021-06-02 00:22:03,060 P27772 INFO [Metrics] AUC: 0.808661 - logloss: 0.442928
2021-06-02 00:22:03,061 P27772 INFO Save best model: monitor(max): 0.808661
2021-06-02 00:22:03,209 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 00:22:03,290 P27772 INFO Train loss: 0.453416
2021-06-02 00:22:03,290 P27772 INFO ************ Epoch=6 end ************
2021-06-02 00:43:23,617 P27772 INFO [Metrics] AUC: 0.809017 - logloss: 0.442563
2021-06-02 00:43:23,619 P27772 INFO Save best model: monitor(max): 0.809017
2021-06-02 00:43:23,782 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 00:43:23,863 P27772 INFO Train loss: 0.453163
2021-06-02 00:43:23,863 P27772 INFO ************ Epoch=7 end ************
2021-06-02 01:02:07,289 P27772 INFO [Metrics] AUC: 0.809037 - logloss: 0.442694
2021-06-02 01:02:07,290 P27772 INFO Save best model: monitor(max): 0.809037
2021-06-02 01:02:07,438 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 01:02:07,538 P27772 INFO Train loss: 0.452944
2021-06-02 01:02:07,538 P27772 INFO ************ Epoch=8 end ************
2021-06-02 01:20:52,898 P27772 INFO [Metrics] AUC: 0.809392 - logloss: 0.442277
2021-06-02 01:20:52,900 P27772 INFO Save best model: monitor(max): 0.809392
2021-06-02 01:20:53,030 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 01:20:53,122 P27772 INFO Train loss: 0.452771
2021-06-02 01:20:53,122 P27772 INFO ************ Epoch=9 end ************
2021-06-02 01:39:31,798 P27772 INFO [Metrics] AUC: 0.809423 - logloss: 0.442242
2021-06-02 01:39:31,800 P27772 INFO Save best model: monitor(max): 0.809423
2021-06-02 01:39:31,971 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 01:39:32,054 P27772 INFO Train loss: 0.452607
2021-06-02 01:39:32,054 P27772 INFO ************ Epoch=10 end ************
2021-06-02 01:58:06,234 P27772 INFO [Metrics] AUC: 0.809538 - logloss: 0.442271
2021-06-02 01:58:06,235 P27772 INFO Save best model: monitor(max): 0.809538
2021-06-02 01:58:06,388 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 01:58:06,476 P27772 INFO Train loss: 0.452429
2021-06-02 01:58:06,476 P27772 INFO ************ Epoch=11 end ************
2021-06-02 02:16:43,722 P27772 INFO [Metrics] AUC: 0.809676 - logloss: 0.442008
2021-06-02 02:16:43,723 P27772 INFO Save best model: monitor(max): 0.809676
2021-06-02 02:16:43,860 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 02:16:43,939 P27772 INFO Train loss: 0.452336
2021-06-02 02:16:43,939 P27772 INFO ************ Epoch=12 end ************
2021-06-02 02:35:18,559 P27772 INFO [Metrics] AUC: 0.809832 - logloss: 0.441797
2021-06-02 02:35:18,560 P27772 INFO Save best model: monitor(max): 0.809832
2021-06-02 02:35:18,688 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 02:35:18,762 P27772 INFO Train loss: 0.452214
2021-06-02 02:35:18,762 P27772 INFO ************ Epoch=13 end ************
2021-06-02 02:54:09,873 P27772 INFO [Metrics] AUC: 0.809867 - logloss: 0.441856
2021-06-02 02:54:09,875 P27772 INFO Save best model: monitor(max): 0.809867
2021-06-02 02:54:10,017 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 02:54:10,102 P27772 INFO Train loss: 0.452094
2021-06-02 02:54:10,102 P27772 INFO ************ Epoch=14 end ************
2021-06-02 03:13:13,472 P27772 INFO [Metrics] AUC: 0.810000 - logloss: 0.441646
2021-06-02 03:13:13,474 P27772 INFO Save best model: monitor(max): 0.810000
2021-06-02 03:13:13,608 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 03:13:13,692 P27772 INFO Train loss: 0.452050
2021-06-02 03:13:13,693 P27772 INFO ************ Epoch=15 end ************
2021-06-02 03:32:14,538 P27772 INFO [Metrics] AUC: 0.809941 - logloss: 0.441686
2021-06-02 03:32:14,540 P27772 INFO Monitor(max) STOP: 0.809941 !
2021-06-02 03:32:14,540 P27772 INFO Reduce learning rate on plateau: 0.000100
2021-06-02 03:32:14,540 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 03:32:14,613 P27772 INFO Train loss: 0.451958
2021-06-02 03:32:14,614 P27772 INFO ************ Epoch=16 end ************
2021-06-02 03:51:17,762 P27772 INFO [Metrics] AUC: 0.813082 - logloss: 0.438833
2021-06-02 03:51:17,763 P27772 INFO Save best model: monitor(max): 0.813082
2021-06-02 03:51:17,911 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 03:51:18,001 P27772 INFO Train loss: 0.441508
2021-06-02 03:51:18,001 P27772 INFO ************ Epoch=17 end ************
2021-06-02 04:10:14,392 P27772 INFO [Metrics] AUC: 0.813562 - logloss: 0.438428
2021-06-02 04:10:14,394 P27772 INFO Save best model: monitor(max): 0.813562
2021-06-02 04:10:14,537 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 04:10:14,619 P27772 INFO Train loss: 0.437368
2021-06-02 04:10:14,620 P27772 INFO ************ Epoch=18 end ************
2021-06-02 04:29:11,145 P27772 INFO [Metrics] AUC: 0.813653 - logloss: 0.438385
2021-06-02 04:29:11,147 P27772 INFO Save best model: monitor(max): 0.813653
2021-06-02 04:29:11,276 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 04:29:11,360 P27772 INFO Train loss: 0.435508
2021-06-02 04:29:11,360 P27772 INFO ************ Epoch=19 end ************
2021-06-02 04:48:13,490 P27772 INFO [Metrics] AUC: 0.813571 - logloss: 0.438497
2021-06-02 04:48:13,491 P27772 INFO Monitor(max) STOP: 0.813571 !
2021-06-02 04:48:13,491 P27772 INFO Reduce learning rate on plateau: 0.000010
2021-06-02 04:48:13,491 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 04:48:13,580 P27772 INFO Train loss: 0.434163
2021-06-02 04:48:13,580 P27772 INFO ************ Epoch=20 end ************
2021-06-02 04:58:49,224 P27772 INFO [Metrics] AUC: 0.813155 - logloss: 0.439233
2021-06-02 04:58:49,225 P27772 INFO Monitor(max) STOP: 0.813155 !
2021-06-02 04:58:49,226 P27772 INFO Reduce learning rate on plateau: 0.000001
2021-06-02 04:58:49,226 P27772 INFO Early stopping at epoch=21
2021-06-02 04:58:49,226 P27772 INFO --- 8058/8058 batches finished ---
2021-06-02 04:58:49,305 P27772 INFO Train loss: 0.429961
2021-06-02 04:58:49,305 P27772 INFO Training finished.
2021-06-02 04:58:49,306 P27772 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Criteo/DeepIM_criteo_x0/criteo_x0_ace9c1b9/DeepIM_criteo_x0_002_90ed257a_model.ckpt
2021-06-02 04:58:49,439 P27772 INFO ****** Train/validation evaluation ******
2021-06-02 04:59:16,293 P27772 INFO [Metrics] AUC: 0.813653 - logloss: 0.438385
2021-06-02 04:59:16,336 P27772 INFO ******** Test evaluation ********
2021-06-02 04:59:16,336 P27772 INFO Loading data...
2021-06-02 04:59:16,337 P27772 INFO Loading data from h5: ../data/Criteo/criteo_x0_ace9c1b9/test.h5
2021-06-02 04:59:17,158 P27772 INFO Test samples: total/4587167, pos/1174769, neg/3412398, ratio/25.61%, blocks/1
2021-06-02 04:59:17,159 P27772 INFO Loading test data done.
2021-06-02 04:59:33,314 P27772 INFO [Metrics] AUC: 0.814034 - logloss: 0.437877

```
