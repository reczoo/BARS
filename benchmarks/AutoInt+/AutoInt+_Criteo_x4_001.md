## AutoInt+_Criteo_x4_001

A notebook to benchmark AutoInt+ on Criteo_x4_001 dataset.

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
This dataset split follows the setting in the AutoInt work. Specifically, we randomly split the data into 8:1:1 as the training set, validation set, and test set, respectively. To make it exactly reproducible, we reuse the code provided by AutoInt and control the random seed (i.e., seed=2018) for splitting.

### Code




### Results
```python
 [Metrics] logloss: 0.439023 - AUC: 0.813240
```


### Logs
```python
2020-06-28 21:24:42,569 P2689 INFO {
    "attention_dim": "64",
    "attention_layers": "5",
    "batch_norm": "False",
    "batch_size": "10000",
    "data_format": "h5",
    "data_root": "../data/Criteo/",
    "dataset_id": "criteo_x4_5c863b0f",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1000, 1000, 1000, 1000, 1000]",
    "embedding_dim": "16",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-05",
    "epochs": "100",
    "every_x_epochs": "1",
    "gpu": "0",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_criteo_x4_5c863b0f_009_11d8455a",
    "model_root": "./Criteo/AutoInt_criteo/min10/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0",
    "net_regularizer": "0",
    "num_heads": "1",
    "optimizer": "adam",
    "patience": "2",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "2019",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/Criteo/criteo_x4_5c863b0f/test.h5",
    "train_data": "../data/Criteo/criteo_x4_5c863b0f/train.h5",
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "True",
    "valid_data": "../data/Criteo/criteo_x4_5c863b0f/valid.h5",
    "verbose": "0",
    "version": "pytorch",
    "workers": "3"
}
2020-06-28 21:24:42,570 P2689 INFO Set up feature encoder...
2020-06-28 21:24:42,570 P2689 INFO Load feature_map from json: ../data/Criteo/criteo_x4_5c863b0f/feature_map.json
2020-06-28 21:24:42,570 P2689 INFO Loading data...
2020-06-28 21:24:42,572 P2689 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/train.h5
2020-06-28 21:25:34,144 P2689 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/valid.h5
2020-06-28 21:26:37,198 P2689 INFO Train samples: total/36672493, pos/9396350, neg/27276143, ratio/25.62%
2020-06-28 21:26:37,400 P2689 INFO Validation samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-28 21:26:37,401 P2689 INFO Loading train data done.
2020-06-28 21:27:48,668 P2689 INFO Start training: 3668 batches/epoch
2020-06-28 21:27:48,668 P2689 INFO ************ Epoch=1 start ************
2020-06-28 21:46:11,149 P2689 INFO [Metrics] logloss: 0.445258 - AUC: 0.806088
2020-06-28 21:46:11,162 P2689 INFO Save best model: monitor(max): 0.360830
2020-06-28 21:46:11,254 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 21:46:11,371 P2689 INFO Train loss: 0.458549
2020-06-28 21:46:11,372 P2689 INFO ************ Epoch=1 end ************
2020-06-28 22:04:04,103 P2689 INFO [Metrics] logloss: 0.442971 - AUC: 0.808422
2020-06-28 22:04:04,104 P2689 INFO Save best model: monitor(max): 0.365451
2020-06-28 22:04:04,228 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 22:04:04,379 P2689 INFO Train loss: 0.452851
2020-06-28 22:04:04,380 P2689 INFO ************ Epoch=2 end ************
2020-06-28 22:21:26,834 P2689 INFO [Metrics] logloss: 0.441857 - AUC: 0.809751
2020-06-28 22:21:26,840 P2689 INFO Save best model: monitor(max): 0.367894
2020-06-28 22:21:26,951 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 22:21:27,105 P2689 INFO Train loss: 0.450963
2020-06-28 22:21:27,105 P2689 INFO ************ Epoch=3 end ************
2020-06-28 22:39:07,251 P2689 INFO [Metrics] logloss: 0.441223 - AUC: 0.810363
2020-06-28 22:39:07,254 P2689 INFO Save best model: monitor(max): 0.369141
2020-06-28 22:39:07,375 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 22:39:07,497 P2689 INFO Train loss: 0.449708
2020-06-28 22:39:07,497 P2689 INFO ************ Epoch=4 end ************
2020-06-28 22:56:28,405 P2689 INFO [Metrics] logloss: 0.441274 - AUC: 0.810354
2020-06-28 22:56:28,408 P2689 INFO Monitor(max) STOP: 0.369080 !
2020-06-28 22:56:28,409 P2689 INFO Reduce learning rate on plateau: 0.000100
2020-06-28 22:56:28,409 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 22:56:28,496 P2689 INFO Train loss: 0.448741
2020-06-28 22:56:28,496 P2689 INFO ************ Epoch=5 end ************
2020-06-28 23:13:42,346 P2689 INFO [Metrics] logloss: 0.439407 - AUC: 0.812768
2020-06-28 23:13:42,348 P2689 INFO Save best model: monitor(max): 0.373362
2020-06-28 23:13:42,460 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 23:13:42,545 P2689 INFO Train loss: 0.435892
2020-06-28 23:13:42,545 P2689 INFO ************ Epoch=6 end ************
2020-06-28 23:32:16,274 P2689 INFO [Metrics] logloss: 0.440149 - AUC: 0.812357
2020-06-28 23:32:16,283 P2689 INFO Monitor(max) STOP: 0.372208 !
2020-06-28 23:32:16,283 P2689 INFO Reduce learning rate on plateau: 0.000010
2020-06-28 23:32:16,284 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 23:32:16,402 P2689 INFO Train loss: 0.430270
2020-06-28 23:32:16,402 P2689 INFO ************ Epoch=7 end ************
2020-06-28 23:50:09,624 P2689 INFO [Metrics] logloss: 0.441880 - AUC: 0.811230
2020-06-28 23:50:09,629 P2689 INFO Monitor(max) STOP: 0.369350 !
2020-06-28 23:50:09,630 P2689 INFO Reduce learning rate on plateau: 0.000001
2020-06-28 23:50:09,630 P2689 INFO Early stopping at epoch=8
2020-06-28 23:50:09,630 P2689 INFO --- 3668/3668 batches finished ---
2020-06-28 23:50:09,750 P2689 INFO Train loss: 0.424580
2020-06-28 23:50:09,751 P2689 INFO Training finished.
2020-06-28 23:50:09,751 P2689 INFO Load best model: /cache/xxx/FuxiCTR/benchmarks/Criteo/AutoInt_criteo/min10/criteo_x4_5c863b0f/AutoInt_criteo_x4_5c863b0f_009_11d8455a_model.ckpt
2020-06-28 23:50:09,888 P2689 INFO ****** Train/validation evaluation ******
2020-06-29 00:05:44,920 P2689 INFO [Metrics] logloss: 0.422455 - AUC: 0.829901
2020-06-29 00:07:00,069 P2689 INFO [Metrics] logloss: 0.439407 - AUC: 0.812768
2020-06-29 00:07:00,254 P2689 INFO ******** Test evaluation ********
2020-06-29 00:07:00,254 P2689 INFO Loading data...
2020-06-29 00:07:00,255 P2689 INFO Loading data from h5: ../data/Criteo/criteo_x4_5c863b0f/test.h5
2020-06-29 00:07:01,097 P2689 INFO Test samples: total/4584062, pos/1174544, neg/3409518, ratio/25.62%
2020-06-29 00:07:01,097 P2689 INFO Loading test data done.
2020-06-29 00:07:31,592 P2689 INFO [Metrics] logloss: 0.439023 - AUC: 0.813240


```
