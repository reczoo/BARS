## AutoInt+_Avazu_x4_002

A notebook to benchmark AutoInt+ on Avazu_x4_002 dataset.

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
[Metrics] logloss: 0.370881 - AUC: 0.795305
```


### Logs
```python
2020-02-22 16:07:47,132 P15866 INFO {
    "attention_dim": "64",
    "attention_layers": "3",
    "batch_norm": "False",
    "batch_size": "10000",
    "dataset_id": "avazu_x4_001_d45ad60e",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[1000, 1000, 1000]",
    "embedding_dim": "40",
    "embedding_dropout": "0",
    "embedding_regularizer": "1e-08",
    "epochs": "100",
    "every_x_epochs": "1",
    "layer_norm": "False",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "model": "AutoInt",
    "model_id": "AutoInt_avazu_x4_003_ecbf7d57",
    "model_root": "./Avazu/AutoInt_avazu/",
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
    "use_hdf5": "True",
    "use_residual": "True",
    "use_scale": "False",
    "use_wide": "False",
    "verbose": "0",
    "workers": "3",
    "data_format": "h5",
    "data_root": "../data/Avazu/",
    "test_data": "../data/Avazu/avazu_x4_001_d45ad60e/test.h5",
    "train_data": "../data/Avazu/avazu_x4_001_d45ad60e/train.h5",
    "valid_data": "../data/Avazu/avazu_x4_001_d45ad60e/valid.h5",
    "version": "pytorch",
    "gpu": "1"
}
2020-02-22 16:07:47,132 P15866 INFO Set up feature encoder...
2020-02-22 16:07:47,133 P15866 INFO Load feature_map from json: ../data/Avazu/avazu_x4_001_d45ad60e/feature_map.json
2020-02-22 16:07:47,133 P15866 INFO Loading data...
2020-02-22 16:07:47,135 P15866 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/train.h5
2020-02-22 16:07:49,590 P15866 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/valid.h5
2020-02-22 16:07:50,908 P15866 INFO Train samples: total/32343172, pos/5492052, neg/26851120, ratio/16.98%
2020-02-22 16:07:51,028 P15866 INFO Validation samples: total/4042897, pos/686507, neg/3356390, ratio/16.98%
2020-02-22 16:07:51,028 P15866 INFO Loading train data done.
2020-02-22 16:08:03,231 P15866 INFO **** Start training: 3235 batches/epoch ****
2020-02-22 16:23:14,927 P15866 INFO [Metrics] logloss: 0.371017 - AUC: 0.795091
2020-02-22 16:23:14,989 P15866 INFO Save best model: monitor(max): 0.424074
2020-02-22 16:23:16,204 P15866 INFO --- 3235/3235 batches finished ---
2020-02-22 16:23:16,240 P15866 INFO Train loss: 0.380972
2020-02-22 16:23:16,240 P15866 INFO ************ Epoch=1 end ************
2020-02-22 16:38:27,868 P15866 INFO [Metrics] logloss: 0.420033 - AUC: 0.760823
2020-02-22 16:38:27,936 P15866 INFO Monitor(max) STOP: 0.340789 !
2020-02-22 16:38:27,936 P15866 INFO Reduce learning rate on plateau: 0.000100
2020-02-22 16:38:27,936 P15866 INFO --- 3235/3235 batches finished ---
2020-02-22 16:38:28,015 P15866 INFO Train loss: 0.289368
2020-02-22 16:38:28,015 P15866 INFO ************ Epoch=2 end ************
2020-02-22 16:53:42,071 P15866 INFO [Metrics] logloss: 0.501465 - AUC: 0.764058
2020-02-22 16:53:42,147 P15866 INFO Monitor(max) STOP: 0.262594 !
2020-02-22 16:53:42,147 P15866 INFO Reduce learning rate on plateau: 0.000010
2020-02-22 16:53:42,147 P15866 INFO Early stopping at epoch=3
2020-02-22 16:53:42,147 P15866 INFO --- 3235/3235 batches finished ---
2020-02-22 16:53:42,223 P15866 INFO Train loss: 0.251849
2020-02-22 16:53:42,223 P15866 INFO Training finished.
2020-02-22 16:53:42,223 P15866 INFO Load best model: /home/hispace/container/data/xxx/FuxiCTR/benchmarks/Avazu/AutoInt_avazu/avazu_x4_001_d45ad60e/AutoInt_avazu_x4_003_ecbf7d57_avazu_x4_001_d45ad60e_model.ckpt
2020-02-22 16:53:43,948 P15866 INFO ****** Train/validation evaluation ******
2020-02-22 16:58:02,045 P15866 INFO [Metrics] logloss: 0.323630 - AUC: 0.867607
2020-02-22 16:58:34,471 P15866 INFO [Metrics] logloss: 0.371017 - AUC: 0.795091
2020-02-22 16:58:34,671 P15866 INFO ******** Test evaluation ********
2020-02-22 16:58:34,671 P15866 INFO Loading data...
2020-02-22 16:58:34,671 P15866 INFO Loading data from h5: ../data/Avazu/avazu_x4_001_d45ad60e/test.h5
2020-02-22 16:58:35,242 P15866 INFO Test samples: total/4042898, pos/686507, neg/3356391, ratio/16.98%
2020-02-22 16:58:35,242 P15866 INFO Loading test data done.
2020-02-22 16:59:07,527 P15866 INFO [Metrics] logloss: 0.370881 - AUC: 0.795305

```
