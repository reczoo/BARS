## DCN_Avazu_x0_001

A notebook to benchmark DCN on Avazu_x0_001 dataset.

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
In this setting, we preprocess the data split by removing the id field that is useless for CTR prediction. In addition, we transform the timestamp field into three fields: hour, weekday, and is_weekend. For all categorical fields, we filter infrequent features by setting the threshold min_category_count=2 (performs well) and replace them with a default <OOV> token. Note that we do not follow the exact preprocessing steps in AutoInt, because the authors neither remove the useless id field nor specially preprocess the timestamp field.

To make a fair comparison, we fix embedding_dim=16 as with AutoInt.


### Code




### Results
```python
[Metrics] logloss: 0.366885 - AUC: 0.764671
```


### Logs
```python
2020-12-27 20:44:03,658 P51456 INFO {
    "batch_norm": "False",
    "batch_size": "4096",
    "crossing_layers": "5",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x0_83355fc7",
    "debug": "False",
    "dnn_activations": "relu",
    "dnn_hidden_units": "[400, 400, 400]",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.001",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "0",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['logloss', 'AUC']",
    "min_categr_count": "1",
    "model": "DCN",
    "model_id": "DCN_avazu_x0_010_d62fb15d",
    "model_root": "./Avazu/DCN_avazu_x0/",
    "monitor": "{'AUC': 1, 'logloss': -1}",
    "monitor_mode": "max",
    "net_dropout": "0.3",
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
2020-12-27 20:44:03,659 P51456 INFO Set up feature encoder...
2020-12-27 20:44:03,659 P51456 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x0_83355fc7/feature_encoder.pkl
2020-12-27 20:44:04,823 P51456 INFO Total number of parameters: 13398011.
2020-12-27 20:44:04,823 P51456 INFO Loading data...
2020-12-27 20:44:04,826 P51456 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/train.h5
2020-12-27 20:44:11,120 P51456 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/valid.h5
2020-12-27 20:44:12,065 P51456 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2020-12-27 20:44:12,065 P51456 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2020-12-27 20:44:12,066 P51456 INFO Loading train data done.
2020-12-27 20:44:17,151 P51456 INFO Start training: 6910 batches/epoch
2020-12-27 20:44:17,151 P51456 INFO ************ Epoch=1 start ************
2020-12-27 21:19:29,774 P51456 INFO [Metrics] logloss: 0.397372 - AUC: 0.743992
2020-12-27 21:19:29,778 P51456 INFO Save best model: monitor(max): 0.346619
2020-12-27 21:19:29,839 P51456 INFO --- 6910/6910 batches finished ---
2020-12-27 21:19:29,974 P51456 INFO Train loss: 0.420003
2020-12-27 21:19:29,974 P51456 INFO ************ Epoch=1 end ************
2020-12-27 21:54:42,653 P51456 INFO [Metrics] logloss: 0.396507 - AUC: 0.745616
2020-12-27 21:54:42,657 P51456 INFO Save best model: monitor(max): 0.349109
2020-12-27 21:54:42,756 P51456 INFO --- 6910/6910 batches finished ---
2020-12-27 21:54:42,890 P51456 INFO Train loss: 0.420978
2020-12-27 21:54:42,891 P51456 INFO ************ Epoch=2 end ************
2020-12-27 22:30:37,145 P51456 INFO [Metrics] logloss: 0.396295 - AUC: 0.745970
2020-12-27 22:30:37,152 P51456 INFO Save best model: monitor(max): 0.349675
2020-12-27 22:30:37,316 P51456 INFO --- 6910/6910 batches finished ---
2020-12-27 22:30:37,823 P51456 INFO Train loss: 0.422253
2020-12-27 22:30:37,824 P51456 INFO ************ Epoch=3 end ************
2020-12-27 23:06:15,793 P51456 INFO [Metrics] logloss: 0.399075 - AUC: 0.742307
2020-12-27 23:06:15,797 P51456 INFO Monitor(max) STOP: 0.343232 !
2020-12-27 23:06:15,797 P51456 INFO Reduce learning rate on plateau: 0.000100
2020-12-27 23:06:15,797 P51456 INFO --- 6910/6910 batches finished ---
2020-12-27 23:06:15,924 P51456 INFO Train loss: 0.422529
2020-12-27 23:06:15,924 P51456 INFO ************ Epoch=4 end ************
2020-12-27 23:42:17,038 P51456 INFO [Metrics] logloss: 0.396215 - AUC: 0.747274
2020-12-27 23:42:17,063 P51456 INFO Save best model: monitor(max): 0.351059
2020-12-27 23:42:17,240 P51456 INFO --- 6910/6910 batches finished ---
2020-12-27 23:42:17,624 P51456 INFO Train loss: 0.399464
2020-12-27 23:42:17,625 P51456 INFO ************ Epoch=5 end ************
2020-12-28 00:06:06,591 P51456 INFO [Metrics] logloss: 0.396993 - AUC: 0.744173
2020-12-28 00:06:06,596 P51456 INFO Monitor(max) STOP: 0.347180 !
2020-12-28 00:06:06,596 P51456 INFO Reduce learning rate on plateau: 0.000010
2020-12-28 00:06:06,596 P51456 INFO --- 6910/6910 batches finished ---
2020-12-28 00:06:06,701 P51456 INFO Train loss: 0.395828
2020-12-28 00:06:06,701 P51456 INFO ************ Epoch=6 end ************
2020-12-28 00:30:27,469 P51456 INFO [Metrics] logloss: 0.401493 - AUC: 0.737824
2020-12-28 00:30:27,473 P51456 INFO Monitor(max) STOP: 0.336331 !
2020-12-28 00:30:27,473 P51456 INFO Reduce learning rate on plateau: 0.000001
2020-12-28 00:30:27,473 P51456 INFO Early stopping at epoch=7
2020-12-28 00:30:27,473 P51456 INFO --- 6910/6910 batches finished ---
2020-12-28 00:30:27,594 P51456 INFO Train loss: 0.386000
2020-12-28 00:30:27,594 P51456 INFO Training finished.
2020-12-28 00:30:27,594 P51456 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Avazu/DCN_avazu_x0/avazu_x0_83355fc7/DCN_avazu_x0_010_d62fb15d_model.ckpt
2020-12-28 00:30:27,847 P51456 INFO ****** Train/validation evaluation ******
2020-12-28 00:30:43,594 P51456 INFO [Metrics] logloss: 0.396215 - AUC: 0.747274
2020-12-28 00:30:43,696 P51456 INFO ******** Test evaluation ********
2020-12-28 00:30:43,696 P51456 INFO Loading data...
2020-12-28 00:30:43,696 P51456 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/test.h5
2020-12-28 00:30:44,547 P51456 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2020-12-28 00:30:44,547 P51456 INFO Loading test data done.
2020-12-28 00:31:13,852 P51456 INFO [Metrics] logloss: 0.366885 - AUC: 0.764671


```
