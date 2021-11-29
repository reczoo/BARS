## DeepIM_Avazu_x0_001

A notebook to benchmark DeepIM on Avazu_x0_001 dataset.

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
[Metrics] AUC: 0.765154 - logloss: 0.366802
```


### Logs
```python
2021-06-01 10:34:52,037 P33865 INFO {
    "batch_size": "4096",
    "data_format": "csv",
    "data_root": "../data/Avazu/",
    "dataset_id": "avazu_x0_83355fc7",
    "debug": "False",
    "embedding_dim": "10",
    "embedding_dropout": "0",
    "embedding_regularizer": "0.005",
    "epochs": "100",
    "every_x_epochs": "1",
    "feature_cols": "[{'active': True, 'dtype': 'float', 'name': ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 'feat_22'], 'type': 'categorical'}]",
    "gpu": "1",
    "hidden_activations": "relu",
    "hidden_units": "[400, 400, 400]",
    "im_batch_norm": "False",
    "im_order": "5",
    "label_col": "{'dtype': 'float', 'name': 'label'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['AUC', 'logloss']",
    "min_categr_count": "1",
    "model": "DeepIM",
    "model_id": "DeepIM_avazu_x0_002_77e8ae05",
    "model_root": "./Avazu/DeepIM_avazu_x0/",
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
    "test_data": "../data/Avazu/Avazu_x0/test.csv",
    "train_data": "../data/Avazu/Avazu_x0/train.csv",
    "use_hdf5": "True",
    "valid_data": "../data/Avazu/Avazu_x0/valid.csv",
    "verbose": "1",
    "version": "pytorch"
}
2021-06-01 10:34:52,038 P33865 INFO Set up feature encoder...
2021-06-01 10:34:52,038 P33865 INFO Load feature_encoder from pickle: ../data/Avazu/avazu_x0_83355fc7/feature_encoder.pkl
2021-06-01 10:34:52,959 P33865 INFO Total number of parameters: 13398042.
2021-06-01 10:34:52,959 P33865 INFO Loading data...
2021-06-01 10:34:52,961 P33865 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/train.h5
2021-06-01 10:34:56,013 P33865 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/valid.h5
2021-06-01 10:34:56,469 P33865 INFO Train samples: total/28300276, pos/4953382, neg/23346894, ratio/17.50%, blocks/1
2021-06-01 10:34:56,469 P33865 INFO Validation samples: total/4042897, pos/678699, neg/3364198, ratio/16.79%, blocks/1
2021-06-01 10:34:56,469 P33865 INFO Loading train data done.
2021-06-01 10:34:59,991 P33865 INFO Start training: 6910 batches/epoch
2021-06-01 10:34:59,992 P33865 INFO ************ Epoch=1 start ************
2021-06-01 10:47:01,558 P33865 INFO [Metrics] AUC: 0.743980 - logloss: 0.397792
2021-06-01 10:47:01,563 P33865 INFO Save best model: monitor(max): 0.743980
2021-06-01 10:47:01,823 P33865 INFO --- 6910/6910 batches finished ---
2021-06-01 10:47:01,880 P33865 INFO Train loss: 0.427627
2021-06-01 10:47:01,881 P33865 INFO ************ Epoch=1 end ************
2021-06-01 10:59:05,529 P33865 INFO [Metrics] AUC: 0.743622 - logloss: 0.397643
2021-06-01 10:59:05,532 P33865 INFO Monitor(max) STOP: 0.743622 !
2021-06-01 10:59:05,532 P33865 INFO Reduce learning rate on plateau: 0.000100
2021-06-01 10:59:05,533 P33865 INFO --- 6910/6910 batches finished ---
2021-06-01 10:59:05,598 P33865 INFO Train loss: 0.426824
2021-06-01 10:59:05,599 P33865 INFO ************ Epoch=2 end ************
2021-06-01 11:11:08,185 P33865 INFO [Metrics] AUC: 0.747350 - logloss: 0.395801
2021-06-01 11:11:08,188 P33865 INFO Save best model: monitor(max): 0.747350
2021-06-01 11:11:08,309 P33865 INFO --- 6910/6910 batches finished ---
2021-06-01 11:11:08,375 P33865 INFO Train loss: 0.402395
2021-06-01 11:11:08,376 P33865 INFO ************ Epoch=3 end ************
2021-06-01 11:23:11,822 P33865 INFO [Metrics] AUC: 0.746814 - logloss: 0.395883
2021-06-01 11:23:11,824 P33865 INFO Monitor(max) STOP: 0.746814 !
2021-06-01 11:23:11,824 P33865 INFO Reduce learning rate on plateau: 0.000010
2021-06-01 11:23:11,824 P33865 INFO --- 6910/6910 batches finished ---
2021-06-01 11:23:11,915 P33865 INFO Train loss: 0.401848
2021-06-01 11:23:11,915 P33865 INFO ************ Epoch=4 end ************
2021-06-01 11:35:14,843 P33865 INFO [Metrics] AUC: 0.742981 - logloss: 0.398504
2021-06-01 11:35:14,847 P33865 INFO Monitor(max) STOP: 0.742981 !
2021-06-01 11:35:14,847 P33865 INFO Reduce learning rate on plateau: 0.000001
2021-06-01 11:35:14,847 P33865 INFO Early stopping at epoch=5
2021-06-01 11:35:14,847 P33865 INFO --- 6910/6910 batches finished ---
2021-06-01 11:35:14,909 P33865 INFO Train loss: 0.389260
2021-06-01 11:35:14,909 P33865 INFO Training finished.
2021-06-01 11:35:14,909 P33865 INFO Load best model: /home/xxx/xxx/FuxiCTR/benchmarks/Avazu/DeepIM_avazu_x0/avazu_x0_83355fc7/DeepIM_avazu_x0_002_77e8ae05_model.ckpt
2021-06-01 11:35:15,031 P33865 INFO ****** Train/validation evaluation ******
2021-06-01 11:35:29,043 P33865 INFO [Metrics] AUC: 0.747350 - logloss: 0.395801
2021-06-01 11:35:29,091 P33865 INFO ******** Test evaluation ********
2021-06-01 11:35:29,091 P33865 INFO Loading data...
2021-06-01 11:35:29,091 P33865 INFO Loading data from h5: ../data/Avazu/avazu_x0_83355fc7/test.h5
2021-06-01 11:35:30,032 P33865 INFO Test samples: total/8085794, pos/1232985, neg/6852809, ratio/15.25%, blocks/1
2021-06-01 11:35:30,032 P33865 INFO Loading test data done.
2021-06-01 11:35:56,703 P33865 INFO [Metrics] AUC: 0.765154 - logloss: 0.366802

```
